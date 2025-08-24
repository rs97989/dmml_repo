# 8_DATA_VERSIONING/auto_commit_lfs.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import fnmatch, re, subprocess, sys

REPO = Path(__file__).resolve().parent.parent
GITATTR = REPO / ".gitattributes"

def sh(cmd: list[str], check=True):
    try:
        return subprocess.run(
            cmd, cwd=str(REPO), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check
        )
    except subprocess.CalledProcessError as e:
        print("\n[CMD FAILED]", " ".join(cmd))
        print(e.stdout)
        raise

def ensure_repo():
    ok = sh(["git", "rev-parse", "--is-inside-work-tree"]).stdout.strip()
    if ok != "true":
        sys.exit(f"Not inside a Git repo. REPO={REPO}")

def ensure_git_identity():
    if not sh(["git", "config", "--get", "user.email"], check=False).stdout.strip():
        sh(["git", "config", "user.email", "data-bot@example.com"], check=False)
    if not sh(["git", "config", "--get", "user.name"], check=False).stdout.strip():
        sh(["git", "config", "user.name", "Data Bot"], check=False)

def load_lfs_patterns() -> list[str]:
    pats: list[str] = []
    if GITATTR.exists():
        for raw in GITATTR.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts and "filter=lfs" in line:
                pats.append(parts[0].strip("'").strip('"'))
    # de-dup preserve order
    seen, out = set(), []
    for p in pats:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

def lfs_tracked_files() -> set[str]:
    cp = sh(["git", "lfs", "ls-files", "-n"], check=False)
    return {ln.strip().replace("\\", "/") for ln in cp.stdout.splitlines() if ln.strip()}

def changed_status_entries():
    """
    Return list of (statusXY, path) from 'git status --porcelain -z'.
    For renames, we return the NEW path with the same status.
    """
    cp = sh(["git", "status", "--porcelain=1", "-z", "--untracked-files=all"])
    items = cp.stdout.split("\0")
    out, i = [], 0
    while i < len(items) and items[i] != "":
        entry = items[i]
        status = entry[:2]
        path1  = entry[3:] if len(entry) > 3 else ""
        if status.startswith("R"):
            i += 1
            newp = items[i] if i < len(items) else ""
            if newp:
                out.append((status, newp.replace("\\", "/")))
        else:
            if path1:
                out.append((status, path1.replace("\\", "/")))
        i += 1
    return out

def filter_lfs_only(paths: list[str], patterns: list[str], already_lfs: set[str]) -> list[str]:
    keep = []
    for p in paths:
        if p in already_lfs or any(fnmatch.fnmatchcase(p, pat) for pat in patterns):
            keep.append(p)
    return keep

def utc_now():
    human = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tag   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%MZ")  # colon-free
    return human, tag

def current_branch() -> str:
    b = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"], check=False).stdout.strip()
    return b or "main"

def github_web_url(remote: str) -> str:
    remote = re.sub(r"https://[^@]+@", "https://", remote)
    if remote.endswith(".git"): remote = remote[:-4]
    if remote.startswith("git@github.com:"):
        return "https://github.com/" + remote.split(":", 1)[1]
    return remote

def main():
    ensure_repo()
    ensure_git_identity()

    patterns = load_lfs_patterns()
    if not patterns:
        sys.exit("No LFS patterns found in .gitattributes.")

    changes = changed_status_entries()
    already = lfs_tracked_files()

    # Split changed LFS paths into present vs deleted
    present, deleted = [], []
    for status, p in changes:
        # keep only LFS-tracked files (already in LFS or matching a pattern)
        if p not in already and not any(fnmatch.fnmatchcase(p, pat) for pat in patterns):
            continue
        # porcelain XY: deleted in worktree is ' D'; deleted in index is 'D '
        if status.strip() == "D" or status == " D" or status == "D ":
            deleted.append(p)
        else:
            present.append(p)

    if not present and not deleted:
        print("No changes in LFS-tracked files.")
        return

    print("LFS-tracked changes:")
    for p in present: print("  ✓ modify/add:", p)
    for p in deleted: print("  ✖ delete     :", p)

    # Stage modifications/additions
    if present:
        CHUNK = 200
        for i in range(0, len(present), CHUNK):
            sh(["git", "add", "-f", "--", *present[i:i+CHUNK]])

    # Stage deletions (use rm even if file already missing)
    for p in deleted:
        sh(["git", "rm", "-f", "--", p])

    ts_human, ts_tag = utc_now()
    msg = f"Auto LFS commit @ {ts_human}"
    cp = sh(["git", "commit", "-m", msg], check=False)
    if "nothing to commit" in cp.stdout.lower():
        print("Nothing to commit after staging.")
        return

    commit_hash = sh(["git", "rev-parse", "HEAD"]).stdout.strip()
    print(f"Committed {commit_hash[:8]}: {msg}")

    tag = f"data-v{ts_tag}"
    sh(["git", "tag", "-a", tag, "-m", msg])
    print(f"Created tag: {tag}")

    # Push commit + tag + LFS objects
    sh(["git", "push"], check=False)
    sh(["git", "push", "--tags"], check=False)
    sh(["git", "lfs", "push", "origin", current_branch()], check=False)

    remote = sh(["git", "remote", "get-url", "origin"], check=False).stdout.strip()
    if remote:
        web = github_web_url(remote)
        print("\nView on GitHub:")
        print("Repo:   ", web)
        print("Branch: ", f"{web}/tree/{current_branch()}")
        print("Commit: ", f"{web}/commit/{commit_hash}")
        print("Tag:    ", f"{web}/tree/{tag}")

if __name__ == "__main__":
    main()
