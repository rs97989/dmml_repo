# Feature Store Documentation

_generated: 2025-08-23T20:53:28.338030Z_

**DB:** `F:\Assignment_DMML\DMML_REPO\6_DATA_TRANSFORMATION_AND_STORAGE\feature_store.db`

## Feature Groups

- **churn_all v1** — pk=`customer_id`, label=`churned`, features=48

- **churn_v1 v1** — pk=`customer_id`, label=`churned`, features=48


## Example: churn_v1 v1 membership (first 25)

- customer_id

- churned

- source

- age

- average_session_length

- notifications_clicked

- num_favorite_artists

- num_platform_friends

- num_playlists_created

- num_shared_playlists

- num_subscription_pauses

- signup_date

- song_skip_rate

- weekly_hours

- weekly_songs_played

- weekly_unique_songs

- customer_service_inquiries_high

- customer_service_inquiries_low

- customer_service_inquiries_medium

- location_alabama

- location_california

- location_florida

- location_georgia

- location_idaho

- location_maine


## Files

- `features.csv` — feature metadata + live stats

- `feature_groups.csv` — groups and versions

- `group_membership.csv` — ordered features per group/version
