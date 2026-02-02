# Advanced Prediction Features Design

## Overview

Add venue performance tracking, workload windows, and Elo ratings to improve prediction accuracy.

**Total new features: 17**
**MatchComparison::DIM: 33 → 50**

## Feature Specifications

### 1. Venue Performance (5 features)

Track team performance at specific venues.

```rust
pub struct VenueStats {
    pub home_venue_win_rate: f32,      // Home team's historical win rate at this venue
    pub home_venue_games: f32,          // Games played (normalized 0-1)
    pub away_venue_win_rate: f32,       // Away team's record at this venue
    pub away_venue_games: f32,          // Games played (normalized 0-1)
    pub venue_familiarity_diff: f32,    // home_games - away_games (normalized)
}
```

**Dependencies:**
- Requires venue data in match records
- Fix scrapers to capture `span.location` from Wikipedia
- Add Six Nations venue inference (home team → known stadium)

### 2. Workload Windows (9 features)

Approximate player fatigue via match density over multiple time windows.

```rust
pub struct WorkloadFeatures {
    // Match counts per window
    pub home_matches_7d: f32,
    pub home_matches_14d: f32,
    pub home_matches_21d: f32,
    pub away_matches_7d: f32,
    pub away_matches_14d: f32,
    pub away_matches_21d: f32,

    // Differentials (positive = home more fatigued)
    pub workload_diff_7d: f32,
    pub workload_diff_14d: f32,
    pub workload_diff_21d: f32,
}
```

**Notes:**
- Supplements existing 28-day `match_density` feature with finer granularity
- Research suggests 7/14/21 day windows are predictive of performance

### 3. Elo Ratings (3 features)

Team strength ratings computed from match history.

```rust
pub struct EloFeatures {
    pub home_elo: f32,    // Home team rating (normalized)
    pub away_elo: f32,    // Away team rating (normalized)
    pub elo_diff: f32,    // home_elo - away_elo
}
```

**Implementation:**
- Standard Elo with K-factor ~32 for rugby
- Starting rating: 1500
- Home advantage adjustment in expected score calculation
- Computed chronologically from match history

## Scraper Changes

### RawMatch struct (wikipedia.rs)

Add venue field:

```rust
pub struct RawMatch {
    pub date: NaiveDate,
    pub home_team: TeamInfo,
    pub away_team: TeamInfo,
    pub home_score: u8,
    pub away_score: u8,
    pub home_tries: Option<u8>,
    pub away_tries: Option<u8>,
    pub round: Option<u8>,
    pub venue: Option<String>,  // NEW
}
```

### Venue extraction

Update `parse_sports_events()` to extract `span.location`:

```rust
let location_selector = Selector::parse("span.location").unwrap();
let venue = event
    .select(&location_selector)
    .next()
    .map(|loc| loc.text().collect::<String>().trim().to_string());
```

### Six Nations venue inference

Fallback mapping when scraper doesn't capture venue:

| Home Team | Stadium |
|-----------|---------|
| England | Twickenham |
| France | Stade de France |
| Ireland | Aviva Stadium |
| Wales | Principality Stadium |
| Scotland | Murrayfield |
| Italy | Stadio Olimpico |

## New Files

| File | Purpose |
|------|---------|
| `src/features/elo.rs` | Elo rating computation |
| `src/features/venue.rs` | Venue performance tracking |
| `src/features/workload.rs` | Workload window computation |

## Modified Files

| File | Changes |
|------|---------|
| `src/data/scrapers/wikipedia.rs` | Add venue to RawMatch, extract in parsing, pass through to_match_records |
| `src/data/scrapers/sixnations.rs` | Same + venue inference map |
| `src/data/dataset.rs` | Integrate new features into MatchComparison, update DIM to 50 |
| `src/features/mod.rs` | Export new modules |
| `src/features/match_repr.rs` | Update if needed for new features |

## Implementation Order

1. Fix scrapers to capture venue data + add `RawMatch.venue` field
2. Add Six Nations venue inference (team → stadium mapping)
3. Implement Elo rating system
4. Implement workload window features
5. Implement venue performance features
6. Update MatchComparison struct (DIM = 50)
7. Re-scrape/backfill venue data
8. Update tests
9. Validate model still trains correctly

## Testing Strategy

- Unit tests for Elo calculation (known match outcomes → expected ratings)
- Unit tests for workload computation (mock match history)
- Unit tests for venue stats aggregation
- Integration test: full pipeline with new features
- Verify model trains without dimension mismatches
