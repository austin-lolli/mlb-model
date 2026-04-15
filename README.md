# MLB game dataset pipeline

Python pipeline that pulls MLB schedule data, merges **season pitching and team offense statistics**, and writes **model-ready Parquet** files: one row per game where both probable starters match stat rows (**scheduled and final** games included unless you use `--final-only`).

**Layout:** one games file per season: **`data/games_{year}.parquet`** (e.g. `games_2026.parquet`). Pitcher and team stat tables live under **`data/stats_tables/`** as `pitchers_{year}.parquet` and `team_batting_{year}.parquet`. The old **`games_live.parquet`** name is retired; use `games_{PIPELINE_SEASON or calendar year}.parquet` instead.

---

## Quick start (CLI subcommands)

From this directory (`mlb-model`), set `PYTHONPATH` once so `import app` works:

**Windows (PowerShell)**

```powershell
$env:PYTHONPATH="."
```

**Linux / macOS (bash or zsh)**

```bash
export PYTHONPATH=.
```

### `live` — sync the current season (default daily / Docker)

Writes **`data/games_{season}.parquet`** and refreshes **`data/stats_tables/pitchers_{season}.parquet`** and **`team_batting_{season}.parquet`**. Season defaults to **`PIPELINE_SEASON`** env, else **calendar year** (useful in January / off-season to set `PIPELINE_SEASON` explicitly).

```powershell
python -m app.main live
python -m app.main live --season 2026
```

```bash
python -m app.main live
python -m app.main live --season 2026
```

Platoon, rolling SP, and bullpen features are **on** by default. Kalshi stays **off** unless you pass **`--kalshi`**.

### `backfill` — one-off historical seasons (new machine / rare)

One **games** file **per** year (no single concatenated multi-year Parquet):

```powershell
python -m app.main backfill --years 2024 2025 2026
```

```bash
python -m app.main backfill --years 2024 2025 2026
```

### `run` — custom date window (escape hatch)

```powershell
python -m app.main run --start 2024-04-01 --end 2024-04-15 --season 2024 --out data/sample.parquet
```

```bash
python -m app.main run --start 2024-04-01 --end 2024-04-15 --season 2024 --out data/sample.parquet
```

Optional **`--export-stats-dir DIR`** on `run` only. **`--stats-dir`** on `live` / `backfill` defaults to **`data/stats_tables`**.

**Training-style labels:** add **`--final-only`** on any subcommand.

**Caches:** rolling and bullpen use **`data/cache/`** for MLB game logs; first runs are slower.

**Deprecated (still works with a warning):** top-level **`--years`** / **`--start`** without a subcommand is treated as **`backfill`** or **`run`**.

**Jupyter:** `pd.read_parquet("data/games_2026.parquet")` (pick the season file you care about).

Platoon splits now come from the MLB Stats API (`statSplits`, team hitting vs RHP/LHP). Use **`--no-platoon-br`** to skip platoon columns if needed.

---

## Data sources

| Layer | Source | Role |
|--------|--------|------|
| Schedule, scores, probables | [MLB Stats API](https://statsapi.mlb.com/) `GET /api/v1/schedule` and `GET /api/v1/people` | Game backbone: teams, scores, `detailedState`, hydrated **probable pitcher** names, **linescore**, and **pitch hand** (`home_sp_throws` / `away_sp_throws`: R/L). |
| Pitcher stats (default) | [MLB Stats API](https://statsapi.mlb.com/) `GET /api/v1/stats` (season, pitching, `playerPool=all`) | Per-pitcher season totals → **K%**, **BB%**, **FIP** (stored under `xFIP` column for schema compatibility), **kbb**. |
| Team offense (default) | MLB Stats API `GET /api/v1/teams/{id}/stats` (season, hitting) | Per-team season **OPS** → scaled to an index in the **`wRC+` column** (see below). |
| Platoon offense (optional) | MLB Stats API `GET /api/v1/teams/{id}/stats` (`stats=statSplits`, `group=hitting`, `sitCodes=vr` / `vl`) — one request per team per split (`playerPool=team` is invalid on `/stats` for `statSplits`) | Per-team **OPS vs LHP / vs RHP**, scaled to the same index style → `home_offense_platoon` / `away_offense_platoon` vs opponent SP handedness. |
| Rolling SP metrics | Cached per-pitcher **game logs** from MLB stats (`pitcher_gamelog` → `rolling.py`) | **K−BB** and **FIP-style** stats over the **14 calendar days before** `game_date` (exclusive of game day). |
| Bullpen A + B | Team reliever IDs (`fetch_team_pitching_mlb`) + same game logs | **Season-to-date** and **14-day** relief-only FIP/ERA (non-start appearances), as-of `game_date`. |
| Kalshi (optional) | [Kalshi Trade API](https://docs.kalshi.com/) public markets | `kalshi_home_implied` / `kalshi_away_implied` from `game_pk` -> ticker map. The pipeline now auto-populates forward-looking MLB game mappings from Kalshi `KXMLBGAME` markets. |
| Pitcher / team stats (optional) | FanGraphs via **pybaseball** (`pitching_stats`, `team_batting`) | Used only if `USE_FANGRAPHS=1`. Provides true **xFIP** and FanGraphs **wRC+** when the site is reachable. |

**Default behavior:** `USE_FANGRAPHS` is **off** (unset or false). FanGraphs often returns HTTP 403 to automated clients; the MLB API path avoids that and is what most runs use today.

---

## Stat definitions (default path: MLB API)

These feed the feature columns in the output Parquet.

### Pitcher-level (matched by probable starter name)

- **K%** — `100 × strikeOuts / battersFaced` for that pitcher’s **season** (same scale as “percent”: e.g. 25 means 25%).
- **BB%** — `100 × baseOnBalls / battersFaced`.
- **kbb** — `K% − BB%` (**not** the K/BB ratio). Typical good starters land in the teens–twenties on this scale.
- **xFIP (column name)** — In the default path this holds **FIP**, not FanGraphs xFIP:  
  `(13×HR + 3×(BB+HBP) − 2×K) / IP + 3.10` with IP parsed from `inningsPitched`. The column name is unchanged so downstream code stays stable.

### Team-level (matched by team)

- **wRC+ (column name)** — In the default path this is **not** FanGraphs wRC+. It is **team OPS relative to league-average OPS**, scaled so **100 ≈ league average**:  
  `100 × (team_OPS / mean_OPS_across_30_teams)`.  
  Same column name as the FanGraphs path for one unified schema.

---

## Output Parquet: columns and meaning

Each row is one game **after** merging stats, subject to filters below. With **default** options, **scheduled** games appear too: there may be no scores yet.

Games are **dropped** if team mapping fails, **either** probable pitcher fails **name match**, or pitcher stat fields are missing. Scores are **not** required to keep a row (unless you use `--final-only`, which restricts to finished games only).

| Column | Meaning |
|--------|---------|
| `game_pk` | MLB stable game id (from schedule). |
| `game_date` | Calendar date of the game (schedule). |
| `detailed_state` | MLB status string (e.g. `Final`, `Scheduled`, `In Progress`). |
| `home_team_name` / `away_team_name` | Full team names from the schedule API. |
| `home_score` / `away_score` | Runs when known; null for not-yet-played games. |
| `home_probable_pitcher` / `away_probable_pitcher` | Probable starter **full name** at fetch time (nullable). |
| `stats_season` | Present when built with **`live`** or **`backfill`**: integer year used for **that row’s** pitcher/team stat tables. |
| `home_wrc_plus` / `away_wrc_plus` | Team offense index from the **`wRC+` column** of the team batting frame (OPS index or true wRC+ depending on source). |
| `home_sp_kbb` / `away_sp_kbb` | Matched probable pitcher’s **`kbb`** = **K% − BB%** (percentage-point scale). |
| `home_sp_xfip` / `away_sp_xfip` | Matched pitcher’s **`xFIP` column** — true **xFIP** if FanGraphs; **FIP** if MLB default path. |
| `home_win` | **1** / **0** only when the game is **finished** (`Final`, `Game Over`, `Completed Early`) **and** both scores exist. **`NaN`** for scheduled, live, postponed, etc. (even if the API shows a partial line score). |
| `sp_kbb_diff` | `home_sp_kbb − away_sp_kbb`. |
| `sp_xfip_diff` | `home_sp_xfip − away_sp_xfip` (lower xFIP/FIP is better for pitchers, so sign matters for interpretation). |
| `offense_diff` | `home_wrc_plus − away_wrc_plus`. |
| `home_probable_pitcher_id` / `away_probable_pitcher_id` | MLB `personId` for the probable starter when present (nullable). |
| `home_team_id` / `away_team_id` | MLB team ids from the schedule (used for bullpen aggregation). |
| `home_sp_throws` / `away_sp_throws` | Probable starter **throws** R or L from `/api/v1/people` (`pitchHand`). |
| `home_offense_platoon` / `away_offense_platoon` | Team offense index vs **platoon split** matching the **opposing** SP handedness (MLB `statSplits`; **NaN** if platoon fetch skipped or failed). |
| `offense_platoon_diff` | `home_offense_platoon − away_offense_platoon`. |
| `home_sp_kbb_roll14` / `away_sp_kbb_roll14` | Rolling **K% − BB%** over the prior **14 calendar days** before `game_date` (same scale as season `kbb`; **NaN** if no qualifying games). |
| `home_sp_xfip_roll14` / `away_sp_xfip_roll14` | Rolling **FIP-style** number over that same window (MLB game-log reconstruction). |
| `home_pen_season_fip` / `away_pen_season_fip` | Bullpen **A**: relief-only **FIP** from season start through the day **before** `game_date`. |
| `home_pen_season_era` / `away_pen_season_era` | Bullpen **A**: relief-only **ERA** over the same season-to-date window. |
| `home_pen_roll14_fip` / `away_pen_roll14_fip` | Bullpen **B**: relief-only **FIP** using appearances in the **14 calendar days** before `game_date`. |
| `home_pen_roll14_era` / `away_pen_roll14_era` | Bullpen **B**: relief-only **ERA** over that rolling window. |
| `home_pen_roll14_minus_season_fip` / `away_pen_roll14_minus_season_fip` | Recent minus season FIP (convenience feature; **NaN** if either side missing). |
| `kalshi_home_implied` / `kalshi_away_implied` | Kalshi **mid** implied probability (0–1) for home/away win from mapped tickers; **NaN** without map or API. |
| `edge_vs_model` | Placeholder **NaN** until you join model probability vs market. |

**As-of / leakage:** Rolling and bullpen windows use only games with `game_date` **strictly before** the row’s `game_date`. Season bullpen stats start at Jan 1 of that season and end before game day. Platoon splits are **season** splits from MLB `statSplits` (not re-cut daily); use for research awareness. Kalshi prices are **fetch-time** snapshots, not historical closes unless you run the pipeline at game time and cache.

### CLI flags: game rows

| Flag | Effect |
|------|--------|
| *(omit)* | Include non-final games; `home_win` is **NaN** until status is **final** (scheduled / live / postponed stay NaN). |
| `--final-only` | Keep only completed games (`Final` / `Game Over` / `Completed Early`); `home_win` is always **0** or **1**. |

### CLI flags: extended features (all **on** by default)

| Flag | Effect |
|------|--------|
| `--no-platoon-br` | Skip platoon columns (`home_offense_platoon`, `away_offense_platoon`, `offense_platoon_diff`). |
| `--no-rolling` | Skip rolling 14-day SP `kbb` / FIP-style columns. |
| `--no-bullpen` | Skip bullpen season and 14-day relief columns. |
| `--kalshi` | Enable Kalshi API calls for mapped games (off by default; implied columns stay **NaN** otherwise). |
| `--kalshi-map CSV` | Path to `game_pk` ↔ `kalshi_ticker_home` CSV (default `data/kalshi_game_map.csv`). Existing rows are honored first; missing current/future rows are auto-filled from open Kalshi MLB game markets when possible. |
| `--no-last-run` | Do not write `data/last_run.json` after a successful run. |

---

## Exported stat tables (`data/stats_tables/`)

**`live`** and **`backfill`** always write **full season** pitcher and team batting tables (same schema as the in-memory merge inputs):

| File | Contents |
|------|----------|
| `pitchers_{season}.parquet` | Columns `Name`, `Team` (FG-style abbrev from MLB team name when using the API path), `K%`, `BB%`, `kbb`, `xFIP`, `id`, `gamesStarted`, `gamesPlayed`, `is_reliever_season` (True when season GS/G is below the reliever threshold and G ≥ 3). See [Stat definitions](#stat-definitions-default-path-mlb-api). |
| `team_batting_{season}.parquet` | Columns `Team`, `wRC+` (OPS index or FG wRC+ depending on source). |

Override directory with **`--stats-dir`** on `live` / `backfill`. The **`run`** subcommand only exports when **`--export-stats-dir`** is set.

---

## How rows are built (high level)

1. **Schedule** — Pull games in the date window; keep probables, scores, team ids, and **SP pitch hands** (extra `people` request).
2. **Pitcher table** — One row per pitcher for the chosen **season** with `Name`, `K%`, `BB%`, `kbb`, `xFIP`.
3. **Team table** — One row per team (`Team` abbrev aligned to internal mapping) with `wRC+`.
4. **Name match** — Map `home_probable_pitcher` / `away_probable_pitcher` strings to `Name` in the pitcher table: normalized **exact** match first, else **rapidfuzz** `token_sort_ratio` ≥ 90.
5. **Team match** — Map schedule team names to the same abbreviations used in the team table (`config.py` maps MLB names → codes; MLB API abbrev quirks like `SD` → `SDP` are normalized for joins).
6. **Extended merges (optional)** — Platoon table by team (MLB `statSplits`), per-game rolling SP stats and bullpen aggregates from **cached** MLB game logs under `data/cache/`. Kalshi implieds only if **`--kalshi`**; for current/future games the pipeline attempts to auto-map `game_pk` to `KXMLBGAME` home-win tickers and persists to `data/kalshi_game_map.csv`.
7. **Game filter** — With **`--final-only`**, restrict to completed games. Otherwise **all** schedule rows in range are candidates; `home_win` is set only when both scores exist.
8. **Stat table export** — **`live`** / **`backfill`** always write pitcher and team batting Parquet under **`--stats-dir`** (default `data/stats_tables`). **`run`** exports only if **`--export-stats-dir`** is passed.
9. **Observability** — Unless **`--no-last-run`**, write **`data/last_run.json`** (`subcommand`, `output_paths`, per-file row counts, flags).

---

## Environment variables

| Variable | Effect |
|----------|--------|
| `USE_FANGRAPHS=1` | Try FanGraphs first for pitcher/team stats (true xFIP, FG wRC+). Falls back to MLB API on HTTP errors. |
| *(unset / false)* | MLB Stats API only for pitcher and team stats (default). |
| `PITCHER_PRIOR_YEAR_FALLBACK=0` | Do not append last season’s pitcher rows for names missing from the current year (default is **on** so early-season probables can still match). |
| `KALSHI_API_BASE` | Override Kalshi Trade API base URL (default `https://api.elections.kalshi.com/trade-api/v2`). |
| `PIPELINE_SEASON` | Integer year for **`live`** when `--season` is omitted (defaults to calendar year if unset). |
| `LIVE_INTERVAL_SEC` | Seconds between **`live`** runs in Docker Compose **`live`** service (default **3600**). |
| `GAMELOG_FETCH_DELAY_SEC` | Seconds to sleep before each **network** game-log fetch on cache miss (default `0`). Raise slightly (e.g. `0.08`) if MLB returns HTTP 429 during large backfills. |
| `GAMELOG_LOG_EVERY` | If set to a positive integer, log at INFO every N **network** game-log fetches (default `0` = off). |

**Linux / macOS — prefix a one-shot command with env vars:**

```bash
PIPELINE_SEASON=2026 USE_FANGRAPHS=1 python -m app.main live --season 2026
```

---

## Docker

`docker-compose.yml` mounts **`./data`** to **`/app/data`** so Parquet outputs and **`data/cache/`** persist on the host.

Use **`docker compose`** (V2 plugin, recommended) or the older standalone **`docker-compose`** — same flags below; only the executable name changes.

**Persistent live service (hourly current season):**

```bash
docker compose --profile live up --build live
```

Foreground attach is the default. To run detached (background) on Linux / macOS:

```bash
docker compose --profile live up --build -d live
```

Runs **`python -m app.main live`** in a loop, then sleeps **`LIVE_INTERVAL_SEC`** (default **3600**). Set **`PIPELINE_SEASON`** if the calendar year is wrong for MLB (e.g. January). Default image **`CMD`** is also **`live`** for one-shot container runs.

**Inline env (Linux / macOS) for `live`:**

```bash
PIPELINE_SEASON=2026 LIVE_INTERVAL_SEC=7200 docker compose --profile live up --build live
```

**PowerShell (Windows) equivalent:**

```powershell
$env:PIPELINE_SEASON="2026"; $env:LIVE_INTERVAL_SEC="7200"; docker compose --profile live up --build live
```

**One-off backfill (then exit):**

```bash
docker compose run --rm app python -m app.main backfill --years 2024 2025
```

**Other profiles** (e.g. refresh): same pattern — `docker compose --profile refresh up --build refresh` (see `docker-compose.yml`).

**Rebuild after code changes** (from `mlb-model`):

```bash
docker compose build --no-cache
```

Bullpen logs should mention **playerPool=all** (not per-team **`playerPool=team`**).

---

## Future work

- **Statcast** — Stub in `app/data/statcast.py` for later pitch-level features (e.g. rolling xwOBA by pitcher and date).
- **Storage** — Parquet now; SQLite/Postgres possible later.

Rolling-window sanity checks live in `tests/test_rolling_window.py`:

```bash
python -m unittest discover -s tests
```

---

## Local Parquet files vs notebooks

The repo may accumulate **extra** Parquet files under **`data/`** from experiments, old backfills, or manual exports. The **curated notebooks under `notebooks/`** (`games_today_tomorrow.ipynb`, `pitcher_match_analysis.ipynb`, `pitcher_relief_by_team.ipynb`) only load paths of this shape:

- **`data/games_{season}.parquet`** (season from `PIPELINE_SEASON` or calendar year), plus legacy **`data/games.parquet`** / **`data/games_live.parquet`** as fallbacks in the games notebook.
- **`data/stats_tables/pitchers_{season}.parquet`** in `pitcher_relief_by_team.ipynb`.

Root-level notebooks (e.g. `MismatchBreakdown.ipynb`, `Playground.ipynb`, `Untitled*.ipynb`) may reference **`data/games_2024_2025_2026.parquet`** and **`data/stats_tables/pitchers_20xx.parquet`** / **`team_batting_20xx.parquet`** — keep those if you still use those notebooks.

**Typically safe to delete** if you rely only on `notebooks/` and the standard pipeline outputs (and you have refreshed **`games_{current_season}.parquet`** elsewhere):

| File (under `data/`) | Notes |
|----------------------|--------|
| `games_2024.parquet`, `games_2025.parquet` | Per-year games outputs; not referenced by `notebooks/*.ipynb` by name (your season file is usually `games_2026.parquet` or similar). |
| `games_2024_2025.parquet` | Ad hoc multi-year export; not used by `notebooks/`. |
| `games_2024_analysis.parquet` | One-off analysis export; not referenced in tracked notebooks. |
| `smoke.parquet`, `smoke_test.parquet` | Local / manual test artifacts; not loaded by notebooks in `notebooks/`. |
| `stats_export_test/pitchers_2026.parquet`, `stats_export_test/team_batting_2026.parquet` | Test export directory; not used by notebooks. |

**Do not delete** if you need them: **`games_2026.parquet`** (or whatever **`games_{PIPELINE_SEASON}`** is), **`games_2024_2025_2026.parquet`** (if you use root analysis notebooks), **`stats_tables/*.parquet`**, and any **`games.parquet` / `games_live.parquet`** you still use as fallbacks.

When in doubt, search the repo: `rg read_parquet` in `*.ipynb`.

---

## Requirements

See `requirements.txt` (`requests`, `pandas`, `pyarrow`, `pybaseball`, `rapidfuzz`). Python 3.11+ is assumed (see `Dockerfile`).
