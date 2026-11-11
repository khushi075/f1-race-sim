# F1 Race Simulator

Simulate and predict Formula 1 races with current and past drivers, using customisable cars, tracks, weather and teammates, and find which driver pairings suit which team.

The core ML problem is **separating driver skill from car performance**: drivers only race a few cars, and good drivers tend to end up in good cars. Teammate comparisons are the main evidence, since teammates share (nearly) the same car.

## Status

| Feature | Status |
|---|---|
| Qualifying data ingestion, 2018–2025 (FastF1) | ✅ |
| Teammate qualifying-gap analysis | ✅ |
| Race-pace gaps (clean laps, same tyre compound) | Planned |
| Baseline finishing-position model | Planned |
| Hierarchical driver + car model | Planned |
| Monte Carlo race simulator | Planned |
| Teammate-pairing search | Planned |

## Feature 1: teammate qualifying gaps

For each race, the two teammates are compared in the **last qualifying segment they both set a time in** (Q3 if both reached it, else Q2, else Q1). The gap is a % of the teammate's lap time, where negative means faster. Gaps over 2% (adjustable) are flagged as likely crashes or technical issues and excluded from season summaries.

**Limitation:** teammates' cars are not identical (setups, upgrade timing, reliability), so this is an approximation.

## Project layout

```
f1sim/
  paths.py      # where data lives (Google Drive in Colab)
  ingest.py     # FastF1 download: resumable, one file per session
  manual.py     # loads hand-entered times for sessions no API covers
  features.py   # teammate gap calculations
manual_data/
  quali_times.csv  # hand-entered qualifying times (see Data sources)
notebooks/
  00_ingest_quali.ipynb        # download data
  01_teammate_quali_gaps.ipynb # analysis and charts
```

Data is stored outside the repo (on Google Drive) and is never committed.

## Running in Google Colab

1. Open `notebooks/00_ingest_quali.ipynb` in Colab: **File → Open notebook → GitHub**, then paste the repo URL.
2. In the setup cell, set `REPO_URL` to this repo and run it. Approve the Google Drive mount.
3. Run the download cell. FastF1 has hard API limits (200 calls/hour to Jolpica, 500/hour overall), so all of 2018–2025 takes several runs. When it stops at the limit, wait about an hour and rerun; finished sessions are skipped.
4. Open `notebooks/01_teammate_quali_gaps.ipynb`, run the setup cell, then the analysis cells. Use the form fields to change season, team and driver.

If the repo is private, cloning in Colab needs a GitHub token. Keeping it public is simpler, and it's a portfolio project anyway.

## Running locally (optional)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m f1sim.ingest --start 2018 --end 2025 --data-dir ./data_store
```

## Data sources

- [FastF1](https://docs.fastf1.dev): F1 live timing data and the Jolpica-F1 API (successor to Ergast)
- `manual_data/quali_times.csv`: qualifying times copied from the official [formula1.com](https://www.formula1.com) results, for sessions where no API has times. Driver and team identity still comes from FastF1; rows are matched by car number, and ingestion fails loudly if the drivers don't match.

Each saved session records where its times came from in `QTimeSource`: `results_api`, `computed_from_laps` or `manual`.

### Known data issues

| Session | Issue | Handling |
|---|---|---|
| 2025 R06 Miami | F1 live timing has no data; Jolpica has the order but no times | Times entered manually from formula1.com |

Team names differ between sources (e.g. "Red Bull Racing" vs "Red Bull"), so the analysis groups teams by Jolpica's `TeamId`.
