# F1 Race Simulator

Simulate and predict Formula 1 races with current and past drivers, using customisable cars, tracks, weather and teammates, and find which driver pairings suit which team.

The core ML problem is **separating driver skill from car performance**: drivers only race a few cars, and good drivers tend to end up in good cars. Teammate comparisons are the main evidence, since teammates share (nearly) the same car.

## Status

| Feature | Status |
|---|---|
| Qualifying data ingestion, 2018–2025 (FastF1) | ✅ |
| Teammate qualifying-gap analysis | ✅ |
| Race laps, weather and race control, 2018–2025 (TracingInsights) | ✅ |
| Teammate race-pace model (dry races) | ✅ |
| Baseline finishing-position model | Planned |
| Hierarchical driver + car model | Planned |
| Monte Carlo race simulator | Planned |
| Teammate-pairing search | Planned |

## Feature 1: teammate qualifying gaps

For each race, the two teammates are compared in the **last qualifying segment they both set a time in** (Q3 if both reached it, else Q2, else Q1). The gap is a % of the teammate's lap time, where negative means faster. Gaps over 2% (adjustable) are flagged as likely crashes or technical issues and excluded from season summaries.

**Limitation:** teammates' cars are not identical (setups, upgrade timing, reliability), so this is an approximation.

## Feature 2: teammate race pace

Race pace is compared with a small lap-time model fitted separately for each race:

`lap_time ~ driver + compound + compound × tyre_age + lap_number`

It adjusts each lap for tyre compound, tyre wear (each compound degrading at its own rate) and lap number (fuel burn-off and track grip). The teammate gap is the difference between the two drivers' estimated pace with those held equal, with a standard error from the model. A robust (Huber) regression is used by default, so laps lost in traffic count less.

- **Clean laps:** only laps FastF1 marks as accurate (no pit in/out laps, no safety car, VSC or red-flag laps, consistent timing).
- **Wet races:** excluded for now, defined as any race where intermediate or wet tyres were used (adjustable to also exclude races with any rain recorded).
- **Minimum laps:** a driver needs at least 10 clean laps in a race to be compared.
- **Identifiability:** laps on a compound only one driver used in a race are dropped (that driver's pace and the tyre's can't be separated), and a race is skipped if the model still can't separate every effect. Both are listed in the notebook, with reasons.

**Limitations:** the fuel and track-evolution effect is assumed linear and the same for every car; drivers in different traffic or running different strategies may still differ in ways the model doesn't capture.

## Project layout

```
f1sim/
  paths.py      # where data lives (Google Drive in Colab)
  ingest.py     # qualifying download: resumable, one file per session
  ingest_tracing.py # race laps, weather, race control from TracingInsights (free)
  race_pace.py  # clean laps, wet races, per-race lap-time model, teammate gaps
  manual.py     # loads hand-entered times for sessions no API covers
  features.py   # teammate gap calculations
manual_data/
  quali_times.csv  # hand-entered qualifying times (see Data sources)
notebooks/
  00_ingest_quali.ipynb        # download data
  01_teammate_quali_gaps.ipynb # analysis and charts
  03_ingest_race_tracinginsights.ipynb # download race laps, weather, race control
  04_teammate_race_pace.ipynb  # race-pace model, charts, quali vs race
```

Data is stored outside the repo (on Google Drive) and is never committed.

## Running in Google Colab

1. Open `notebooks/00_ingest_quali.ipynb` in Colab: **File → Open notebook → GitHub**, then paste the repo URL.
2. In the setup cell, set `REPO_URL` to this repo and run it. Approve the Google Drive mount.
3. Run the download cell. FastF1 has hard API limits (200 calls/hour to Jolpica, 500/hour overall), so all of 2018–2025 takes several runs. When it stops at the limit, wait about an hour and rerun; finished sessions are skipped.
4. Open `notebooks/01_teammate_quali_gaps.ipynb`, run the setup cell, then the analysis cells. Use the form fields to change season, team and driver.
5. For race data, run `notebooks/03_ingest_race_tracinginsights.ipynb` the same way. One run covers all seasons; there are no rate limits.
6. Open `notebooks/04_teammate_race_pace.ipynb` for the race-pace analysis.

If the repo is private, cloning in Colab needs a GitHub token. Keeping it public is simpler, and it's a portfolio project anyway.

## Running locally (optional)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m f1sim.ingest --start 2018 --end 2025 --data-dir ./data_store
```

## Data sources

- [FastF1](https://docs.fastf1.dev): F1 live timing data and the Jolpica-F1 API (successor to Ergast). Qualifying times come from Jolpica.
- [TracingInsights](https://github.com/TracingInsights): race laps, weather and race control messages, from their public per-season repos (e.g. `TracingInsights/2025`), licensed [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). The data is FastF1-derived. Every row stores the git commit it came from (`SourceCommit`).
- `manual_data/quali_times.csv`: qualifying times copied from the official [formula1.com](https://www.formula1.com) results, for sessions where no API has times. Driver and team identity still comes from FastF1; rows are matched by car number, and ingestion fails loudly if the drivers don't match.

Each saved session records where its times came from in `QTimeSource`: `results_api`, `computed_from_laps` or `manual`.

### Known data issues

| Session | Issue | Handling |
|---|---|---|
| 2025 R06 Miami | F1 live timing has no data; Jolpica has the order but no times | Times entered manually from formula1.com |
| All races | Since October 2026, F1's timing APIs require an F1 TV login, so FastF1 can't download lap data without a subscription | Race data from TracingInsights instead |
| 2018 R01 Australia | TracingInsights has laps but no weather or race control file | Laps used; no weather/race control for this race |

Team names differ between sources (e.g. "Red Bull Racing" vs "Red Bull"), so the analysis groups teams by Jolpica's `TeamId`.
