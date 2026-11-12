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
| Race results 2018–2025 (Jolpica) | ✅ |
| Baseline model: DNF chance + finishing order | ✅ |
| Hierarchical driver skill + car pace model (PyMC) | ✅ |
| Monte Carlo race simulator (dry races) | ✅ |
| Out-of-sample test of the simulator (2025) | ✅ |
| Teammate-pairing search | Planned |

## Results and conclusions

All numbers come from the notebooks run on 2018–2025 data: 173 races, 152 of them dry. Test results are for 2025, using models that hadn't seen the races they predict.

### What the results show

1. **Free data is enough.** Since October 2026, F1's timing APIs need an F1 TV login. Qualifying times and race results still come free from Jolpica, and every race lap (with tyres, track status and weather) from TracingInsights' public archive. One session (2025 Miami qualifying) had no times in any API and was entered by hand from formula1.com.

2. **Teammate gaps are a clean, readable signal.** In 2025 qualifying, Verstappen beat Tsunoda 20–0 by a median 0.68%, while Norris and Piastri were nearly level (14–10, 0.04%). The per-race lap model turns 144,541 clean dry laps into 1,360 race-pace teammate comparisons. Its typical uncertainty is about 0.1 s per lap, so close gaps can be told apart from noise.

3. **Grid position explains most of a race result, and retirements are close to random.**

   | Baseline, 2025 test (finishers) | Spearman | Avg error (positions) | Exactly right |
   |---|---|---|---|
   | Finish in grid order | 0.755 | 2.34 | 24.3% |
   | Linear model | 0.791 | 2.23 | 21.2% |
   | Gradient boosting | 0.776 | 2.34 | 20.0% |

   The linear model beats grid order modestly, and gradient boosting adds nothing: with about 3,000 training rows dominated by grid and form, extra flexibility fits noise. Pre-race information barely predicts DNFs (AUC 0.56; log-loss 0.378 vs 0.380 for a flat rate).

4. **The driver + car model gives believable rankings, with one known blind spot.** For 2025 it puts Verstappen far ahead (−0.57% of a lap vs the average driver; 90% interval −0.69 to −0.45, not overlapping anyone), then Leclerc (−0.19%). McLaren has the fastest car (−0.51%), then Mercedes (−0.24%). Race gaps come out about 1.46× larger than one-lap gaps, consistent with tyre wear amplifying differences. The blind spot is a dominant driver: the 2025 Red Bull is rated only 5th-fastest, because Verstappen's margin over his teammates is read as his skill rather than his car. For real driver–car pairings, skill + car is well measured. For swaps (a driver in a car they never drove), this split matters more than the intervals show.

5. **A simulator needs fresh information to forecast, and then it competes.**

   | Simulator test, 2025 (16 dry races from round 7, finishers, real grid) | Spearman | Avg error | Exactly right |
   |---|---|---|---|
   | Finish in grid order | 0.763 | 2.36 | 25.2% |
   | Simulator (pace forecast 1–6 races ahead) | 0.739 | 2.70 | 17.0% |
   | Simulator + weekend qualifying update | 0.792 | 2.31 | 24.1% |
   | Linear model | 0.805 | 2.31 | 20.6% |
   | Gradient boosting | 0.798 | 2.31 | 19.9% |

   Pace forecast weeks ahead loses to the grid, because the grid already reflects that weekend's pace. Updating the forecast with the weekend's qualifying times makes the simulator beat grid order and match the baseline models on average error. Its win probabilities are informative: the favourite won 81% of these races (50% without the update), and the actual winner's average simulated win chance was 48% (28% without). In a development run on 2024, a season that played no part in building the update, the pattern held: Spearman 0.859 vs 0.858 for the linear model and 0.800 for grid order. Rerun notebook 09 with `TEST_YEAR = 2024` to reproduce it.

6. **Small samples, modest claims.** With 15–16 test races per season, the differences between the updated simulator and the linear model are within noise: they're comparable at ranking finishers. The simulator's added value is calibrated probabilities (wins, podiums, points) and what-if scenarios, which a ranking model can't give.

7. **Checking the simulator against reality caught a real flaw.** Symmetric heavy-tailed qualifying noise suits fitting (ruined laps can't drag estimates) but not simulation: real outliers are mostly slow (348 laps beyond 3σ slower vs 123 faster). It gave midfield teams about 21 poles in 152 dry races, against none in reality. Making the simulated tail slow-only brought that to about 1.5.

The baseline numbers above were produced before circuit history was matched by circuit ID instead of event name. Rerunning on the same data, that fix moved the linear model by at most 0.002 Spearman and gradient boosting by about 0.01, so the conclusions are unchanged.

### Limitations

- **Dry races only.** Wet races are excluded from the race-pace data and the simulator; there's no estimate of wet-weather skill yet.
- **Skill vs car for dominant drivers.** Teammate comparisons can't fully separate a much stronger driver from his car (see point 4).
- **Cars start fresh each season.** The model needs a few races of a new season before it knows the new cars, so the first races of a season can't be forecast well.
- **Race mechanics are simple.** One overtaking threshold per circuit, with no strategy, tyre or safety-car modelling. At Silverstone and the Nürburgring, real races shuffled more than race pace alone explains.
- **Team form in the baseline resets when a team rebrands** (e.g. Racing Point → Aston Martin).

### Next steps

- Teammate-pairing search: use the simulator to find which two drivers score the most for a given team.
- A wet-weather skill term from the wet laps already downloaded.
- Carry car pace over between seasons (a team's new car starting near its old one) to forecast early-season races.
- A Streamlit app so others can run their own what-if races.

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

## Feature 3: baseline race model

Two parts, matching how the race simulator will work:

1. **DNF model:** chance a driver doesn't take the chequered flag (retired, disqualified, didn't start; lapped finishers count as finishers). Benchmark: the training-set DNF rate.
2. **Position model:** finishing order among finishers. Benchmark: finish in grid order.

Each is fitted as a linear model and a gradient-boosted tree model, trained on 2018–2024 and tested on 2025. Inputs use only information available before each race: grid, this weekend's qualifying gap to teammate, recent finishing form for driver and team, recent race-pace gap to teammate, recent DNF rates, and the circuit's history (how much the order usually changes, and its DNF rate; matched by circuit, so renamed events share one history). All models are scored on the same drivers.

**Limitations:** team form carries over between seasons even when the car changes a lot, and restarts when a team rebrands.

## Feature 4: driver skill + car pace (hierarchical Bayesian model)

Every observation of a driver's pace (% of a lap) is modelled as `session offset + driver skill + car pace + noise`, fitted with PyMC:

- **Observations:** each driver's best lap in each qualifying segment they took part in (Q1/Q2/Q3, as % behind that segment's fastest, with an offset per segment for track evolution), and each driver's race pace relative to the field from the per-race lap model, with its standard error added to the noise. A scale factor lets race gaps differ in size from one-lap gaps.
- **Driver skill:** one value per driver per season, drifting season to season (random walk), centred on each season's average driver.
- **Car pace:** one value per team per race, evolving race to race within a season (upgrades), centred on each race's average car.
- **Noise:** Student-t, separate for qualifying and race, so crashes or ruined laps don't drag estimates.

Teammates sharing a car separate skill from car pace, and drivers who change teams link the grid together. Every estimate comes with a 90% interval. The fitted posterior is saved to Drive for the race simulator.

**Limitation:** skill is measured against teammates. An unusually weak teammate, or a car built around one driver, inflates that driver's skill and deflates the car (e.g. Red Bull 2023).

## Feature 5: Monte Carlo race simulator

Simulates a dry race thousands of times for any field: drivers from any season they raced (2018–2025), in any team-season's car, at any circuit. Each simulation:

1. Draws one sample from the driver/car model, so uncertainty about skill and cars carries into the results.
2. **Qualifying:** skill + car pace at this circuit, plus qualifying noise, sets the grid. Noise is heavy-tailed on the slow side only (crashes, traffic): with the model's symmetric tail, midfield cars took about 21 simulated poles in 152 dry races versus 0 in reality; now about 1.5.
3. **Retirements:** each car retires at its team-season's dry-race DNF rate, adjusted for the circuit's DNF rate (both shrunk toward the average when data is thin).
4. **Race:** from the grid, a car passes the car ahead only when it's faster by more than the circuit's **overtaking threshold**. Each circuit's threshold is calibrated so simulated position changes match its real dry races (shrunk toward the all-circuit average for circuits with few races).
5. **Points:** 25-18-15-12-10-8-6-4-2-1.

Outputs: win, podium, points and DNF chances, a finishing-position probability table, and expected points per driver and team, with 90% intervals.

**Limitations:** dry races only (no wet-weather skill estimate yet). For swapped entries (a driver in a car they never drove), results depend on the model's skill/car split, which teammate comparisons only partly pin down, so they're flagged. Overtaking is a single threshold per circuit, with no strategy, tyre or safety-car effects.

## Out-of-sample test (2025)

The simulator is tested on races its model never saw. At 2025 rounds 7, 13 and 19 the driver/car model is refitted on all data before that round, and drivers and cars are carried forward to the next block of races by the model's random walks. Overtaking thresholds are recalibrated on earlier races only. Each dry race in the block is then simulated from its real grid with the actual finishers, and drivers are ranked by mean simulated finish. That is the same task the baseline model was scored on (grid known, order among finishers predicted), so the simulator, grid order and the baseline models are compared on exactly the same drivers.

**Weekend update.** The first version of this test showed the simulator ranking finishers worse than grid order: its pace was a forecast made 1–6 races earlier, while the grid reflects that weekend's real pace (in-sample, with pace known, it beat the grid clearly). So before each race the forecast is updated with that weekend's qualifying times, which are known before the race: a Gaussian (Kalman) update over pace and segment offsets, with ruined laps down-weighted using the model's Student-t noise, keeping teammates' shared-car correlation (`simulator.weekend_update`). Because 2025 had already been looked at when this was added, 2024 is used as a second, untouched test season. Nothing was tuned on test results. Rounds 1–6 aren't tested: the model starts each season's cars fresh and needs some data from the new season first.

## Project layout

```
f1sim/
  paths.py      # where data lives (Google Drive in Colab)
  ingest.py     # qualifying download: resumable, one file per session
  ingest_tracing.py # race laps, weather, race control from TracingInsights (free)
  race_pace.py  # clean laps, wet races, per-race lap-time model, teammate gaps
  ingest_results.py # race results (finishing order, grid, status, points) from Jolpica
  baseline.py   # pre-race features, DNF + position models, evaluation
  skill_model.py # hierarchical Bayesian driver skill + car pace model
  simulator.py  # Monte Carlo race simulator: qualifying, DNFs, overtaking, points
  backtest.py   # out-of-sample test: checkpoint refits, forecasts, scoring
  manual.py     # loads hand-entered times for sessions no API covers
  features.py   # teammate gap calculations
manual_data/
  quali_times.csv  # hand-entered qualifying times (see Data sources)
notebooks/
  00_ingest_quali.ipynb        # download data
  01_teammate_quali_gaps.ipynb # analysis and charts
  03_ingest_race_tracinginsights.ipynb # download race laps, weather, race control
  04_teammate_race_pace.ipynb  # race-pace model, charts, quali vs race
  05_ingest_race_results.ipynb # download race results, cross-check with lap data
  06_baseline_model.ipynb      # baseline DNF + position models vs benchmarks
  07_driver_car_model.ipynb    # driver skill + car pace model, charts
  08_race_simulator.ipynb      # simulate a race with any field
  09_backtest_simulator.ipynb  # out-of-sample test vs grid order and the baseline
```

Data is stored outside the repo (on Google Drive) and is never committed.

## Running in Google Colab

1. Open `notebooks/00_ingest_quali.ipynb` in Colab: **File → Open notebook → GitHub**, then paste the repo URL.
2. In the setup cell, set `REPO_URL` to this repo and run it. Approve the Google Drive mount.
3. Run the download cell. FastF1 has hard API limits (200 calls/hour to Jolpica, 500/hour overall), so all of 2018–2025 takes several runs. When it stops at the limit, wait about an hour and rerun; finished sessions are skipped.
4. Open `notebooks/01_teammate_quali_gaps.ipynb`, run the setup cell, then the analysis cells. Use the form fields to change season, team and driver.
5. For race data, run `notebooks/03_ingest_race_tracinginsights.ipynb` the same way. One run covers all seasons; there are no rate limits.
6. Open `notebooks/04_teammate_race_pace.ipynb` for the race-pace analysis.
7. Run `notebooks/05_ingest_race_results.ipynb` to download race results (one run).
8. Open `notebooks/06_baseline_model.ipynb` for the baseline model.
9. Open `notebooks/07_driver_car_model.ipynb` for the driver skill + car pace model (about 20–25 minutes to fit on Colab's free CPU; afterwards, untick `FIT_NEW` to reload the saved fit).
10. Open `notebooks/08_race_simulator.ipynb` to simulate races. It needs race results with circuit information (notebook 05; tick `FORCE` once if your results were downloaded before circuits were added).
11. Open `notebooks/09_backtest_simulator.ipynb` for the out-of-sample test (three fits of about 20–25 minutes each; finished fits are saved and reused if Colab disconnects).

If the repo is private, cloning in Colab needs a GitHub token. Keeping it public is simpler, and it's a portfolio project anyway.

## Running locally (optional)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m f1sim.ingest --start 2018 --end 2025 --data-dir ./data_store
```

## Data sources

- [FastF1](https://docs.fastf1.dev): F1 live timing data and the Jolpica-F1 API (successor to Ergast). Qualifying times come from Jolpica.
- [Jolpica-F1](https://github.com/jolpica/jolpica-f1) API, directly: race results (finishing order, grid, status, points, circuit).
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
