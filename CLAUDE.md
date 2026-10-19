# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

IntelliRate is a thesis project (University of the Peloponnese, ECE) that dynamically prices hotel rooms. It reconstructs the hotel's Best Available Rate (BAR) from real bookings, fits ML demand models to them, and then trains RL agents (PPO and Double DQN) in a simulator built from those models. It compares each ML model × RL algorithm combination with the hotel's actual prices and with the best fixed price.

## Environment and commands

This is Windows with Python 3.12. Create the conda env from [conda_env/intellirate-cpu.yml](conda_env/intellirate-cpu.yml) (the `prefix:` line is machine-specific), then activate it:

```
conda env create -f conda_env/intellirate-cpu.yml
conda activate intellirate-cpu
```

Run the pipeline in order, from the repo root. All paths are relative to the current working directory.

```
python bar_rates.py            # BOOKINGS.xlsx + rate_rules.csv + rate_occupancy.csv -> bar_daily.csv, bar_bookings.csv, bar_validation.txt, bar_daily.png
python ml_demand_model.py      # BOOKINGS.xlsx + bar_*.csv + hotel_pms_data.csv -> demand_models.pkl, sim_data.pkl, ml_comparison.csv, ml_cv_grid.csv, demand_fit.png
python rl_agent.py             # sim_data.pkl -> results_<season>_<model>_e<elasticity>/, comparison_<tag>.csv/.png, elasticity_log_<model>.csv
```

Useful `rl_agent.py` flags for quick or partial runs:
- `--models rf` (or `xgb`) runs only one pipeline.
- `--algos ppo` (or `ddqn`) trains only one RL algorithm (default: all keys in `ALGOS`).
- `--timesteps 20000 --eval-every 5000 --seeds 42 --no-sensitivity` gives a fast smoke run (the default is 500k steps × 3 seeds for each algorithm, validated every 25k steps, plus a sensitivity sweep). Note that a smoke run overwrites the `results_*` / `comparison_*` outputs of a full run with the same season, elasticity and world.
- `--season 2025` and `--elasticity 1.2` change the scenario.
- `--common-world rf` makes every pipeline use the same "true" demand model while each agent keeps its own forecast. The output directories and tags get a `_world-<m>` suffix.
- `--jobs 3` (the default) trains the seeds of each ML × RL combination in parallel processes, each with one torch thread. Parallelism is per seed, so at most `len(--seeds)` processes run. `--jobs 1` runs sequentially and gives identical results. A GPU doesn't help here: the networks are tiny and the bottleneck is Python overhead per step.

The repo has no tests, linter or build step. `.gitignore` uses patterns: the private data in any format (`BOOKINGS.*`) and every generated artifact (`*.pkl`, `*.zip`, `bar_*` outputs, `ml_comparison.csv`, `ml_cv_grid.csv`, `demand_fit.png`, `results_*/`, `comparison_*`, `elasticity_log_*.csv`, `logs/`) are ignored, so runs for any season, elasticity or world stay out of the repo. The two rate config files (`rate_rules.csv`, `rate_occupancy.csv`) are tracked. The pipeline runs without warnings (the Excel export's openpyxl style warning is silenced in `load_raw`); keep it that way rather than running with `-W ignore`.

**Testing rule: tests, checks and experiments must not make any permanent change to the repo's files unless the user explicitly says so.** That covers code, configs and docs, and also the generated outputs (`bar_*`, `*.pkl`, `ml_comparison.csv`, plots, `results_*`, `comparison_*`). Run them on a copy of the project in the scratch directory (copy the `.py` files, the two rate CSVs, `BOOKINGS.xlsx` and any inputs needed), patch only that copy, and report before applying anything. The rule also means no smoke runs of `rl_agent.py` in the repo, because they overwrite full-run results. Applying a change only after the user approves it, and then regenerating the outputs it requires, is fine.

Two project rules: only the **summer season** is studied (arrivals 1 April – 31 October, `SUMMER` in `ml_demand_model.py`), and the **hotel stays anonymous** — never write its name (it appears in the data's property, website and member-programme columns) into code, configs, outputs or docs.

## Architecture

There are four files, forming one pipeline: **BAR reconstruction → ML demand model → simulator → RL agent**.

0. **[bar_rates.py](bar_rates.py)** reconstructs the hotel's double-occupancy Best Available Rate (BAR) from the recorded prices. The rate system builds a price as `occupancy BAR × stacked discount factors`; the script unwinds that per booking:
   - [rate_rules.csv](rate_rules.csv): one row per discount/commission (NR, Genius, Last Minute Deal, Expedia promos and 18% commission, website direct offer / member programme), with detection column + regex, season, booking-date window, factor, optional `round_to` (rates rounded after discounts) and `confirmed`/`inferred` status. A `@infer` row (`genius_unflagged`) is decided per booking against the price ladder.
   - [rate_occupancy.csv](rate_occupancy.csv): the occupancy grid. 2025: no single rate, triple at a fixed price (no BAR information). 2026: per-level single and triple prices. Children aged ≥ 4 count as adults; only Booking.com records ages, so unknown ages are inferred from the price.
   - Double price levels count only when seen on ≥ 2 channel groups. Multi-night stays are tested against, and split by, the one-night-stay BAR (never against themselves). The daily BAR is the weighted median per stay night (cancelled bookings included).
   - Outputs `bar_daily.csv` (BAR per night, `source` observed / interpolated / filled), `bar_bookings.csv` (per booking: rules, occupancy, inference method, `bar`, `bar_ref`), and `bar_validation.txt` / `bar_daily.png`.

1. **[ml_demand_model.py](ml_demand_model.py)** loads `BOOKINGS.xlsx`, drops PII columns, and keeps room type `STD`, seasons 2025/2026, summer arrivals and arrivals on or before the export date. It then builds a daily table of booking *requests* (cancelled bookings included), the hotel price (the daily BAR from `bar_daily.csv`, not smoothed), and calendar/holiday features. The holidays are Greek: Orthodox Easter, Whit Monday and national days. `add_availability` adds `online_avail` and `censored` (sold out or closed online: `sold ≥ online_avail`; 41 nights in 2025, 34 in 2026). Censored nights stay in training but are never scored, because their demand above the rooms on sale was never recorded.

   RF and XGBoost (Poisson objective) each play **two roles with separate settings** (protocol "E revised", see the ML validation finding):
   - **Forecaster** (what the agent sees): settings from `GRID`, chosen only inside `TUNE_SEASON` = 2025 by `GroupKFold` blocked by ISO week, scored on uncensored days, best mean CV deviance (currently rf `min_samples_leaf=20, max_features=1.0`; xgb `max_depth=1, n_estimators=600`). It is then refit on 2025 and tested **once** on all of `TEST_SEASON` = 2026 (163 uncensored days) against the 2025 month mean, with 95% CIs from resampling whole ISO weeks. Out of time, neither model beats that baseline (gain CIs include 0); the level is about 0.54 of actual, but the env shows the agent only the pattern.
   - **World** (`lambda_true`): `WORLD_PARAMS` (rf `min_samples_leaf=3, max_features=0.6`; xgb `max_depth=3, n_estimators=400`), fitted on both seasons. It describes what happened, so it keeps the real day-of-week, holiday and peak structure; forecast-tuned settings would smooth it away.
   - `ml_comparison.csv` has one row per `evaluation`: `test_2026`, `baseline` and `pooled_cv_reference`. The last is the old pooled GroupKFold with neighbouring 2026 weeks in training: optimistic (gain +1.0), so report it only as an upper bound. `ml_cv_grid.csv` has the CV score of every config.
   - **Guardrails (the run fails):**
     - **G1:** the tune and test seasons split the table, settings are chosen on `TUNE_SEASON` rows only, and every forecast comes from a model that never saw that date.
     - **G2:** at least `MIN_TEST_DAYS` uncensored test days.
     - **G3:** the world's daily variance matches the real one within `WORLD_VAR_TOL` (1%). It is checked before `calibrate_to_availability`; the calibrated world the agents train in is 1.21–1.31× as volatile as reality (see the code review finding).
     - **G4:** the calibration converges within `CALIB_TOL`.
   - **Flagged checks (`!!!`, the run continues):**
     - a chosen config at an edge of `GRID` that is not a natural limit;
     - a forecaster significantly worse than the baseline;
     - a world with daily Spearman < 0.5, or any month outside 0.8–1.25 × actual before calibration.

   For each model it stores:
   - `lambda_true`: the world fit, rescaled so that each season's total matches the real total. It is then recalibrated by `calibrate_to_availability`: one factor per season and model (≈ 1.14–1.19), chosen so that the simulator, at the hotel's prices and under the real online availability, sells the real number of room-nights (within 0.5%). Demand on nights closed online was never recorded. The factor does not depend on elasticity. It applies to every night, so fully recorded (uncensored) nights also get 11–14% more demand than really arrived. It drives the simulated "world".
   - `online_avail`: rooms on sale online each night, `capacity − (PMS rooms − confirmed BOOKINGS rooms)` clipped to [0, capacity] (`load_pms`, `online_availability`). 2026: on average 10.4 of 17 rooms, with 11 nights closed. 2025: 8.8 of 18, with 15 nights closed.
   - `lambda_forecast`: the forecaster's predictions, never in-sample. 2025 is out-of-fold, and 2026 comes from the model trained on 2025. The agent sees these as its forecast.
   - `dispersion_k`: the negative-binomial overdispersion of the simulated world's gamma shocks, **one per season and model**, stored in `seasons[year]["dispersion_k"][model]`. `None` means pure Poisson. Per season it is chosen so that the world's total daily variance equals the real one, `Var(y) = E[fit] + Var(fit) + E[fit²]/k`, because the in-sample fit already holds part of the data's noise. It is computed on the fit **after** the per-season rescale, so the match is exact (guardrail G3). (Taking `k` from the OOF residuals, as originally, counted that noise twice and made the world 24–59% more volatile than reality. A single shared `k` left 2026 −7 to −10% and 2025 +17 to +25%. Computing it before the rescale left up to 3% off.) Current values: 2026 ≈ 4.6 (rf) / 4.2 (xgb); 2025 xgb ≈ 16 and rf `None`, because the rf in-sample fit already carries all of 2025's variance. The `dispersion_k` column in `ml_comparison.csv` is the forecaster's 2025 OOF value: it sets the 80% interval coverage and describes forecast error, not the world.
   - Per-month pools of real booking profiles (rooms, nights, Total, orig_price, cancelled). `orig_price` is the booking's own BAR (`bar_ref`), so scaling `Total` to another BAR keeps the guest's occupancy, channel and plan factors. A month with fewer than 20 profiles falls back to the season pool.

   Everything goes into `sim_data.pkl`, keyed by `seasons[year]`.

2. **[hotel_season_env.py](hotel_season_env.py)**: `HotelEnv` is a Gymnasium env where one step is one arrival date and one episode is one season.
   - The action is one of 11 BAR multipliers (0.75–1.25×) of the season reference price `p_ref`, the median daily BAR.
   - Price response: `g(p) = exp(-elasticity·(p/p_ref − 1))`. `lambda_base = lambda_true / g(hotel_price)` removes the hotel's own price effect, so demand can be re-priced.
   - All randomness for the season (gamma shocks, uniforms for the counts and the profile picks) is drawn in `reset()`, before any price is set. Counts use inverse-CDF Poisson, so a given seed is monotone in price. This gives paired, common-random-number comparisons between policies. Keep this property when you change the env.
   - Each request samples a real booking profile. Cancelled profiles give no revenue, and profiles that don't fit under `capacity` over their nights are lost. Capacity is the season's physical STD rooms, stored per season in `sim_data.pkl` (`CAPACITY` in `ml_demand_model.py`): 18 in 2025 (including room 108) and 17 in 2026 (room 108 was renovated and is no longer STD). `reset()` pre-fills `committed` with the real offline stays (`capacity − online_avail`), so online sales are limited to the real online availability. The offline calendar is the same in every episode; only online demand is random. Revenue is the profile's real `Total` scaled by `price / orig_price`.
   - The observation has 9 dims in [-1, 1]: forecast, committed rooms today and tomorrow (offline stays included), cyclic month and day of week, holiday flag, and season progress.
   - `HotelEnv(..., sim=dict)` takes the sim data directly; the calibration uses this.
   - Baselines use `step_price(price)` to pass an arbitrary price, for example the hotel's BAR.
   - **The env is the project's MDP** (the code never names it). The state is the day `t` plus the `committed` calendar, the action is the multiplier, the reward is revenue minus variable cost, the horizon is finite (one season) and γ = 0.9 is set in `rl_agent.py` (see the RL hyperparameter finding). Strictly it is a POMDP: the observation shows committed rooms only for tonight and tomorrow, so longer stays are partly hidden. With a mean stay of 1.26 nights, today's price barely affects later states, so the problem is close to a contextual bandit.

3. **[rl_agent.py](rl_agent.py)** runs one pipeline per ML model, and each pipeline trains every RL algorithm:
   - RL algorithms live in the `ALGOS` registry (`key -> (SB3 class, params, display name)`). `train_agent` is generic: SB3, CPU, `VecNormalize` on reward only, which is also valid for Double DQN because off-policy SB3 normalises replay-buffer rewards at sample time. To add an algorithm, add a registry entry.
   - Double DQN (`"ddqn"`, which replaced DQN) is `DoubleDQN(DQN)`. SB3 has no Double DQN, so it overrides `train()`, copied from SB3 2.8.0's `DQN.train` (re-check it after an SB3 upgrade). Only the next-state value changes (`next_q_values`): the online net selects the action and the target net evaluates it. The Experiment findings section has the comparison with DQN, made with identical `DQN_PARAMS`; those have since been tuned.
   - It validates the simulator: the hotel's prices in the sim are compared with the real season. Occupancy and RevPAR in `results.csv` are over the online available room-nights.
   - **Train / validation / test (protocol `val-test`).** All three are simulated seasons of the same world, told apart by seed:
     - **Training:** each training seed's own SB3 stream (about 1,500 seasons at 300k steps).
     - **Validation:** seeds from `VAL_SEED0 = 100000`, the old evaluation seasons, which past design choices already used. It chooses things and is never reported as a result:
       - `ValidationCheckpoint` runs the greedy policy on the first `N_VAL = 50` seasons every `--eval-every` steps (default `EVAL_EVERY = 25_000`) and at the end of training. It saves the best checkpoint as `<algo>_seed<n>.zip`, and that checkpoint is the one tested, not the final one.
       - The best fixed price is chosen on `N_VAL_FIXED = 200` validation seasons (no training, so more seasons are cheap; 50 sometimes missed a near-tie).
     - **Test:** `N_TEST = 200` seasons from `TEST_SEED0 = 500000`, shared by every policy (common random numbers). Every reported number comes from them, including the simulator check. Never choose anything on them: design choices use validation only, and the test is looked at once.
     - The callback has its own env, and a deterministic `predict` draws no random numbers, so training is identical to a run without it (checked: identical `monitor.csv` for all 12 agents).
   - It finds the best fixed price once per ML model, then trains each algorithm for each seed.
   - It reports the gain over the hotel split into a price-*level* effect (best fixed vs hotel, driven by the elasticity assumption) and a *day-by-day* effect per algorithm (RL vs best fixed). A negative day-by-day number means the agent hasn't converged.
   - **Error bars:** each seed row has the paired SE over the test seasons. The mean-of-seeds rows and the `comparison` columns `rl_vs_hotel_se_%` / `rl_vs_best_fixed_se_%` combine the variance between training seeds with the season noise (`seed_se`: `sqrt(Var_seeds/n + season_se²)`). The `*_ci95_%` columns use t(n−1), so 3 seeds give wide CIs (t = 4.30). Report the mean of all seeds, never the best one. More seeds need no code change (`--seeds 42 7 123 1 2024 --jobs 5` on a machine with enough cores).
   - It also sweeps elasticity for the best fixed price, with no training: chosen on validation, reported against the hotel on the first `SENSITIVITY_N_EVAL = 100` test seasons.
   - Per ML model it writes `results_*/` containing:
     - `results.csv` (test numbers; seed rows add `checkpoint_step`, `val_reward`, `val_vs_hotel_%`);
     - `validation_curve.csv`;
     - `summary.txt`;
     - `learning_curve_<algo>.png` (top: training episodes, exploration included; bottom: greedy validation curve, with the selected checkpoint starred);
     - `price_path.png`;
     - `<algo>_seed<n>.zip` (the selected checkpoint).

     `comparison_<tag>.csv/.png` gets one row per ML × RL combination, with `rl_*` columns.
   - Every run appends one row per ML model to `elasticity_log_<model>.csv` (`append_elasticity_log`). Each row has the timestamp, the `protocol` (`val-test`; logs from before the change lack the column and must be renamed before the next run), season, elasticity, timesteps and seeds, the hotel revenue, the best fixed multiplier and its revenue, plus each algorithm's mean-of-seeds revenue and occupancy, all taken from `results.csv`. It tracks results across runs at different `--elasticity` values. A rerun adds a row instead of replacing one, and a skipped algorithm leaves its columns empty. If the log's columns no longer match `ALGOS`, the run fails instead of appending. Runs in the repo append to it too, which is another reason to keep smoke runs in a scratch copy.

## Booking data and rates

`BOOKINGS.xlsx` is a private channel-manager export. Every row is in EUR, and `Total` is the room charge only (`Total == Stay`). `load_raw` keeps the recorded rate `price = Total / Room-Nights`; all rate normalisation lives in the BAR stage (`bar_rates.py` + the two rate CSVs). Change rates there, not in the ML/RL files. Rate structure (details and status per rule in the CSVs):
- Channel: `Source`, except website bookings (`Application == "WEBHOTELIER"`, `Source` empty or Google Free Booking Links), which form the "Direct" channel. Expedia Collect `Total` is net of 18% commission; all other channels are gross (Booking.com commission is not removed).
- Rate parity across channels. NR is −10% everywhere. Booking.com Genius (−10%, flagged "Genius Booker" in `Channel Notes for Hotelier`), the Booking.com Last Minute Deal (−5%, in `Room Type Name`), the Expedia promos in `Rate` (Blue Member −10% in 2025, Member / Mobile −7% in 2026) and every other promo stack by multiplication. 2025 website: permanent −12% direct offer. 2026 website: member programme (`SmartGuest`) −10%, or −14.5% for bookings made from 1 Sep 2026 (those rates are rounded to the euro).
- Occupancy pricing: see `rate_occupancy.csv`. BAR always means the double-occupancy BAR.
- Flexible bookings cancel at a rate of about 36%, NR bookings at about 2%. `Cancel Fee` is always 0. In the simulator a cancelled profile earns nothing and never holds capacity.
- The 2026 season is cut off at the export date (16 Oct, the latest `Booking Date` in the refreshed export; it was 27 Sep), while 2025 runs to 31 Oct. The PMS file stops at 14 Oct arrivals, so the last 2026 nights lack offline stays (see the code review finding). The daily table starts at each season's first actual arrival (3 Apr in both seasons), not on 1 Apr.
- The hotel's online booking systems opened late in 2025, on about 1 Apr. About a third of the 2025 demand is therefore unrecorded, including about three quarters of April. **Decision: 2025 is kept as recorded, with no correction.** Treat 2026 as the main RL season, and describe 2025 as a limitation rather than fixing it.

Other data files. Only the PMS file is used by the pipeline, and every private one is gitignored by pattern (`BOOKINGS.*`, `hotel_pms_data.*`, `avl-status*`, `kaggle_data.*`). `avl-status*` and `kaggle_data.*` are currently kept outside the repo folder.
- `hotel_pms_data.csv`: the hotel PMS export of every STD stay, online and offline (arrivals 1 Apr 2025 – 14 Oct 2026). Format: Greek cp1253, `;`-separated, quoted. Title and filter lines come first, the header is on line 4, blank lines separate the records and two footer lines close the file. 28 rows carry one stray empty field after the arrival date. Columns: number, arrival, departure, room, price list (`Group` / `open` / `Compl` / `Over`), price, status, adults, children and source. There are no guest or debtor columns, but print the title and filter lines only by shape and save only per-night aggregates. The source tag misses about 30% of online stays, so identify online stays by subtracting BOOKINGS, not by the tag. **Used only for per-night online availability:** `load_pms` reads arrival, departure and room, and nothing else. Groups mostly booked before online sales opened, and offline bookings are typed in later from a paper calendar, so PMS entry dates are not booking dates. Room 108 is STD in 2025 only, as above, but the 2026 rows still hold 3 one-night stays in it. Confirmed online STD stays are missing from the PMS on 19 nights of 2026 (32 room-nights) and 5 of 2025, probably upgrades or room moves (see the code review finding).
- `avl-status*.xlsx`: the booking engine's final availability per room type and night (1 Apr 2025 – 31 Oct 2026). `Classic Room` is the STD row. Cell fill: green = on sale, red = stop-sell, grey = 0. Values are exact only on 0 and red nights; elsewhere they are a lower bound. Online STD inventory is usually only about 4–8 rooms, and STD was sold out or stop-sold online on about 54 summer nights in 2026 (67 in 2025). Its file name and row 2 contain the hotel's name: match it by pattern, never print row 2, and keep it out of git.
- `kaggle_data.xlsx`: the public Hotel Booking Demand dataset (Portugal, 2015–17). It is a candidate second case study (the Resort Hotel) for reproducibility, not training data. Its `adr` column is corrupted by the Excel conversion, so use the original CSV.

## Experiment findings (Oct 2026, season 2026, elasticity 1.0)

- **Rooms rarely run short at 17 online rooms**, so dynamic pricing has almost nothing to gain: the best fixed price ≈ the hotel (+0.01%), and a capacity-aware rule adds ≤ 0.2%. The trained agents (500k steps) sit 0.4–1.1% below the best fixed price because they wander between price levels. That is a lack of signal, not a range problem.
- **Multipliers:** 0.75–1.25× in 5% steps is fine at ε = 1. Widening the range doesn't help the RL; a narrower range (≈ 0.90–1.20×) halved the RL loss in a 2-seed test. The sensitivity sweep's best-fixed search is capped by the grid at ε ≤ 0.8.
- **Extended observation** (7-night committed rooms, 7-day forecast, yesterday's price) and a **variable cost** (€20/room-night, profit reward) gave no consistent RL improvement in the 17-room simulator. Cost raises the best price level (1.15×), which is a level effect.
- **Capacity is the key lever.** Lowering a constant capacity raises the gain only as fast as simulator realism falls; 14 (the online maximum) is defensible but changes little. A per-night online capacity from the availability report gives about +2–5% RevPAR over the hotel, but that simulator under-books the real season by 11–19%. It needs demand correction on sold-out (censored) nights and recalibration to about ±2% before its gains can be reported. (Superseded: the PMS availability below replaced the constant capacity.)
- **PMS online availability vs BOOKINGS-only (full runs, 500k × 3 seeds, refreshed export, same settings):**
  - **Sim vs real:** PMS gives bookings +0.2–0.3%, room-nights −0.9% and revenue −1.0 to −1.1% (BOOKINGS-only: within 0.4%).
  - **Lost requests at the hotel's prices:** about 90–100 per season (BOOKINGS-only: about 2).
  - **Best fixed price:** 1.15×, +1.73% (rf) / +1.77% (xgb) vs hotel. BOOKINGS-only: 1.00–1.05×, +0.05–0.08%. This is a scarcity level effect.
  - **PPO vs hotel:** +1.51% (rf) / +1.49% (xgb), seed spread 0.1–0.3%. BOOKINGS-only: −0.57 / −0.85%.
  - **PPO vs best fixed:** −0.22 / −0.27% (BOOKINGS-only: −0.62 / −0.93%), so there is still no day-by-day gain.
  - **Sensitivity of best fixed vs hotel:**

    | ε | 0.6 | 0.8 | 1.0 | 1.2 | 1.5 |
    |---|---|---|---|---|---|
    | PMS | +11% | +6% | +2% | 0% | +1% |
    | BOOKINGS-only | +8% | +3% | 0% | +1% | +6% |

    With PMS, a lower price can't fill rooms that aren't there, which is why ε = 1.5 gives only +1%.
  - The gain relies on the recalibration's assumption of unrecorded demand on closed or near-full nights.
  - The training curves flatten by about 300–400k steps, so 1M steps is expected to add little.
- **Double DQN vs DQN (PMS, 500k × 3 seeds, same world and evaluation seasons, identical `DQN_PARAMS`):**
  - **rf:** DQN +0.87 / −0.36 / −1.76% per seed (mean −0.42%, spread 1.07%). Double DQN +1.52 / +1.20 / +0.58% (mean **+1.10%**, spread 0.39%). Every seed improves.
  - **xgb:** DQN +0.98% (spread 0.31%), Double DQN +0.94% (spread 0.50%), essentially unchanged.
  - Removing overestimation fixed DQN's instability where it existed. Double DQN is still below PPO (+1.5%) and below best fixed (−0.62 / −0.82%).
- **7-night observation under PMS scarcity (scratch test, not applied; 500k × 3 seeds, same world and evaluation seasons):**
  - **The test:** today's forecast and committed rooms for tonight and tomorrow were replaced with committed rooms (offline + sold) and the forecast for the next 7 nights (9 → 20 dims).
  - **Result: no gain.**
    - PPO: +1.49 / +1.48% vs hotel, against +1.51 / +1.49% with 9 dims.
    - Double DQN: rf +1.07% (was +1.10%), xgb +0.61% (was +0.94%, within its 0.5 seed spread).
    - Both stay below best fixed (PPO −0.23 / −0.28%).
    - The 20-dim agents learn slightly faster early but level off at the same revenue.
  - **Why:** every episode uses the same offline calendar, so the calendar features already encode upcoming group blocks. Stays are mostly one night (mean 1.26), so tonight's free rooms carry almost all the signal.
  - The observation stays at 9 dims. The agents' gap to best fixed is not caused by missing information about future scarcity.
- **Volatility fix (per-season `dispersion_k`):** the world no longer invents busy days (2026: about 0.9 full nights per season at the hotel's prices, down from 1.4–1.5). The best fixed price is +0.06–0.10% over the hotel and a capacity-aware rule adds nothing, so the constant-capacity conclusion holds in a world that matches reality in both totals and variability. The findings above were measured before this fix.
- **PMS pre-fill test (read-only, preliminary PMS file):** offline stays (groups ≈ 57–63% of STD room-nights, plus complimentary and offline direct) leave ≤ 3 rooms for online on 34 (2026) / 59 (2025) nights. Pre-filling them gives a simple capacity rule about +1% over the hotel, but the simulator then under-books the real online season by 15–17% (date-only matching left 110 online bookings unmatched, so some are double counted, and demand on closed nights is censored). Superseded by the per-night subtraction (no booking matching) plus recalibration above.
- **ML validation protocols (scratch tests; E revised, at the end of this item, is applied in `ml_demand_model.py`):**
  - **Variants, all scored on uncensored nights:**
    - **A:** current pooled GroupKFold.
    - **B1:** 5-fold GroupKFold within 2025 to choose the settings, refit on 2025, test on all 2026.
    - **B2:** train Apr–Aug 2025, validate Sep–Oct 2025, refit on 2025, test on 2026.
    - **C:** train 2025 + Apr 2026, validate May–Jun 2026, refit up to 30 Jun, test 1 Jul – 14 Oct 2026.
    - **D:** 5-fold GroupKFold on 2025 + 2026 up to 30 Jun, refit on that, test 1 Jul – 14 Oct 2026.
    - **E:** B1 rerun in the second round below; **E revised** in the third round.
  - **Censored nights:** sold out or closed online (`sold ≥ online_avail`): 41 in 2025, 34 in 2026.
  - **Out of time, neither model beats the season-month mean on 2026** (Poisson deviance gain CIs include 0; from 15 Aug the baseline is significantly better). A's apparent edge comes from interpolation.
  - **The error is the level, not the pattern:**
    - Calendar-only models forecast 2026 at 55–67% of actual, because Jul–Oct 2026 ran 65% above 2025 and trees can't extrapolate that growth, even with spring 2026 in training (C).
    - The pattern is as good as the baseline's or slightly better. The env normalises the forecast by its maximum, so the agent only sees the pattern.
  - **C's 48-day validation contradicted its test.**
  - **B1 and B2 chose the same settings,** so their test results are identical:
    - with a widened grid, B1/E chooses rf `min_samples_leaf=20, max_features=1.0` and xgb `max_depth=1, n_estimators=600`, about 7% better CV deviance than the current `RF_PARAMS` / `XGB_PARAMS`;
    - B2 extrapolates to unseen months (rf validation level 1.23) and has only 40 validation days;
    - B1 is recommended.
  - **NB intervals are too narrow on 2026:** 80% intervals cover 66–77% of days.
  - **Pitfalls if censored nights are dropped from training:**
    - **Rescale:** `build_sim_data` must rescale `lambda_true` to the real total over uncensored nights only. Otherwise the calibration factor rises (2026: 1.23–1.25 instead of falling to 1.14–1.15).
    - **Baseline:** a month × weekday mean baseline needs shrinkage toward the month mean, because empty cells give deviances of 6–11.
  - **D vs E (second round, wider grid, same 92 uncensored test nights 1 Jul – 14 Oct 2026):**
    - D's level is better (0.64 vs 0.59–0.61) and its 80% intervals are calibrated (0.79 vs 0.71). The raw deviance gain over E is +4% (rf) / +2% (xgb), with CIs that include 0.
    - E's pattern is better (deviance after rescaling to the actual total): rf +3% (n.s.), xgb +9% (significant). The cause is the settings: D picked a deeper xgb (`max_depth=2, n_estimators=200`), while rf settings are the same in both.
    - Neither beats the 2025 month mean. "2025 month mean × spring 2026/2025 trend" (trend 2.09, inflated by the unrecorded spring 2025) beats both clearly (deviance 2.45 vs about 3.1), although it overshoots the level by 24%.
  - **Monte Carlo CV (50 splits, 20% held out) changes nothing:** holding out weeks or days chooses the same settings as GroupKFold on D's data. Holding out weeks lowers the CV standard error by about 30%. Holding out days lowers it further only through leakage between neighbouring days. The real issue is a flat score surface: 11–14 configs fall within 1 SE of the best, so D and E's different xgb choices are near-ties.
  - **Third round: E with the one-SE rule vs E revised vs D (full pipeline runs in scratch):**
    - **E one-SE (rejected):** the one-SE rule (5 folds on 212 days, SE ≈ 0.13) falls to the simplest corner (rf `min_samples_leaf=50`; xgb depth 1, 100 trees). Used for the world too, it drops day of week and holidays (importance ≈ 0), misses the August peak, and its daily variance is 20–28% off reality (calibration 1.19–1.29).
    - **The world and the forecaster need different settings.** The forecaster must generalise to an unseen season; the world (`lambda_true`) describes what happened and must keep its day-of-week, holiday and peak structure.
    - **E revised vs D, same 92 out-of-time nights (1 Jul – 14 Oct 2026):** D is slightly better raw (n.s.), E revised has the better pattern (xgb significantly). Over all of 2026, E revised is fully out of time and on a par with the 2025 month mean. D's 2026 forecast is interpolated for Apr–Jun, so its full-season score is not a forecast.
    - **Agent view (max-normalised forecast vs actual, 2026):** E revised 0.39–0.40, current (leaky) 0.38, D 0.28–0.36, E one-SE 0.31.
    - **The current pooled forecast's 2026 gain (+1.0 over the 2025 month mean, significant) is leakage** from neighbouring 2026 weeks, not forecasting skill.
  - **Chosen protocol: E revised (applied; implementation and guardrails in the Architecture section).**
    - **Forecaster (`lambda_forecast`):** settings with the best GroupKFold CV score (by ISO week, uncensored days) inside 2025, no one-SE rule. 2025 is out-of-fold, 2026 is predicted by the model trained on 2025. It matches the env, which uses one static forecast per date and shows the agent only its shape.
    - **Test:** all of 2026 once, on uncensored nights, against the 2025 month mean, with 95% CIs from resampling ISO weeks. Report both this honest score and the pooled one (optimistic).
    - **World (`lambda_true`):** keep the current `RF_PARAMS` / `XGB_PARAMS` fit on 2025 + 2026. Compute `dispersion_k` on the fit **after** the per-season rescale, so the world's variance matches reality exactly (2026: 4.6 rf / 4.2 xgb; 2025 xgb 16; calibration factors move ≤ 0.6%).
    - **D:** a one-paragraph robustness check (better level and interval coverage, not pattern). It becomes the realistic choice only if the env re-forecasts during the season (booking lead time).
    - The sim schema is unchanged; the agent's forecast and `dispersion_k` change, so `rl_agent.py` needs a full rerun. A 20k-step smoke run in scratch was too short to compare the two forecasts (differences −1.2 to +1.1 points, changing sign). Don't pick the protocol by RL revenue, because that selects on the test.
    - A wider xgb grid (up to 2000 trees) confirmed 600 trees as an interior optimum; the grid now goes to 1000.
- **Full RL run on the E revised ML (17 Oct 2026; 500k × seeds 42 / 7 / 123, season 2026, ε = 1.0), compared with the previous full run (16 Oct, pooled-CV forecast):**

  | vs hotel | Before | E revised | Seed spread |
  |---|---|---|---|
  | Best fixed rf / xgb | 1.15×, +1.73% / 1.15×, +1.77% | 1.20×, +1.59% / 1.15×, +1.74% | – |
  | RF + PPO | +1.51% | +1.49% | 0.06 |
  | RF + Double DQN | +1.10% | +1.03% | 0.26 |
  | XGB + PPO | +1.49% | +1.37% | 0.42 |
  | XGB + Double DQN | +0.94% | +1.03% | 0.27 |

  - **The honest forecast costs nothing:** every change (−0.12 to +0.09 points) is within the seed spread and about one paired SE (≈ 0.1). The rf best fixed moved to 1.20× because of the small world change (`dispersion_k` after the rescale); 1.15× and 1.20× are a near-tie. Sim vs real revenue: −0.9 / −1.0%.
  - **Still no day-by-day gain:**
    - PPO vs best fixed: −0.11% (rf, was −0.22) / −0.37% (xgb, was −0.27);
    - Double DQN vs best fixed: −0.55% / −0.69% (was −0.62 / −0.82).
  - **The agents learn the level, not the pattern.** ADR is 146–155 € against the hotel's 136 €, at occupancy 32–34% against 35.8%, with fewer lost requests. The price paths wander (PPO mostly 184–200 €, with drops to 128–152 € on quiet days) and don't track the forecast.
  - **The learning curves plateau at about 150–200k steps** and then swing by about ±1,000 €, with seeds overtaking each other. The final checkpoint is effectively a random draw from that band.
  - **One run beat best fixed:** XGB + PPO seed 7 reached +1.86% against best fixed +1.74%. Picking it after the fact would be selection on the test seasons, so it can't be reported.
  - **Sensitivity is unchanged:** +11 / +6 / +2 / 0 / +1% at ε = 0.6 / 0.8 / 1.0 / 1.2 / 1.5.
  - **What the RL validation step must fix:**
    - seed and checkpoint selection on the reported seasons;
    - the best fixed price chosen on the same seasons it is reported on;
    - error bars that ignore the variance between training seeds.

    All three are fixed by the `val-test` protocol (next item).
- **RL train / validation / test protocol (smoke runs in scratch, 17 Oct 2026; 100k × seeds 42 / 7 / 123, both ML models and both algorithms; old code vs the `val-test` protocol, which is applied):**
  - **The old code didn't overfit its evaluation seasons.** I re-tested its agents on 200 fresh seasons: the gain was the same as on the reported seasons (mean difference 0.00 rf / 0.04 xgb points; each agent within ±0.15). Every training episode is a fresh draw, and nothing was chosen on the agents.
  - **Its best fixed price was about 0.1 points optimistic** (+1.62 / +1.73% reported, +1.51 / +1.63% on fresh seasons). That is mostly noise between the season sets; the best level on fresh seasons was the same (rf) or a near-tie (xgb 1.20× vs 1.15×).
  - **The final checkpoint was the real problem.** The checkpoint chosen on validation beat the final one on fresh seasons by +0.72 (rf) / +0.82 (xgb) points on average, and by up to +2.4 for one Double DQN seed. The reported SE (≈ 0.1) also hid seed differences of up to 3 points.
  - **The new protocol's checkpoint selection isn't optimistic:** validation − test is −0.09 (rf) / −0.04 (xgb) on average, and each agent is within ±0.3.
  - **Test results** (vs hotel, 95% CI including seed variance):

    | | rf | xgb |
    |---|---|---|
    | PPO | +1.16 ± 0.47% | +1.56 ± 0.51% |
    | Double DQN | +0.42 ± 0.85% | +0.45 ± 0.71% |
    | Best fixed | +1.51% | +1.63% |

    PPO vs best fixed: −0.35 ± 0.33 (rf) / −0.07 ± 0.31 (xgb). There is still no day-by-day gain.
  - These are 100k-step numbers; the full 300k run will replace them.
  - **The checks don't cover overfitting to the simulator itself** (same world, offline calendar and ε in training and test). That is what the robustness tests in future work are for.
- **PMS data as demand training data (scratch test, not applied): no.**
  - Group stays move against online requests (weekly correlation −0.4 to −0.6, also on uncensored nights), and total PMS occupancy carries almost no signal (≈ +0.1).
  - As a prior-year pattern for 2026, every PMS index is significantly worse than the 2025 online month mean. Older PMS-only seasons (2023–24, no online channel) would therefore not help the demand model.
  - The offline calendar repeats strongly between years (weekly correlation 0.83, groups 0.87). Older PMS seasons are useful for modelling offline stays (varying availability between episodes), not demand.
- **Code review (18 Oct 2026; read-only checks on a scratch copy, nothing applied):**
  - **The calibration makes the world busier and more volatile than reality, also on fully recorded nights.** `calibrate_to_availability` scales every night of a season by one factor, and G3 is checked before it:

    | World ÷ real | 2025 rf / xgb | 2026 rf / xgb |
    |---|---|---|
    | Requests, uncensored nights | 1.12 / 1.14 | 1.12 / 1.12 |
    | Requests, censored nights | 1.27 / 1.35 | 1.69 / 1.78 |
    | Daily variance after calibration | 1.21 / 1.25 | 1.29 / 1.31 |

    The extra demand on uncensored nights is scarcity the real season didn't have, and scarcity is what lifts the best fixed price to 1.15–1.20× (most of the gain over the hotel). Statements that the world matches reality in variability hold only before calibration. Fix: see future work (world calibration).
  - **The level gain lies outside the hotel's observed prices.** The hotel's daily BAR stays within 0.92–1.13× `p_ref` in 2025 (142–175 €) and 0.92–1.11× in 2026 (147–177 €), while the best fixed price is 1.15–1.20× (2026: 184–192 €). The +1.6–1.7% level effect therefore extrapolates the assumed `g(p)` beyond any price in the data.
  - **Data alignment** (each changes `online_avail`, so a fix needs an ML + RL rerun):
    - The export date is 16 Oct 2026, but the PMS holds arrivals only up to 14 Oct. The night of 16 Oct has 0 PMS rooms, so all 17 rooms go on sale online, and 15 Oct misses that day's offline arrivals.
    - On the nights where confirmed online stays are missing from the PMS (19 nights / 32 room-nights in 2026, 5 / 5 in 2025), `PMS − online` goes negative, is clipped to 0, and the whole hotel opens online. The same bias is invisible on other nights.
    - Room 108's 3 stays in 2026 count as offline STD stays against a capacity of 17 that excludes it, and the PMS exceeds capacity on 2 nights per season (clipped).
  - **Latent code issues (no effect on current results):**
    - `load_rules` uses `.replace("", None)`: NaN in pandas 3 (checked), but pandas ≤ 2.x forward-fills the row above, which would silently shift booking-date windows (`web_member_sep` would never match).
    - `occupancy_class` falls back to `""` when people per room isn't a whole number (e.g. 3 adults in 2 rooms), which gives a silent NaN BAR. No such booking exists today.
    - `load_bar` only checks booking IDs, so edited rate CSVs without a `bar_rates.py` rerun go unnoticed.
    - The "Best combination" line in `rl_agent.py` picks the maximum on the test seasons, against the protocol.
    - `deviance_gain` shares one `rng` across models, so its CIs aren't paired between models and depend on call order. `rl_seed_spread_%` uses `ddof=0`, unlike every other SE.
  - **Verified correct:**
    - `DoubleDQN.train` matches SB3 2.8.0's `DQN.train` line for line, apart from the target.
    - A deterministic `predict` draws no random numbers (DQN skips `np.random.rand()`).
    - `VecNormalize` reward normalisation is valid for both algorithms.
    - The Orthodox Easter dates are right (20 Apr 2025, 12 Apr 2026).
    - Mean rooms per booking is 1.002, so `lambda` counting rooms while the env draws bookings makes no difference.
  - **Efficiency:** greedy evaluation spends about 320 of its 380 µs per step in SB3's single-observation `predict`. Stepping the 50 validation seasons in lockstep, with one batched `predict` per day, gave bit-identical rewards (PPO and Double DQN seed-42 checkpoints) and was 3–4× faster (3.7 → 1.0 s per validation).
- **RL hyperparameters (validation grid, 19 Oct 2026; applied):**
  - **The simulator holds day-by-day value, and the agents missed most of it.** A per-date price schedule (best multiplier per date on 1,000 validation-range seasons, scored on 400 held-out seasons from seed 300000) beats best fixed by +0.75% (SE 0.07). PPO with the old settings at 1M steps: +0.26% on test. The signal is tiny: on the median day, the best price earns 4.6 € more than best fixed's, against a daily reward SD of 399 €. 48 of its 197 dates sit at the 1.25× cap.
  - **Grid:** 300k steps, rf world, seeds 42 / 7 / 123, scored as % vs best fixed on the 50 checkpoint-validation seasons (mean after 150k steps):
    - **PPO:** −0.87 → −0.10 with γ 0.9 / GAE λ 0.5 (seed SD 0.40 → 0.12). γ 0.5 ties; LR decay adds nothing. Why: the problem is close to a contextual bandit, and γλ = 0.94 adds about 17 days of unrelated demand noise to each advantage.
    - **Double DQN:** −1.44 → −0.62 (seed SD 0.30 → 0.04) with batch 512 + γ 0.9 + soft target updates (τ 0.005 every gradient step) + linear LR decay, **tested only together**. Alone: the batch raises the level, γ cuts the swings, soft updates do nothing, LR decay hurts.
  - **Tested at 300k only.** LR decay and `exploration_fraction` are fractions of the run, so longer runs stretch both.
  - **Optuna is not worth it yet:** a small search space, trial noise of 0.2–0.4 points against differences of 0.1–0.3, and a ceiling of about 0.75 points.
- **Calibration on vs off (scratch test, 19 Oct 2026; nothing applied):** the same world fits with and without `calibrate_to_availability`, on the repo's validation and test seeds. The calibrated version reproduces `sim_data.pkl` and `results_2026_*` exactly.
  - **Without calibration the simulator under-books:** at the hotel's prices it sells 11–13% fewer room-nights and earns 11–13% less than the real season (both seasons and models). With calibration: within 1%.
  - **Night by night, the uncalibrated world is closer to reality** (2026, rf / xgb; 2025 sold-out nights: 38–39 / 43–44 vs 41 real):

    | | Uncalibrated | Calibrated | Real |
    |---|---|---|---|
    | Requests on uncensored nights | 0.96 / 0.95× | 1.11 / 1.12× | 1 |
    | Daily variance | 1.00× | 1.29 / 1.31× | 1 |
    | Sold-out nights per season | 37 / 38 | 43 / 44 | 34 |

  - **Most of the shortfall is on uncensored nights.** 2026 online room-nights (rf / xgb): censored nights 46 / 45 uncalibrated, 50 / 48 calibrated, 65 real; uncensored nights 605 / 599, 691 / 688, 677 real. The calibration reaches the total by over-selling uncensored nights (+2% in 2026, +6% in 2025), because extra demand barely fills censored nights (+17% demand adds 3–4 room-nights there).
  - **Why:** recorded requests are already net of lost guests (a stay that hit a full night was never recorded, whatever its arrival date), and the simulator loses requests again (66–73 per season uncalibrated). The rescale to the season total also moves demand from uncensored to censored nights (0.95–0.96× on uncensored). Scaling only censored nights can't close the gap.
  - **Best fixed vs hotel, 2026** (SE ≈ 0.11):

    | ε | 0.6 | 0.8 | 1.0 | 1.2 | 1.5 |
    |---|---|---|---|---|---|
    | Uncalibrated rf / xgb | +10.2 / +10.4% | +5.2 / +5.4% | +0.9 / +1.2% | −0.1 / −0.1% | +1.9 / +2.0% |
    | Calibrated rf / xgb | +10.8 / +10.9% | +6.0 / +6.1% | +1.5 / +1.6% | −0.1 / −0.1% | +1.0 / +1.1% |

    Both choose 1.15× at ε = 1. The calibration adds about 0.5 points (roughly a third of the gain), so most of the scarcity comes from the PMS offline stays. At ε = 1.5 it lowers the gain: with less demand, a lower price fills more empty rooms. 2025, ε = 1: +0.35 / +0.19% uncalibrated, +0.74 / +0.76% calibrated.
  - The per-date calibration prototype (next item) widens the range of the ε = 1 level gain; report the range given there.
- **Per-date calibration prototype (scratch test, 19 Oct 2026; nothing applied):**
  - **Method:**
    - Rescale the world fit on uncensored dates only.
    - Raise each date's demand by an uplift between 1 and a cap, until the simulator, at the hotel's prices, records on that date as many requests as the fit says.
    - Recorded = accepted + cancelled requests whose stay would have fitted. The env records every cancelled request, fitting or not; counting those left a further ~1.5 points of under-booking. The change is to the count only; revenue is unchanged.
    - Demand on full or closed dates is unidentified, so the cap (1.5 / 2 / 3) is an assumption.
    - Calibration uses seeds from 200000; the checks use the test seeds.
  - **Convergence:**
    - 2026: within 1% per date in 6–17 rounds.
    - 2025: xgb 1.4% and rf 6.7% after 40 rounds.
    - Dates at uplift 1 carry only sampling noise, so they are left out of the check.
  - **Realism, 2026** (rf / xgb; recorded counts only cancelled requests that fitted):

    | | Uncalibrated | One factor (current) | Per-date, cap 2 | Real |
    |---|---|---|---|---|
    | Recorded requests, uncensored dates | 0.91 / 0.90× | 1.04 / 1.03× | 1.00 / 1.00× | 1 |
    | Variance of recorded requests | 0.89 / 0.88× | 1.08 / 1.08× | 1.00 / 1.01× | 1 |
    | Room-nights at the hotel's prices | −12.3 / −13.4% | −0.2 / −0.8% (target) | −2.9 / −3.2% | 0 |
    | Sold-out nights per season | 37 / 38 | 43 / 43 | 48 / 49 | 34 |
    | Demand on censored dates ÷ recorded | 1.45 / 1.51× | 1.69 / 1.78× | 2.84 / 3.01× | unknown |

  - **Better than the current calibration:**
    - Recorded requests and their variance match reality; it passes G3 after calibration.
    - Room-nights land within 3% (5% in 2025) without being the target.
  - **Not ready:**
    - 48–49 sold-out nights against 34 real, more than either other world.
    - Room-nights are still 3% short (5% in 2025), mostly on censored nights, which stay 13–16% under-filled even at cap 3 (about 30% in 2025).
    - Both misses point to too much day-to-day randomness. Part may be regression to the mean, because the censored split comes from the one real season.
  - **Best fixed vs hotel, 2026, ε = 1** (SE 0.12–0.15):

    | World | rf | xgb | Best level |
    |---|---|---|---|
    | Uncalibrated | +0.9% | +1.2% | 1.15× |
    | One factor (current) | +1.5% | +1.6% | 1.15× |
    | Per-date, cap 1.5 | +2.1% | +2.4% | 1.20× |
    | Per-date, cap 2 | +2.4% | +2.8% | 1.20× / 1.25× |
    | Per-date, cap 3 | +2.8% | +3.1% | 1.25× |

    - Per-date, cap 2, other ε: +11.7 / +11.8% (0.6), +6.9 / +7.3% (0.8), +0.1 / +0.1% (1.2), +0.5 / +0.5% (1.5).
    - 2025, ε = 1: +1.2 / +1.0%.
  - **The level gain is positive in every world tested, but its size depends on the hidden demand on full nights**, which the data can't identify: +0.9 to +3.1% at ε = 1. The cap alone moves it by about 0.65 points. Report it as this range, with the reason.
  - Per-date worlds push the best price to the 1.20–1.25× grid edge, further outside the hotel's observed prices, so the price-range check matters more.
- **One factor with `dispersion_k` re-fitted (read-only check, 19 Oct 2026; nothing applied; 2026, rf / xgb, recorded requests as in the item above):**
  - **Matching the recorded-request variance after calibration needs k ≈ 6.2 / 5.4** (now 4.7 / 4.3): recorded variance 1.00× on the calibration seeds (1.03× on test), room-nights +0.5 / −0.2%, sold-out nights 43 / 43, best fixed +1.39 / +1.59% vs hotel against +1.51 / +1.63% now (SE 0.11). The current world's variance excess is small once recorded requests are compared (1.08×), and removing it moves the level gain by about 0.1 point.
  - **Re-solving `k` with the old G3 formula** (world demand vs recorded variance) over-corrects: k ≈ 12, recorded variance 0.92×.
  - **Sold-out nights don't depend on `k`** (41–44 in every one-factor variant), so the excess over 34 isn't from the gamma shocks.
  - **Variants that fail:** factors per online-availability bucket come out nearly uniform and change nothing; ×8 on censored nights alone still leaves room-nights −9.5%.
- **Elasticity:** keep 1.0 as the main scenario. Per the hotel, its main competitor is usually 5–15% cheaper for this room type (sometimes more expensive), so ε well below 1 would imply implausible premiums; report about 0.8–1.5 as sensitivity.

Cross-file contracts: `bar_rates.py` imports `load_raw`, `SEASONS` and `SUMMER` from `ml_demand_model.py`, which in turn reads `bar_daily.csv` / `bar_bookings.csv` (`load_bar` fails loudly if they are missing or don't cover every booking: re-run `bar_rates.py` after changing `BOOKINGS.xlsx`, the rate CSVs or the load filters). `rl_agent.py` imports `PRICE_MULTIPLIERS` from the env. The env depends on the exact `sim_data.pkl` schema produced by `build_sim_data` (per season: `capacity`, `online_avail`, `dispersion_k`, `dates`, `lambda_true`, `lambda_forecast`, `hotel_price`, `p_ref`, calendar arrays, `profiles`, `actual`; top level: `seasons`, `models`), so after changing that schema, `BOOKINGS.xlsx` or `hotel_pms_data.csv`, you must re-run `ml_demand_model.py`. `ml_demand_model.py` imports `HotelEnv` (inside `calibrate_to_availability`) to calibrate. Model keys `"rf"` / `"xgb"` are shared across the ML, env and RL files.

## Future and proposed work

Nothing here is implemented. Importance: **P1** = needed before the final thesis run (correctness, realism or protocol); **P2** = likely to raise agent revenue or make the results more defensible; **P3** = if time allows; **Later** = beyond the thesis, described as a limitation. Effort: 1 (quick) … 5 (long). Test every item in a scratch copy, choose only on validation, tuning or holdout seasons (never test), and apply after approval. Items that change `online_avail` or the world need an ML + RL rerun, so do them together.

**Order:** world and data fixes (one ML rerun) → RL code fixes → reward test → tuning → benchmarks and robustness → final full run → docs.

**World and data** (P1 unless marked):
- **World calibration.** Choose the world by realism (recorded requests, their variance, room-nights, sold-out nights), never by RL revenue. Until it is settled, report the ε = 1 level gain as the range +0.9 to +3.1% with the reason (see the calibration findings).
  - **G3 after calibration (P1, 1–2):** compute G3 on the simulated recorded requests (accepted + cancelled that fitted) against the real ones, after `calibrate_to_availability`, and re-fit `dispersion_k` to it (2026: k ≈ 6.2 / 5.4). It moves the level gain by only about 0.1 point.
  - **Per-date calibration (P2, 3–4):** the prototype matches recorded requests per date and their variance. Open issues: 46–50 sold-out nights against 34 real, room-nights 2.5–3.5% short, and a result that depends on the cap (+2.1 to +3.1%). Next: re-fit `dispersion_k` together with the uplift, aiming at recorded variance and sold-out nights, then check whether the room-nights close.
  - **Censored likelihood (P3, 4):** an untested alternative. Scaling only censored nights doesn't work.
- **PMS coverage (1):** end the 2026 table at the PMS's last arrival (14 Oct) by filtering `bk` after `load_pms`. Today, 16 Oct opens all 17 rooms online and 15 Oct misses its offline arrivals.
- **Room 108 (1):** drop its 2026 rows in `load_pms`.
- **PMS gaps (1):** flag (`!!!`) nights where the PMS holds fewer rooms than confirmed online stays, which opens the whole hotel online. Describe it as a limitation; the true offline count is unknown.
- **Defensive PMS loader (P2, 2):** keep only statuses that occupy a room, and fail on unknown statuses, non-STD rooms, a changed header or missing coverage instead of clipping silently. Needed before any PMS file with future reservations.
- **Latent fixes (P3, 1):** `load_rules` uses `.replace("", np.nan)`; `occupancy_class` raises instead of returning `""`; `deviance_gain` gets a fresh `default_rng(BOOT_SEED)` per call, so its CIs are paired between models.
- **Input hash and tests (P3, 2):** `bar_rates.py` writes a hash of `BOOKINGS.xlsx` and both rate CSVs, and `load_bar` checks it. Add a small check script for the env's invariants: the same seed gives the same season, and demand is monotone in price.

**RL code:**
- **Batched evaluation (P1, 2):** step the seasons in lockstep, with one batched `predict` per day (bit-identical, 3–4× faster), in `ValidationCheckpoint`, the test runs and tuning.
- **Protocol fixes (P1, 1):** remove the "Best combination" line (a maximum on test); compute `rl_seed_spread_%` with `ddof=1`.
- **One process pool (P2, 1):** all ML × RL × seed jobs in one pool, about 25–35% less wall time.
- **Small (P3, 1):**
  - check the elasticity-log columns at the start of `main()` (today it fails only after all the training);
  - `SENSITIVITY_N_EVAL = N_TEST`, so the ε = 1 row matches the main result;
  - a wider multiplier grid in the sensitivity sweep at ε ≤ 0.8;
  - align the documented defaults (steps, seeds, jobs) with the code once the final budget is set.
- **Diagnostics (P3, 1):** log in `ValidationCheckpoint` (TensorBoard) the validation gain, a histogram of the chosen multipliers, mean price / occupancy / lost requests and the price–forecast correlation, plus PPO's KL, entropy and explained variance.

**Agent revenue** (the main aim; the per-date ceiling leaves at most about 0.5 points over best fixed):
- **Variance-reduced training reward (P2, 2):** train on `r(a) − r(best fixed)`, the second term a dry run (no booking) on the same pre-drawn randomness and state; evaluation is unchanged.
  - *Why:* the day-by-day signal is tiny (median action gap 4.6 € against a daily SD of 399 €), and most of that noise is shared by both prices.
  - `reset()` draws all randomness before pricing, so the baseline doesn't depend on the action; only multi-night stays add a small bias.
  - *Test:* plain vs this reward, 150k × 3 seeds, both algorithms, decided on holdout seasons. Run it before tuning.
- **Optuna tuning (P2, 3):** on the final world and reward.
  - **Gate first:** at the same final training budget, pretuned settings vs a 16-trial study per algorithm (150k × 3 seeds, rf world), decided on holdout seasons 400000–400199.
  - **If Optuna wins, run the full study:**
    - objective: mean % vs best fixed over the rf and xgb worlds;
    - 24 trials per algorithm, with the pretuned config enqueued first;
    - 3 tuning seeds (1001–1003, never the final seeds);
    - tuning seasons 300000–300099 (the calibration uses 200000+);
    - pruning on the validation curve, and the top 3 trials rescored with fresh seeds;
    - 300k steps per trial, about 4–5 hours.
  - Until it is adopted, install `optuna` in scratch only.
- **Budget gate (P2, 1):** train the chosen settings at 300k and 900k, scored on tuning seasons. Use 900k only if it is more than 1 SE better: LR decay and exploration are fractions of the run, and the curves plateau at 150–200k.
- **Seed ensemble (P2, 1):** a majority vote of all seeds' greedy actions (ties go to the median multiplier), as an extra row. Adopt it only if it beats the mean of seeds on validation. It uses every seed, so it isn't a best-seed pick, and it should damp the wandering between price levels.
- **Narrower action grid (P3):** 0.90–1.20× halved the RL loss in a 2-seed test. As the main grid it would change the level effect and comparability, so the price-range check covers it.

**Benchmarks and robustness** (the thesis story):
- **Per-date upper bound (P1, 2):** the per-date schedule (chosen on validation seasons; +0.6–0.75% over best fixed in the current world) as a results row, so RL is reported as a share of the day-by-day value available.
- **Robustness (P2, 2–3):** `--robustness`, evaluation only. It tests each checkpoint, the hotel and best fixed on the test seasons in:
  - the other ML world;
  - ε 0.8 / 1.2;
  - demand ×0.85 / ×1.15, through a `demand_scale` env argument that keeps common random numbers (×0.85 ≈ the uncalibrated world);
  - the 2025 offline calendar.

  If the agents prove fragile, train across worlds (domain randomisation).
- **Price-range check (P2, 1–2):** best fixed, the per-date bound and one PPO run with multipliers in the hotel's observed range (0.95–1.10×), to show how much of the level gain extrapolates `g(p)`. It matters more if a per-date world (best price 1.20–1.25×) is adopted.
- **Contextual-bandit baseline (P3, 2–3):** the MDP is close to one. It separates "RL is weak" from "there is little value to find".
- **Best fixed price:** keep it (no switch, not removed). It separates the elasticity-driven level effect from the RL's day-by-day value.

**Later** (state as limitations):
- **Booking lead time (5):** requests arrive over a booking window, and the agent reprices as the date approaches. It is the biggest realism gain and needs a new env. It also enables cancellations that hold rooms until they cancel (a candidate cause of the simulator's extra sold-out nights) and re-forecasting during the season (protocol D).
- **Booking-pace ML feature (3):** rooms on hand 45 days out (2026 median lead time 45 days, 75th percentile 89). Better forecasts haven't moved RL revenue so far (honest vs leaky forecast, 7-day observation), so expect little.
- **Learned price response (5):** needs competitor prices per date. Today one assumed ε covers every day and channel.
- **Varying offline calendars (3–4):** older PMS exports (2023–24) would let each episode draw a group calendar (weekly correlation 0.83 between years). Revisit the 7-night observation with it.
- **Second case study (4):** the Kaggle Resort Hotel, from the original CSV.
- **2024 bookings:** excluded. The online system opened late, so recording shrinks month by month and would distort the seasonal shape. They are usable only with a recording-completeness feature, which would also reopen the 2025 decision.
- **More demand features:** local events, flights, and school holidays in the source markets.
- **Smaller env items:** length of stay that reacts to price; offline stays that arrive late; the hotel's online allotment (usually 4–8 rooms, while the agent can sell every free room).
- **Data to ask the hotel for:** competitor prices per date, and older PMS exports.
