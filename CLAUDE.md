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
python bar_rates.py            # BOOKINGS.xlsx + rate_rules.csv + rate_occupancy.csv -> bar_exports/: bar_daily.csv, bar_bookings.csv, bar_inputs.sha256, bar_validation.txt, bar_daily.png, bar_daily_pct.png (+ .xlsx)
python ml_demand_model.py      # BOOKINGS.xlsx + bar_*.csv + hotel_pms_data.csv -> demand_models.pkl, sim_data.pkl, ml_comparison.csv, ml_cv_grid.csv, demand_fit.png
python rl_agent.py             # sim_data.pkl -> results_<season>_<model>_e<elasticity>_t<steps>/, comparison_<tag>.csv/.png, elasticity_log_<model>.csv
```

Evaluation only (no training; they read existing outputs and write only their own folder):

```
python ml_reference.py         # same inputs as ml_demand_model.py -> ml_reference/: ml_table.csv (ml_comparison.csv + GLM / Ridge reference rows), ml_reference_comparison.csv, ml_reference_cv_grid.csv, summary.txt
python robustness.py           # a finished run's checkpoints + sim_data.pkl -> robustness_<season>_e<elasticity>_t<steps>/: baselines.csv, significance.csv, robustness.csv, policy_map.csv, rooms_left_rule.csv, per_season.csv, reproduction.csv, summary.txt, + figure data per ML model (about 9 min with --jobs 6, the default)
```

Nothing runs them automatically: after an ML rerun, rerun `ml_reference.py`; after an RL run, rerun `robustness.py` (it refuses checkpoints from another world) and re-run and save the notebook below.

The price recommendation notebook, `pricing_tool.ipynb` (item 6), takes about 10 s and writes no files.
- **VS Code:** install the Jupyter extension, choose the `intellirate-cpu` kernel and Run All.
- **Browser:** open Anaconda Prompt in `base` (Jupyter is installed there, not in `intellirate-cpu`), run `jupyter lab pricing_tool.ipynb` and choose the kernel "Python [conda env:intellirate-cpu]". The default "Python 3" is base's Python, without the packages. Base's `nb_conda_kernels` finds the env only when `conda` is on the PATH.

`thesis_notes.md` (repo root, gitignored) is the brief for writing the thesis with Claude chat: aims, data, methods with their settings, results and limitations, in plain terms and without the code's internal labels. Update it whenever a result, setting or finding here changes, and keep the hotel anonymous in it.

Useful `rl_agent.py` flags for quick or partial runs:
- `--models rf` (or `xgb`) runs only one pipeline.
- `--algos ppo` (or `ddqn`) trains only one RL algorithm (default: all keys in `ALGOS`).
- `--timesteps 20000 --eval-every 5000 --seeds 42 --no-sensitivity` gives a fast smoke run (the default is 900k steps × 5 seeds, 42 / 7 / 123 / 1 / 2024, for each algorithm, validated every 25k steps, plus a sensitivity sweep). The `results_*` / `comparison_*` names carry the timesteps (`_t900k`, `_t20k`; `steps_tag`), so a smoke run doesn't overwrite a full run's outputs. It still appends a row to `elasticity_log_*`, and a rerun with the same season, elasticity, world and timesteps overwrites.
- `--season 2025` and `--elasticity 1.2` change the scenario.
- `--common-world rf` makes every pipeline use the same "true" demand model while each agent keeps its own forecast. The output directories and tags get a `_world-<m>` suffix.
- `--jobs 5` (the default) trains the seeds of each ML × RL combination in parallel processes, each with one torch thread. Parallelism is per seed, so at most `len(--seeds)` processes run. `--jobs 1` runs sequentially and gives identical results. A GPU doesn't help here: the networks are tiny and the bottleneck is Python overhead per step.

The repo has no tests, linter or build step. `.gitignore` uses patterns: the private data in any format (`BOOKINGS.*`, `hotel_pms_data.*`; see Other data files) and every generated artifact (`*.pkl`, `*.zip`, `bar_exports/`, `ml_comparison.csv`, `ml_cv_grid.csv`, `demand_fit.png`, `results_*/`, `comparison_*`, `elasticity_log_*.csv`, `robustness_*/`, `ml_reference/`, `logs/`) are ignored, so runs for any season, elasticity or world stay out of the repo. The two rate config files (`rate_rules.csv`, `rate_occupancy.csv`) are tracked. The pipeline runs without warnings (the Excel export's openpyxl style warning is silenced in `load_raw`); keep it that way rather than running with `-W ignore`. Check a change with `python -W error <script>`: a warning is easy to miss in the printed report, and with `-W error` the run stops at it. A warning raised while an object is garbage-collected (an unclosed file, for example) only prints as "Exception ignored" and doesn't stop the run, so also grep the log for `Warning`.

**Testing rule: tests, checks and experiments must not make any permanent change to the repo's files unless the user explicitly says so.** That covers code, configs and docs, and also the generated outputs (`bar_exports/`, `*.pkl`, `ml_comparison.csv`, plots, `results_*`, `comparison_*`, `elasticity_log_*`). Run them on a copy of the project in the scratch directory (copy the `.py` files, the two rate CSVs, `BOOKINGS.xlsx`, `hotel_pms_data.csv`, `bar_exports/` and any other inputs needed), patch only that copy, and report before applying anything. To show that a change leaves results alone, compare the copy's outputs with the repo's: the BAR outputs, the xgb arrays, `demand_models.pkl` and `demand_fit.png` are bit-identical between runs, but the rf numbers (`n_jobs=-1`) differ at about 1e-16 relative, so compare `sim_data.pkl` and the ML CSVs by value, not by file hash. The `.xlsx` files differ only in their creation time (`docProps/core.xml`). The rule also means no smoke runs of `rl_agent.py` in the repo, because they append rows to the elasticity logs and overwrite any run of the same length. Applying a change only after the user approves it, and then regenerating the outputs it requires, is fine.

Two project rules: only the **summer season** is studied (arrivals 1 April – 31 October, `SUMMER` in `ml_demand_model.py`), and the **hotel stays anonymous** — never write its name (it appears in the data's property, website and member-programme columns) into code, configs, outputs or docs.

## Architecture

There are four files, forming one pipeline: **BAR reconstruction → ML demand model → simulator → RL agent**, plus two evaluation-only scripts and a notebook that read its outputs (items 4–6).

0. **[bar_rates.py](bar_rates.py)** reconstructs the hotel's double-occupancy Best Available Rate (BAR) from the recorded prices. The rate system builds a price as `occupancy BAR × stacked discount factors`; the script unwinds that per booking:
   - [rate_rules.csv](rate_rules.csv): one row per discount/commission (NR, Genius, Last Minute Deal, Expedia promos and 18% commission, website direct offer / member programme), with detection column + regex, season, booking-date window, factor, optional `round_to` (rates rounded after discounts) and `confirmed`/`inferred` status. A `@infer` row (`genius_unflagged`) is decided per booking against the price ladder.
   - [rate_occupancy.csv](rate_occupancy.csv): the occupancy grid. 2025: no single rate, triple at a fixed price (no BAR information). 2026: per-level single and triple prices. Children aged ≥ 4 count as adults; only Booking.com records ages, so unknown ages are inferred from the price.
   - Double price levels count only when seen on ≥ 2 channel groups. Multi-night stays are tested against, and split by, the one-night-stay BAR (never against themselves). The daily BAR is the weighted median per stay night (cancelled bookings included).
   - Outputs, all in `bar_exports/` (`BAR_DIR` in `ml_demand_model.py`): `bar_daily.csv` (BAR per night, `source` observed / interpolated / filled), `bar_bookings.csv` (per booking: rules, occupancy, inference method, `bar`, `bar_ref`), and `bar_validation.txt` / `bar_daily.png`. `bar_daily_pct.png` is the same plot in % of the season's `p_ref` (the simulator's 1.00× price, the median daily BAR over the season's arrival dates). Each PNG also gets an `.xlsx` of the same name with the plot as editable Excel charts (`excel_plot`, styles copied from the matplotlib figure) and the plotted data per season; it never changes the PNGs or CSVs.

1. **[ml_demand_model.py](ml_demand_model.py)** loads `BOOKINGS.xlsx`, drops PII columns, and keeps room type `STD`, seasons 2025/2026, summer arrivals and arrivals on or before the export date; `main` then drops arrivals after the PMS's last arrival (see Booking data). It then builds a daily table of booking *requests* (cancelled bookings included), the hotel price (the daily BAR from `bar_daily.csv`, not smoothed), and calendar/holiday features. The holidays are Greek: Orthodox Easter, Whit Monday and national days. `add_availability` adds `online_avail` and `censored` (sold out or closed online: `sold ≥ online_avail`; 41 nights in 2025, 32 in 2026). Censored nights stay in training but are never scored, because their demand above the rooms on sale was never recorded.

   RF and XGBoost (Poisson objective) each play **two roles with separate settings** (protocol "E revised"; why, see ML forecaster in Experiment findings):
   - **Forecaster** (what the agent sees): settings from `GRID`, chosen only inside `TUNE_SEASON` = 2025 by `GroupKFold` blocked by ISO week, scored on uncensored days, best mean CV deviance (currently rf `min_samples_leaf=20, max_features=1.0`; xgb `max_depth=1, n_estimators=600`). It is then refit on 2025 and tested **once** on all of `TEST_SEASON` = 2026 (163 uncensored days) against the 2025 month mean, with 95% CIs from resampling whole ISO weeks. Out of time, neither model beats that baseline (gain CIs include 0); the level is about 0.54 of actual, but the env shows the agent only the pattern.
   - **World** (`lambda_true`): `WORLD_PARAMS` (rf `min_samples_leaf=3, max_features=0.6`; xgb `max_depth=3, n_estimators=400`), fitted on both seasons. It describes what happened, so it keeps the real day-of-week, holiday and peak structure; forecast-tuned settings would smooth it away.
   - `ml_comparison.csv` has one row per `evaluation`: `test_2026`, `baseline` and `pooled_cv_reference`. The last is the old pooled GroupKFold with neighbouring 2026 weeks in training: optimistic (gain +1.0), so report it only as an upper bound. `ml_cv_grid.csv` has the CV score of every config. Linear reference rows come from `ml_reference.py` (item 5).
   - **Guardrails (the run fails):**
     - **G1:** the tune and test seasons split the table, settings are chosen on `TUNE_SEASON` rows only, and every forecast comes from a model that never saw that date.
     - **G2:** at least `MIN_TEST_DAYS` uncensored test days.
     - **G3:** after `calibrate_to_availability`, in the world the agents train in, the daily variance of the simulated *recorded* requests at the hotel's prices matches the real requests' within `WORLD_VAR_TOL` (1%). If even pure Poisson is more volatile, the season is flagged instead.
     - **G4:** at the hotel's prices the simulator sells the real room-nights within `CALIB_TOL` (0.5%). G3 and G4 are solved together and must converge within `CALIB_ROUNDS`.
   - **Flagged checks (`!!!`, the run continues):**
     - a chosen config at an edge of `GRID` that is not a natural limit;
     - a forecaster significantly worse than the baseline;
     - a world with daily Spearman < 0.5, or any month outside 0.8–1.25 × actual before calibration;
     - a calibrated season where even pure Poisson is more volatile than reality (2025: 1.13× rf, 1.05× xgb).

   For each model it stores:
   - `lambda_true`: the world fit, rescaled so that each season's total matches the real total. It is then recalibrated by `calibrate_to_availability` (`simulate_hotel`, `CALIB_SEASONS = 200` seasons from seed 200000): one factor per season and model (2025 1.14 / 1.16, 2026 1.16 / 1.18, rf / xgb), chosen so that the simulator, at the hotel's prices and under the real online availability, sells the real number of room-nights (G4). It is solved together with `dispersion_k` (G3). Demand on nights closed online was never recorded. Neither depends on elasticity. The factor applies to every night, so in 2026 fully recorded (uncensored) nights record about 3% more requests than really arrived and censored nights about 29% fewer. It drives the simulated "world".
   - `online_avail`: rooms on sale online each night, `capacity − (PMS rooms − confirmed BOOKINGS rooms)` clipped to [0, capacity] (`load_pms`, `occupied_rooms`, `online_availability`). A room counts once per night (the PMS holds 4 room-nights entered twice, flagged `!!!`). `online_availability` fails if a table date lies outside the PMS's first-to-last arrival, and flags (`!!!`) the nights where the PMS holds fewer rooms than the confirmed online stays: 5 nights / 5 room-nights in 2025, 18 / 28 in 2026. There offline is 0 and the whole hotel opens online; the true offline count is unknown (a limitation). 2026: on average 10.5 of 17 rooms, with 11 nights closed. 2025: 8.8 of 18, with 15 nights closed.
   - `lambda_forecast`: the forecaster's predictions, never in-sample. 2025 is out-of-fold, and 2026 comes from the model trained on 2025. The agent sees these as its forecast.
   - `dispersion_k`: the negative-binomial overdispersion of the simulated world's gamma shocks, **one per season and model**, stored in `seasons[year]["dispersion_k"][model]`. `None` means pure Poisson. `train_models` gives a starting value: the `k` for which the world's total daily variance equals the real one before calibration, `Var(y) = E[fit] + Var(fit) + E[fit²]/k`, computed on the fit after the per-season rescale (the in-sample fit already holds part of the data's noise). `calibrate_to_availability` then re-fits it, with secant steps on 1/k and the factor re-solved at each `k`, so that the recorded requests match reality in the calibrated world (G3). Current values: 2026 6.23 (rf) / 5.88 (xgb); 2025 `None` for both, the flagged Poisson floor, because one factor also scales the fit's own variance by its square (1.14² ≈ 1.30). The `dispersion_k` column in `ml_comparison.csv` is the forecaster's 2025 OOF value: it sets the 80% interval coverage and describes forecast error, not the world.
   - Per-month pools of real booking profiles (rooms, nights, Total, orig_price, cancelled). `orig_price` is the booking's own BAR (`bar_ref`), so scaling `Total` to another BAR keeps the guest's occupancy, channel and plan factors. A month with fewer than 20 profiles falls back to the season pool.

   Everything goes into `sim_data.pkl`, keyed by `seasons[year]`.

2. **[hotel_season_env.py](hotel_season_env.py)**: `HotelEnv` is a Gymnasium env where one step is one arrival date and one episode is one season.
   - The action is one of 11 BAR multipliers (0.75–1.25×) of the season reference price `p_ref`, the median daily BAR.
   - Price response: `g(p) = exp(-elasticity·(p/p_ref − 1))`. `lambda_base = lambda_true / g(hotel_price)` removes the hotel's own price effect, so demand can be re-priced.
   - All randomness for the season (gamma shocks, uniforms for the counts and the profile picks) is drawn in `reset()`, before any price is set. Counts use inverse-CDF Poisson, so a given seed is monotone in price. The gamma shocks are inverse-CDF too (`gammaincinv`), so every season uses the same number of draws: a seed gives the same season for any `k` and any world, and its shocks change smoothly with `k` (`rng.gamma` used a variable number of draws). This gives paired, common-random-number comparisons between policies and between worlds. Keep this property when you change the env.
   - Each request samples a real booking profile. Cancelled profiles give no revenue, and profiles that don't fit under `capacity` over their nights are lost. `info["recorded"]` counts what the booking data would hold: the rooms of accepted requests plus cancelled requests whose stay would have fitted (`cancelled` and `lost` count cancelled requests whether they fit or not). Capacity is the season's physical STD rooms, stored per season in `sim_data.pkl` (`CAPACITY` in `ml_demand_model.py`, the length of `STD_ROOMS`): 18 in 2025 (including room 108) and 17 in 2026 (room 108 was renovated and is no longer STD). `reset()` pre-fills `committed` with the real offline stays (`capacity − online_avail`), so online sales are limited to the real online availability. The offline calendar is the same in every episode; only online demand is random. Revenue is the profile's real `Total` scaled by `price / orig_price`.
   - The observation has 9 dims in [-1, 1]: forecast, committed rooms today and tomorrow (offline stays included), cyclic month and day of week, holiday flag, and season progress.
   - `HotelEnv(..., sim=dict)` takes the sim data directly; the calibration uses this.
   - `demand_scale` (default 1.0, bit-identical to before) multiplies `lambda_base`, with the pre-drawn randomness unchanged. Only `robustness.py` sets it.
   - Baselines use `step_price(price)` to pass an arbitrary price, for example the hotel's BAR. `dry_run(price)` returns the reward date `t` would earn at a price, from the same pre-drawn randomness and calendar, without booking anything (the training baseline in `rl_agent.py`).
   - **The env is the project's MDP** (the code never names it). The state is the day `t` plus the `committed` calendar, the action is the multiplier, the reward is revenue minus variable cost, the horizon is finite (one season) and γ = 0.9 is set in `rl_agent.py` (see RL design decisions). Strictly it is a POMDP: the observation shows committed rooms only for tonight and tomorrow, so longer stays are partly hidden. With a mean stay of 1.37 nights (71% one-night stays; the `1.26` in `rl_agent.py`'s comment is outdated), today's price barely affects later states, so the problem is close to a contextual bandit.

3. **[rl_agent.py](rl_agent.py)** runs one pipeline per ML model, and each pipeline trains every RL algorithm:
   - RL algorithms live in the `ALGOS` registry (`key -> (SB3 class, params, display name)`). `train_agent` is generic: SB3, CPU, `VecNormalize` on reward only, which is also valid for Double DQN because off-policy SB3 normalises replay-buffer rewards at sample time. To add an algorithm, add a registry entry.
   - **Training reward (`BASELINE_REWARD`, on for both algorithms since 22 Oct):** `BaselineReward` trains on revenue minus the best fixed price's `dry_run` revenue on the same day. The demand noise both share cancels (the day-by-day signal is about 5 € against a 399 € daily SD), and the subtracted term doesn't depend on today's action; only multi-night stays add a small bias. `Monitor` sits inside the wrapper, so `monitor.csv` and the learning curves keep the true revenue. Validation and test always use plain revenue.
   - Double DQN (`"ddqn"`, which replaced DQN; see RL design decisions) is `DoubleDQN(DQN)`. SB3 has no Double DQN, so it overrides `train()`, copied from SB3 2.8.0's `DQN.train` (re-check it after an SB3 upgrade). Only the next-state value changes (`next_q_values`): the online net selects the action and the target net evaluates it.
   - It validates the simulator: the hotel's prices in the sim are compared with the real season. Occupancy and RevPAR in `results.csv` are over the online available room-nights.
   - **Train / validation / test (protocol `val-test`).** All three are simulated seasons of the same world, told apart by seed:
     - **Training:** each training seed's own SB3 stream (about 4,600 seasons at the default 900k steps).
     - **Validation:** seeds from `VAL_SEED0 = 100000`, the old evaluation seasons, which past design choices already used. It chooses things and is never reported as a result:
       - `ValidationCheckpoint` runs the greedy policy on the first `N_VAL = 50` seasons every `--eval-every` steps (default `EVAL_EVERY = 25_000`) and at the end of training. It saves the best checkpoint as `<algo>_seed<n>.zip`, and that checkpoint is the one tested, not the final one.
       - The best fixed price is chosen on `N_VAL_FIXED = 200` validation seasons (no training, so more seasons are cheap; 50 sometimes missed a near-tie).
       - The per-date schedule is chosen on `N_VAL_SCHEDULE = 1000` validation seasons. The seed ensemble is compared with the mean of seeds on seasons from `ENSEMBLE_SEED0 = 100050` (`N_ENSEMBLE_VAL = 200`), which checkpoint selection never used.
     - **Test:** `N_TEST = 200` seasons from `TEST_SEED0 = 500000`, shared by every policy (common random numbers). Every reported number comes from them, including the simulator check. Never choose anything on them: design choices use validation only, and the test is looked at once.
     - The callback has its own envs, and a deterministic `predict` draws no random numbers, so training is identical to a run without it (checked: identical `monitor.csv` for all 12 agents).
     - Agents are evaluated with `run_batched`: the seasons in lockstep, one batched `predict` per day, 3–5× faster than one season at a time. Results are the same up to float rounding, which can flip a near-exact tie between two actions.
   - It finds the best fixed price once per ML model, then trains each algorithm for each seed.
   - It reports the gain over the hotel split into a price-*level* effect (best fixed vs hotel, driven by the elasticity assumption) and a *day-by-day* effect per algorithm (RL vs best fixed). A negative day-by-day number means the agent hasn't converged.
   - **Benchmarks** (no training; chosen on validation, reported on test):
     - the per-date schedule (`action_sweep` → `best_schedule`: the best level per date). It is a benchmark, not a ceiling: a policy that sees the rooms left can beat it.
     - the best fixed price and the per-date schedule within the hotel's own price range (`observed_range`: 0.95–1.10× in both seasons), where `g(p)` is not extrapolated.

     Every row has `above_range_%`, the share of days priced above the hotel's highest price. The comparison adds `schedule_vs_best_fixed_%` and `best_fixed_in_range_vs_hotel_%`.
   - **Seed ensemble** (an odd number of seeds, at least 3): the median of the seeds' greedy actions each day, as an extra row per algorithm, plus the comparison columns `ensemble_vs_best_fixed_%` and `ensemble_beats_seeds_on_val`. Adopt it only if it beats the mean of seeds on validation; the `rl_*` columns stay mean-of-seeds.
   - **Error bars:** each seed row has the paired SE over the test seasons. The mean-of-seeds rows and the `comparison` columns `rl_vs_hotel_se_%` / `rl_vs_best_fixed_se_%` combine the variance between training seeds with the season noise (`seed_se`: `sqrt(Var_seeds/n + season_se²)`). The `*_ci95_%` columns use t(n−1), so few seeds give wide CIs (t = 2.78 at the default 5, 4.30 at 3). Report the mean of all seeds, never the best one. More seeds need no code change (`--seeds ... --jobs n` on a machine with enough cores).
   - It also sweeps elasticity for the best fixed price, with no training: chosen on validation, reported against the hotel on all `N_TEST` test seasons (`SENSITIVITY_N_EVAL`), so its ε = 1.0 row equals the main result. A best fixed price at the edge of the grid (0.75 or 1.25×; 1.25× at ε ≤ 0.8) is flagged `!!!` here and in the main result. The grid is not widened: 1.25× already lies 13% above the hotel's highest price.
   - Per ML model it writes `results_*/` containing:
     - `results.csv` (test numbers; seed rows add `checkpoint_step`, `val_reward`, `val_vs_hotel_%`);
     - `validation_curve.csv`;
     - `summary.txt`;
     - `learning_curve_<algo>.png` (top: training episodes, exploration included; bottom: greedy validation curve, with the selected checkpoint starred);
     - `price_path.png`;
     - `<algo>_seed<n>.zip` (the selected checkpoint).

     `comparison_<tag>.csv/.png` gets one row per ML × RL combination, with `rl_*` columns.
   - Every run appends one row per ML model to `elasticity_log_<model>.csv` (`append_elasticity_log`). Each row has the timestamp, the `protocol` (`val-test`), `sim_data` (the first 12 hex digits of `sim_data.pkl`'s SHA-256: rows with the same value come from the same world, and every ML rerun gives a new one, since rf refits differ by about 1e-16), season, elasticity, timesteps and seeds, the hotel revenue, the best fixed multiplier and its revenue, plus each algorithm's mean-of-seeds revenue and occupancy, all taken from `results.csv`. It tracks results across runs at different `--elasticity` values. A rerun adds a row instead of replacing one, and a skipped algorithm leaves its columns empty. If an existing log's columns don't match `log_columns()`, the run fails at the start, before any training: after a column change, rename the old logs (`_pre-valtest`, `_pre-simdata`). The rows of older worlds, before the 29 Oct rerun, are in `elasticity_log_<model>_pre-0104c77364ca.csv`. Runs in the repo append to it too, which is another reason to keep smoke runs in a scratch copy.

4. **[robustness.py](robustness.py)** (evaluation only, no training) re-evaluates a finished run's saved checkpoints. Defaults: the final run (season 2026, ε 1.0, 900k, 5 seeds).
   - **Fingerprint check (fails before any work):** the elasticity log's last row of the run must carry the current `sim_data.pkl` hash and `results.csv`'s revenues, and every checkpoint must exist. All five 29 Oct runs (`results_2026_*_e<ε>_t900k`, ε 0.6 / 0.8 / 1.0 / 1.2 / 1.5) pass it, and each has its own `robustness_2026_e<ε>_t900k/`.
   - **Reproduction check:** the nominal world must rebuild every value of `results.csv` and the comparison exactly (`reproduction.csv`), otherwise it stops.
   - **Rooms-left rule:** one level per bucket of rooms left tonight (`RULE_BUCKETS`), monotone, read from `obs[1]` only. Coordinate ascent from best fixed on the best-fixed validation seasons.
   - **Worlds (`WORLDS`, every policy deployed unchanged):** nominal, the other ML world (own forecast kept), ε 0.8 / 1.2, demand ×0.85 / ×1.15. Best fixed is reported at the nominal level and re-chosen on validation per world (an upper reference).
   - **Significance:** paired over the 200 test seasons. Mean of seeds: `seed_se`, t(n−1); seeds and ensembles: z, descriptive. Holm (FWER 5%) over the primary family: mean of seeds × algorithm × ML model × {hotel, best fixed, schedule, rule}.
   - **Outputs:**
     - `baselines.csv`: the thesis table, with `season` and `elasticity` columns so the runs' tables stack;
     - `significance.csv`;
     - `robustness.csv`;
     - `policy_map.csv`: multiplier by rooms left tonight and forecast tercile, tidy;
     - `rooms_left_rule.csv`;
     - `per_season.csv`: the raw data behind every number;
     - `reproduction.csv` and `summary.txt`.
     - **Figure data** for Excel, per ML model (`<name>_<model>.csv`, nominal world). They have one column per series, and a column named after a policy is its price in EUR (its revenue in the learning curves):
       - `price_path`: the season of `price_path.png` (test seed 500000), with the benchmarks, the rule and the ensembles added;
       - `per_date`: each policy's mean price and rooms left per date over the test seasons;
       - `capacity_curve`: mean price and multiplier by rooms left tonight;
       - `learning_curve_train` / `learning_curve_val`: the data of `learning_curve_<algo>.png`, reference lines and selected checkpoints included.

     `--jobs 1` and the default `--jobs 6` give identical files.

5. **[ml_reference.py](ml_reference.py)** (evaluation only) adds reference forecasters to the ML table, validated exactly like rf / xgb (protocol E revised):
   - **Models:** a Poisson GLM and a Ridge regression, with an L2 grid.
   - **Encoding:** one-hot month and weekday plus the three holiday flags. Not `season` (constant in 2025) or the day-of-year trend.
   - They never enter `sim_data.pkl`, so `ml_comparison.csv` stays as it is. Adding them there would mean an ML rerun, which gives a new `sim_data` hash and breaks the checkpoints' fingerprint.
   - **Guards (it fails otherwise):**
     - its daily table equals `sim_data.pkl`'s;
     - its CV loop reproduces `ml_cv_grid.csv` for xgb;
     - the baseline row equals `ml_comparison.csv`'s.

6. **[pricing_tool.ipynb](pricing_tool.ipynb)** (evaluation only) is the prototype of the thesis's price recommendation tool, requested by the supervisor.
   - **Input:** an arrival date of 2026 and the rooms left tonight (optionally tomorrow; by default only the real offline stays are taken then).
   - **Output:**
     - the BAR recommended by the PPO seed ensemble (rf), next to the hotel's BAR and best fixed;
     - every ML × RL ensemble's price with its seeds' min–max;
     - a what-if row over rooms left;
     - the price path of test season `TEST_SEED0`.
   - **How it works:** it reads the 900k checkpoints and traces them with `robustness.check_checkpoints`. It builds the observation by resetting the env and setting `committed` for tonight and tomorrow, then calling `_obs()`; a change to the observation must be reflected there.
   - `%matplotlib inline` comes after the loading, because `rl_agent` and `ml_demand_model` switch matplotlib to Agg.
   - The committed notebook keeps its outputs: they are the example a reader without the data sees.

## Booking data and rates

`BOOKINGS.xlsx` is a private channel-manager export. Every row is in EUR, and `Total` is the room charge only (`Total == Stay`). `load_raw` keeps the recorded rate `price = Total / Room-Nights`; all rate normalisation lives in the BAR stage (`bar_rates.py` + the two rate CSVs). Change rates there, not in the ML/RL files. Rate structure (details and status per rule in the CSVs):
- Channel: `Source`, except website bookings (`Application == "WEBHOTELIER"`, `Source` empty or Google Free Booking Links), which form the "Direct" channel. Expedia Collect `Total` is net of 18% commission; all other channels are gross (Booking.com commission is not removed).
- Rate parity across channels. NR is −10% everywhere. Booking.com Genius (−10%, flagged "Genius Booker" in `Channel Notes for Hotelier`), the Booking.com Last Minute Deal (−5%, in `Room Type Name`), the Expedia promos in `Rate` (Blue Member −10% in 2025, Member / Mobile −7% in 2026) and every other promo stack by multiplication. 2025 website: permanent −12% direct offer. 2026 website: member programme (`SmartGuest`) −10%, or −14.5% for bookings made from 1 Sep 2026 (those rates are rounded to the euro).
- Occupancy pricing: see `rate_occupancy.csv`. BAR always means the double-occupancy BAR.
- Flexible bookings cancel at a rate of about 36%, NR bookings at about 2%. `Cancel Fee` is always 0. In the simulator a cancelled profile earns nothing and never holds capacity.
- The 2026 season ends at the PMS's last arrival (14 Oct, 195 days), the earlier of that and the BOOKINGS export date (16 Oct, the latest `Booking Date` in the refreshed export): offline stays are unknown after it, so the 9 bookings arriving 15–16 Oct are dropped in `main` (`bar_rates.py` keeps them; `p_ref` is 160 € either way). 2025 runs to 31 Oct. The daily table starts at each season's first actual arrival (3 Apr in both seasons), not on 1 Apr.
- The hotel's online booking systems opened late in 2025, on about 1 Apr. About a third of the 2025 demand is therefore unrecorded, including about three quarters of April. **Decision: 2025 is kept as recorded, with no correction.** Treat 2026 as the main RL season, and describe 2025 as a limitation rather than fixing it.

Other data files. Only the PMS file is used by the pipeline, and every private one is gitignored by pattern (`BOOKINGS.*`, `hotel_pms_data.*`, `avl-status*`, `kaggle_data.*`). `avl-status*` and `kaggle_data.*` are currently kept outside the repo folder.
- `hotel_pms_data.csv`: the hotel PMS export of every STD stay, online and offline (arrivals 1 Apr 2025 – 14 Oct 2026). Format: Greek cp1253, `;`-separated, quoted. Title and filter lines come first, the header is on line 4, blank lines separate the records and two footer lines close the file. 28 rows carry one stray empty field after the arrival date. Columns: number, arrival, departure, room, price list (`Group` / `open` / `Compl` / `Over`), price, status, adults, children and source. There are no guest or debtor columns, but print the title and filter lines only by shape and save only per-night aggregates. The source tag misses about 30% of online stays, so identify online stays by subtracting BOOKINGS, not by the tag. **Used only for per-night online availability:** `load_pms` reads arrival, departure, room and status, and nothing else. It fails on a changed header (`PMS_HEADER`), a malformed row, departure ≤ arrival, a status not in `PMS_STATUS` (today only `IN` / `OUT`, both holding a room; add new ones there, e.g. reservations in a file with future stays) or a room that is STD in no season (`STD_ROOMS`). Groups mostly booked before online sales opened, and offline bookings are typed in later from a paper calendar, so PMS entry dates are not booking dates. Room 108 is STD in 2025 only, as above; its 3 one-night stays in 2026 are dropped by `load_pms`. Confirmed online STD stays are missing from the PMS on 18 nights of 2026 (28 room-nights) and 5 of 2025, probably upgrades or room moves (flagged, see `online_avail`).
- `avl-status*.xlsx`: the booking engine's final availability per room type and night (1 Apr 2025 – 31 Oct 2026). `Classic Room` is the STD row. Cell fill: green = on sale, red = stop-sell, grey = 0. Values are exact only on 0 and red nights; elsewhere they are a lower bound. Online STD inventory is usually only about 4–8 rooms, and STD was sold out or stop-sold online on about 54 summer nights in 2026 (67 in 2025). Its file name and row 2 contain the hotel's name: match it by pattern, never print row 2, and keep it out of git.
- `kaggle_data.xlsx`: the public Hotel Booking Demand dataset (Portugal, 2015–17). It is a candidate second case study (the Resort Hotel) for reproducibility, not training data. Its `adr` column is corrupted by the Excel conversion, so use the original CSV.

## Experiment findings (season 2026, elasticity 1.0 unless stated)

Grouped by topic; each item gives the current conclusion. Numbers are rf / xgb on the test seasons unless stated. Superseded intermediate runs are in git history.

### Final results (the thesis numbers)

The 29 Oct run, after a clean BAR + ML rerun (it reproduces the 22 Oct run exactly, every number below included): 900k × 5 seeds, CRN reward, G3 world (`sim_data` 0104c77364ca), in `results_2026_*_e1.0_t900k`, `comparison_2026_e1.0_t900k` and `robustness_2026_e1.0_t900k/`. Mean-of-seeds rows have 95% CIs that include the seed variance:

| Policy | vs hotel | vs best fixed |
|---|---|---|
| Best fixed (1.15×) | +1.50 / +1.64% | – |
| Best fixed in the hotel's range (1.10×) | +1.34 / +1.44% | −0.16 / −0.20% |
| Per-date schedule | +2.16 / +2.28% | +0.65 / +0.63% |
| Rooms-left rule | +1.97 / +2.02% | +0.46 / +0.38% |
| PPO, mean of seeds | +2.11 ± 0.30 / +2.25 ± 0.30% | +0.61 ± 0.26 / +0.60 ± 0.21% |
| PPO, seed ensemble | +2.17 / +2.31% | +0.66 / +0.65% |
| Double DQN, mean of seeds | +1.98 ± 0.31 / +2.15 ± 0.35% | +0.48 ± 0.18 / +0.50 ± 0.21% |
| Double DQN, seed ensemble | +2.16 / +2.28% | +0.65 / +0.63% |

- **The gain has two parts.** The price *level* (best fixed vs hotel) comes from scarce rooms and the assumed ε; the *day-by-day* part (RL vs best fixed) is what the RL adds. About 90% of the level gain holds inside the hotel's price range, but all the day-by-day value needs prices above it: inside the range the per-date schedule gains nothing over 1.10× fixed.
- **Significance** (mean of seeds, Holm over 16 tests): both algorithms beat the hotel and best fixed (Holm p ≤ 0.028) in both worlds and don't differ from the per-date schedule (−0.03 to −0.18%). Their lead over the rooms-left rule (+0.01 to +0.22%) is not significant after Holm (largest XGB + PPO: p 0.012, Holm 0.10). The ensembles beat the rule by +0.18 to +0.28% (p ≤ 0.022), but their SE holds season noise only, so this is descriptive.
- **The seed ensemble is the adopted policy:** it beat the mean of seeds on validation in all four combinations (the rule set before the run). Report the mean of seeds next to it.
- **The agents learned a capacity rule.** The PPO ensemble's multiplier falls smoothly with the rooms left tonight (1.25× at ≤ 5, 1.15× at 8, 1.01× at 17), and the PPO seeds average 1.12× in every forecast tercile, so the forecast is ignored. Double DQN does the same, less sharply and at higher prices. The rooms-left rule chosen on validation: rf 1.25× at 0–7 rooms left, 1.15× at 8–10, 1.05× at ≥ 11; xgb 1.25× at 0–4, 1.20× at 5–7, 1.15× at 8–10, 1.10× at 11–13, 1.05× at ≥ 14. The RL finds that rule by itself and reaches the per-date schedule (PPO 94–96% of its value), but a simple rule gets most of the day-by-day value.
- **Behaviour vs the hotel:** ADR 151–158 € against 136 €, online occupancy 32–33% against 36%, lost requests 55–69 per season against 85–95. Prices above the hotel's highest BAR on 42–45% (PPO) and 54–62% (Double DQN) of days. Checkpoints chosen at 75k–901k (PPO) and 350k–825k steps (Double DQN); seed spread 0.03–0.11 points.
- **Sim vs real at the hotel's prices:** revenue −0.6 / −0.4%, bookings +0.4 / +0.7%.
- **Sensitivity (best fixed vs hotel):** +10.8 / +10.9% (ε 0.6), +5.9 / +6.1% (0.8), +1.5 / +1.6% (1.0), −0.1 / −0.1% (1.2), +1.2 / +1.3% (1.5). The best price is 1.25×, the grid edge, at ε ≤ 0.8. At ε 1.5 the gain stays small because a lower price can't fill rooms that offline stays hold.

### Agents trained at each elasticity (29 Oct runs)

Same settings as the main run, one run per ε, `results_2026_*_e<ε>_t900k` and `robustness_2026_e<ε>_t900k/`. % vs best fixed:

| ε | Best fixed (vs hotel) | Per-date schedule | Rooms-left rule | PPO mean / ensemble | Double DQN mean / ensemble |
|---|---|---|---|---|---|
| 0.6 | 1.25× (+10.81 / +10.89) | −0.01 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00, 0.00 / 0.00 | −0.01 / −0.01, 0.00 / 0.00 |
| 0.8 | 1.25× (+5.86 / +6.08) | −0.12 / −0.15 | 0.00 / 0.00 | 0.00 / 0.00, 0.00 / 0.00 | −0.01 / −0.03, 0.00 / 0.00 |
| 1.0 | 1.15× (+1.50 / +1.64) | +0.65 / +0.63 | +0.46 / +0.38 | +0.61 / +0.60, +0.66 / +0.65 | +0.48 / +0.50, +0.65 / +0.63 |
| 1.2 | 1.00× (−0.06 / −0.07) | +1.22 / +1.48 | +0.92 / +1.01 | +1.13 / +1.33, +1.14 / +1.35 | +1.04 / +1.24, +1.32 / +1.44 |
| 1.5 | 0.85× (+1.20 / +1.26) | +2.46 / +2.47 | +1.61 / +1.57 | +2.21 / +2.19, +2.22 / +2.23 | +2.34 / +2.35, +2.49 / +2.48 |

- **At ε ≤ 0.8 the grid cap leaves no day-by-day value.** Best fixed is 1.25×, the agents and the rule price there on (almost) every day, and the whole gain over the hotel is the level.
- **The day-by-day value grows with ε.** Above 1, the unconstrained optimum `p_ref/ε` lies below the hotel's level, so nights with free rooms call for low prices and scarce nights for high ones. Vs best fixed, the mean of seeds is Holm-significant at ε 1.2 (+1.04 to +1.33, Holm p ≤ 0.004) and at ε 1.5 (+2.19 to +2.35, Holm p ≤ 0.0002). Vs the hotel: +0.98 to +1.27% at ε 1.2, +3.43 to +3.64% at ε 1.5. The agents price lower than the hotel there: ADR 131–133 € (ε 1.2) and 114–116 € (ε 1.5) against 136 €, online occupancy 37.3–37.8% and 43.9–44.5% against 35.9%.
- **Only at ε 1.5 do the agents beat the rooms-left rule after Holm** (+0.59 to +0.77, Holm p ≤ 0.006). At ε 1.2 their lead is +0.12 to +0.32 (smallest Holm p 0.063). They never beat the per-date schedule, and at ε 1.5 XGB + PPO falls significantly below it (−0.28, Holm p 0.046). Double DQN overtakes PPO at ε 1.5.
- The ensemble beats the mean of seeds on validation in every combination except rf PPO at ε 0.6 and 0.8, where both sit at 1.25× (exact ties).
- **Robustness at ε 1.2 and 1.5:** the gain vs best fixed holds in the other ML world and with demand ±15% (+0.82 to +1.68 at ε 1.2, +1.79 to +2.85 at ε 1.5, every mean-of-seeds and ensemble row). Deployed at another ε, every run keeps its own price level and loses to a re-chosen best fixed, by up to 5.8% (ε 1.2 agents in the ε 0.8 world) and 10.9% (ε 1.5 agents in the ε 0.8 world).

### Robustness (`robustness.py`; every policy deployed unchanged)

The ε 1.0 run; the other runs are in Agents trained at each elasticity. PPO mean of seeds, % vs best fixed at the nominal 1.15×:

| World | Gain |
|---|---|
| Nominal | +0.61 / +0.60 |
| Other ML world | +0.51 / +0.67 |
| Demand ×0.85 | +0.42 / +0.67 |
| Demand ×1.15 | +0.53 / +0.72 |
| ε 1.2 | +1.51 / +1.46 (+0.28 / +0.38 vs the re-chosen 1.00×) |
| ε 0.8 | −0.26 / −0.22 (−1.70 / −1.74 vs the re-chosen 1.25×, the grid edge) |

- **The day-by-day gain survives another world and demand ±15%:** +0.31 to +0.74% for every mean-of-seeds and ensemble row.
- **Elasticity is the weak point.** Deployed unchanged, the agents keep their ε = 1 price level. At ε 0.8 they trail a re-chosen 1.25× by 1.1–1.8% (while still beating the hotel by 4.0–4.9%); at ε 1.2 Double DQN falls below the re-chosen 1.00× (−0.26 / −0.37%). Agents trained at the true ε don't have this problem (see Agents trained at each elasticity); training across ε would cover an unknown ε (future work).
- The ensembles beat the mean of seeds in almost every world (the exceptions are within 0.07 points). The re-chosen best fixed rows equal `sensitivity.csv`.

### Where the level gain comes from, and its caveats

- **Scarcity from the PMS offline stays** (groups ≈ 57–63% of STD room-nights). With BOOKINGS-only availability (a constant 17 rooms) rooms rarely ran short, best fixed ≈ the hotel (+0.05–0.10%) and the agents stayed below it. Lowering the capacity instead, or a per-night capacity from the availability report, raised the gain only as fast as realism fell (under-booking 11–19%).
- **ε = 1 is a neutral assumption.** Unconstrained revenue `p·g(p)` peaks at `p_ref/ε`, so at ε = 1 the hotel's median BAR is optimal when rooms are plentiful, and the whole gain comes from scarcity and day-by-day pricing. This also explains the sensitivity edges (at ε 0.8 the optimum is exactly 1.25×).
- **It extrapolates `g(p)`.** The hotel's daily BAR stays within 0.92–1.11× `p_ref` in 2026 (147–177 €; 2025: 0.92–1.13×), while best fixed is 1.15× (184 €).
- **Its size depends on the hidden demand on full or closed nights,** which the data can't identify. At ε = 1: +0.9 / +1.2% uncalibrated, +1.5 / +1.6% in the applied one-factor world, +2.1 to +3.1% with per-date calibration (uplift cap 1.5–3; scratch prototype). Report the range +0.9 to +3.1% with this reason. Without calibration the simulator under-books by 11–13%, because recorded requests are already net of lost guests and the simulator loses requests again.
- **The world remembers 2026's days** (`lambda_true` ranks them at Spearman 0.74 / 0.70, against 0.27 for an out-of-bag RF and 0.35 for the forecast), but the day-by-day gain doesn't rely on it. In worlds with forecast- or OOB-shaped demand the PPO agents keep +0.43 / +0.56% vs best fixed (rf, seasons 700000+, +0.51% in the training world), and a per-date schedule chosen in a shaped world keeps about two thirds of its value.
- **Revenue basis:** `Total` is gross for Booking.com and net for Expedia Collect. Every policy gets the same channel mix, so relative gains are unaffected; state it as a limitation.

### Simulator realism (the applied one-factor G3 world, at the hotel's prices)

- `k` 6.23 / 5.88, variance of recorded requests 0.996 / 0.997× real, room-nights −0.5 / −0.3%.
- **The remaining gap is sold-out nights:** 42 / 43 per season against 32 real; about 25 fall on real sold-out nights and about 17 on nights that had rooms left. Recorded requests are 1.03× real on uncensored dates and 0.71× on censored ones; one factor can't fix both, and `k` doesn't move sold-out nights. Scaling only censored nights fails (×8 still leaves room-nights −9.5%), and so do factors per availability bucket. The per-date calibration prototype is in Future work.

### ML forecaster (why protocol E revised)

- **Out of time, no forecaster beats the 2025 month mean** (163 uncensored 2026 days; deviance gain, CIs include 0): rf −0.11, xgb −0.03, Poisson GLM +0.14 [−0.08, +0.39], Ridge +0.13 [−0.09, +0.36]. The GLM has the best pattern deviance (2.19). From 15 Aug the baseline is significantly better.
- **The error is the level, not the pattern.** Calendar-only models forecast 2026 at 55–67% of actual, because Jul–Oct 2026 ran 65% above 2025 and trees can't extrapolate growth. The env normalises the forecast by its maximum, so the agent sees only the pattern. A level-corrected baseline (2025 month mean × the spring 2026/2025 trend) beats both models clearly (deviance 2.45 vs about 3.1) but overshoots the level by 24%. NB 80% intervals cover only 66–77% of 2026 days.
- **The pooled CV's gain (+1.0, significant) is leakage** from neighbouring 2026 weeks; report it only as an upper bound.
- **The forecaster and the world need different settings.** The one-SE rule falls to the simplest corner; used for the world it drops day of week and holidays, misses the August peak and puts the variance 20–28% off. The CV surface is flat (11–14 configs within 1 SE; Monte Carlo CV chooses the same settings).
- **Other protocols:** D (train to 30 Jun 2026, test Jul–Oct) has a better level and interval coverage but not a better pattern; it becomes the realistic choice only if the env re-forecasts during the season. Validating on Sep–Oct 2025 has too few days and extrapolates; a 48-day 2026 validation contradicted its test.
- **The honest forecast cost the RL nothing** (every change within the seed spread), and better forecasts haven't moved RL revenue so far.
- **If censored nights are ever dropped from training:** rescale `lambda_true` to the real total over uncensored nights only, and shrink a month × weekday baseline toward the month mean.
- **PMS data as demand training data: no.** Group stays move against online requests (weekly correlation −0.4 to −0.6), and every PMS index forecasts 2026 worse than the 2025 online month mean. The offline calendar repeats strongly between years (weekly correlation 0.83, groups 0.87), so older PMS seasons are useful only for modelling offline stays.

### RL design decisions (each chosen on validation, tuning or holdout seasons)

- **Double DQN replaced DQN:** with identical settings it removed DQN's seed instability (rf: mean −0.42 → +1.10% vs hotel, seed spread 1.07 → 0.39 points; xgb unchanged).
- **Hyperparameters (300k grid, rf):** PPO γ 0.9 / GAE λ 0.5 (−0.87 → −0.10% vs best fixed, seed SD 0.40 → 0.12): the problem is close to a contextual bandit, and γλ = 0.94 added about 17 days of unrelated demand noise to each advantage. Double DQN: batch 512 + γ 0.9 + soft target updates (τ 0.005) + linear LR decay, tested only together (−1.44 → −0.62). LR decay and `exploration_fraction` are fractions of the run, so longer runs stretch both. Optuna wasn't worth it then (trial noise 0.2–0.4 points against differences of 0.1–0.3).
- **CRN training reward** (rf, 300k, holdout seasons 400000–400199): +0.28 ± 0.12 (PPO) / +0.62 ± 0.16 (Double DQN) points vs best fixed over plain revenue, and every seed trained with it beat every seed trained on revenue. At 900k: PPO +0.48 → +0.61 and Double DQN −0.18 → +0.48 vs best fixed (rf). Results and log rows before 22 Oct used plain revenue.
- **Checkpoint selection:** the final checkpoint was the old protocol's real problem; the validated one beat it on fresh seasons by +0.72 / +0.82 points (up to +2.4). The selection isn't optimistic (validation − test −0.09 / −0.04 points).
- **Budget 900k:** in a 900k run, 6 of 10 PPO seeds chose a checkpoint at ≥ 700k steps, and the validation mean over 600–900k was 250–310 € above that over 150–300k.
- **Observation (9 dims):** 7 nights of committed rooms and forecasts (20 dims) gave no gain. Every episode uses the same offline calendar, so the calendar features already encode upcoming group blocks, and with a mean stay of 1.37 nights tonight's free rooms carry almost all the signal. A variable cost (€20 / room-night) only raised the best price level.
- **Price grid 0.75–1.25×:** widening it didn't help; 0.90–1.20× halved the RL loss in a 2-seed test (old constant-capacity world). Not widened: 1.25× already lies 13% above the hotel's highest price, and the cap binds on about a quarter of days.
- **Seed ensemble:** needs an odd number of seeds (the median of two actions always takes the lower price).
- **Season seeds already used:** validation 100000+, calibration 200000+, a schedule check 300000–300399, the CRN decision 400000–400199, test 500000+, diagnostics 700000–700199 and 900000–900029.

### Code review (23 Oct, read-only): no bug changes a result

- **Env invariants** (2 worlds × 30 seasons): requests are monotone in price on every day, `dry_run` equals the step's reward exactly, and a seed reproduces its season.
- **Also verified:** `DoubleDQN.train` against SB3 2.8.0; a deterministic `predict` draws no random numbers; `VecNormalize` is valid for both algorithms; terminal handling, seed ranges, the ML guardrails, the calibration loop, the SEs and the profiles; the Orthodox Easter dates (20 Apr 2025, 12 Apr 2026); mean rooms per booking is 1.002, so `lambda` counting rooms while the env draws bookings makes no difference. `robustness.py` rebuilds every 22 Oct result with the current code.
- **Cosmetic only:** `stay_mean`'s docstring says an unknown night gives NaN, but `groupby().mean()` skips NaN (it never happens: the nightly BAR is interpolated); in `holiday_features`, 25 March lies outside the summer window.
- **Anonymity (rechecked 27 Oct):** no word of the data's `Property` / `Property Name` values appears in any tracked file, the notebook's outputs or git history. The only hit is the export's "Summary" footer row.

### Price recommendation tool (`pricing_tool.ipynb`)

- On a whole simulated season it rebuilt every observation and action of the rollout for all four ensembles (195 days, no mismatch), and its prices by rooms left match `policy_map.csv` within about 0.02×.
- The runs agree when rooms are scarce (all at 1.25×) and diverge when many are left: on 14 Jul with 10 rooms left both PPO ensembles say 176 € and both Double DQN ensembles 192 €, and the xgb Double DQN seeds range from 160 to 200 €.

### Elasticity

Keep 1.0 as the main scenario. Per the hotel, its main competitor is usually 5–15% cheaper for this room type (sometimes more expensive), so ε well below 1 would imply implausible premiums; report about 0.8–1.5 as sensitivity, with the agents trained at each ε (see Agents trained at each elasticity).

Cross-file contracts: `bar_rates.py` imports `load_raw`, `file_hash`, `SEASONS`, `SUMMER`, `XLSX_PATH` and `BAR_DIR` from `ml_demand_model.py`, which in turn reads `bar_exports/bar_daily.csv` / `bar_bookings.csv` (`load_bar` fails loudly if they are missing, don't cover every booking, or `bar_inputs.sha256` no longer matches `BOOKINGS.xlsx` and the rate CSVs: re-run `bar_rates.py` after changing them or the load filters). `rl_agent.py` imports `PRICE_MULTIPLIERS` and `SIM_DATA_PATH` from the env, and `file_hash` from `ml_demand_model.py` inside `append_elasticity_log`, so worker processes don't load the ML stack. The env depends on the exact `sim_data.pkl` schema produced by `build_sim_data` (per season: `capacity`, `online_avail`, `dispersion_k`, `dates`, `lambda_true`, `lambda_forecast`, `hotel_price`, `p_ref`, calendar arrays, `profiles`, `actual`; top level: `seasons`, `models`). `actual` also holds the calibration targets `requests_var` and `sold_out_nights`. After changing that schema, `BOOKINGS.xlsx` or `hotel_pms_data.csv`, you must re-run `ml_demand_model.py`. `ml_demand_model.py` imports `HotelEnv` inside `simulate_hotel` to calibrate, and calibrates on the env's `info["recorded"]` and `info["room_nights"]`. Model keys `"rf"` / `"xgb"` are shared across the ML, env and RL files. `robustness.py` imports its evaluation helpers and constants from `rl_agent.py` (one implementation) and reads the elasticity log's `sim_data`, `results.csv` and the comparison CSV, so a change to `summarise`, `seed_se` or the result names makes its reproduction check fail. `ml_reference.py` imports the loaders and scoring from `ml_demand_model.py` and repeats `main`'s load filters; if those change, its `sim_data` guard fails. `pricing_tool.ipynb` imports from `rl_agent.py`, `robustness.py` and the env.

## Future and proposed work

**Status (29 Oct 2026):**
- The ML and RL code is frozen for the professors' review, and the 22 Oct run is final. The 29 Oct clean rerun (BAR, ML, then RL at ε 0.6–1.5 with `robustness.py` for each, `ml_reference.py` and the notebook) reproduced it exactly. Only printing changes are planned before the review; everything else below comes after it.
- The 27 Oct evaluation additions (`robustness.py`, `ml_reference.py`, `pricing_tool.ipynb`, the env's `demand_scale`) change no result.
- Send the repo through git (a link or `git archive`), never as a zipped folder: a zip would include the private data and `sim_data.pkl`.

Nothing here is implemented. Importance: **P1** = needed before the final thesis run (correctness, realism or protocol); **P2** = likely to raise agent revenue or make the results more defensible; **P3** = if time allows; **Later** = beyond the thesis, described as a limitation. Effort: 1 (quick) … 5 (long). Test every item in a scratch copy, choose only on validation, tuning or holdout seasons (never test), and apply after approval. Items that change `online_avail` or the world need an ML + RL rerun, so do them together.

**Order:**
1. World calibration and tuning, only if the thesis needs them. Each needs a full rerun; afterwards rerun `robustness.py`, `ml_reference.py` and the notebook.
2. Docs.

**World and data** (P1 unless marked):
- **World calibration.** Choose the world by realism (recorded requests, their variance, room-nights, sold-out nights), never by RL revenue. Until it is settled, report the ε = 1 level gain as the range +0.9 to +3.1% with the reason (see Where the level gain comes from).
  - **Per-date calibration (P2, 3–4):** the remaining realism gap of the applied one-factor world is about 42–43 sold-out nights against 32 real, 17 of them on nights that had rooms left. The prototype matches recorded requests per date and their variance. Its open issues are 46–50 sold-out nights (against 34 real before the PMS fix), room-nights 2.5–3.5% short, and a result that depends on the cap (+2.1 to +3.1%). Next: build it on the applied joint solver (uplift per date plus `k`), aiming at recorded variance and sold-out nights, then check whether the room-nights close.
  - **Censored likelihood (P3, 4):** an untested alternative. Scaling only censored nights doesn't work.
- **Env invariant checks (P3, 1):** checked in memory on 23 Oct, and all held (see Code review). A small script would make the check repeatable.

**RL code:**
- **One process pool (P2, 1):** all ML × RL × seed jobs in one pool, about 25–35% less wall time.
- **Diagnostics (P3, 1):** log in `ValidationCheckpoint` (TensorBoard) the validation gain, a histogram of the chosen multipliers, mean price / occupancy / lost requests and the price–forecast correlation, plus PPO's KL, entropy and explained variance.

**Agent revenue** (the main aim): the per-date schedule is worth about +0.65 points over best fixed, and the final run reaches it (PPO +0.61 / +0.60, the ensembles +0.63–0.66). It is not a ceiling, since a policy that sees the rooms left can beat it. Still, tuning can add little, and only through capacity pricing that a rooms-left rule already gets mostly right (see the capacity-rule finding).
- **Optuna tuning (P2, 3):** on the final world and reward.
  - **Gate first:** at the same final training budget, pretuned settings vs a 16-trial study per algorithm (150k × 3 seeds, rf world), decided on holdout seasons 410000–410199 (400000–400199 decided the CRN reward).
  - **If Optuna wins, run the full study:**
    - objective: mean % vs best fixed over the rf and xgb worlds;
    - 24 trials per algorithm, with the pretuned config enqueued first;
    - 3 tuning seeds (1001–1003, never the final seeds);
    - tuning seasons 300000–300099 (the calibration uses 200000+);
    - pruning on the validation curve, and the top 3 trials rescored with fresh seeds;
    - 300k steps per trial, about 4–5 hours.
  - Until it is adopted, install `optuna` in scratch only.
- **Narrower action grid (P3):** 0.90–1.20× halved the RL loss in a 2-seed test. As the main grid it would change the level effect and comparability, so the price-range check covers it.

**Benchmarks and robustness** (the thesis story):
- **More robustness worlds (P3, 1):** `robustness.py` covers the other ML world, ε 0.8 / 1.2 and demand ×0.85 / ×1.15. Forecast-shaped demand and the 2025 offline calendar would each be one more `WORLDS` entry. At ε 0.8 the agents keep their ε = 1 level and trail a re-chosen 1.25× by about 1.7%; training across ε (domain randomisation) would address that.
- **Price-range check, RL part (P3, 2):** one PPO run with multipliers limited to the hotel's range (0.95–1.10×); it needs a multipliers option in the env. The benchmarks already show no day-by-day value inside that range, so expect none.
- **Contextual-bandit baseline (P3, 2–3):** the MDP is close to one. The rooms-left rule (an official baseline in `robustness.py`) is a simple bandit and already answers the main question: the RL roughly equals the rule, and its lead is not significant after Holm. A bandit learned on the full observation would add little.
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
