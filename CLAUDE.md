# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

IntelliRate is a thesis project (University of the Peloponnese, ECE) that dynamically prices hotel rooms. It reconstructs the hotel's Best Available Rate (BAR) from real bookings, fits ML demand models to them, and then trains RL agents (PPO and DQN; a Double DQN is planned) in a simulator built from those models. It compares each ML model × RL algorithm combination with the hotel's actual prices and with the best fixed price.

## Environment and commands

This is Windows with Python 3.12. Create the conda env from [conda_env/intellirate-cpu.yml](conda_env/intellirate-cpu.yml) (the `prefix:` line is machine-specific), then activate it:

```
conda env create -f conda_env/intellirate-cpu.yml
conda activate intellirate-cpu
```

Run the pipeline in order, from the repo root. All paths are relative to the current working directory.

```
python bar_rates.py            # BOOKINGS.xlsx + rate_rules.csv + rate_occupancy.csv -> bar_daily.csv, bar_bookings.csv, bar_validation.txt, bar_daily.png
python ml_demand_model.py      # BOOKINGS.xlsx + bar_*.csv -> demand_models.pkl, sim_data.pkl, ml_comparison.csv, demand_fit.png
python rl_agent.py             # sim_data.pkl -> results_<season>_<model>_e<elasticity>/, comparison_<tag>.csv/.png
```

Useful `rl_agent.py` flags for quick or partial runs:
- `--models rf` (or `xgb`) runs only one pipeline.
- `--algos ppo` (or `dqn`) trains only one RL algorithm (default: all keys in `ALGOS`).
- `--timesteps 20000 --seeds 42 --no-sensitivity` gives a fast smoke run (the default is 300k steps × 3 seeds for each algorithm, plus a sensitivity sweep). Note that a smoke run overwrites the `results_*` / `comparison_*` outputs of a full run with the same season, elasticity and world.
- `--season 2025` and `--elasticity 1.2` change the scenario.
- `--common-world rf` makes every pipeline use the same "true" demand model while each agent keeps its own forecast. The output directories and tags get a `_world-<m>` suffix.
- `--jobs 3` (the default) trains the seeds of each ML × RL combination in parallel processes, each with one torch thread. Parallelism is per seed, so at most `len(--seeds)` processes run. `--jobs 1` runs sequentially and gives identical results. A GPU doesn't help here: the networks are tiny and the bottleneck is Python overhead per step. Running all ML × RL × seed jobs in one pool (more seeds on more cores) is planned.

The repo has no tests, linter or build step. `.gitignore` uses patterns: the private data in any format (`BOOKINGS.*`) and every generated artifact (`*.pkl`, `*.zip`, `bar_*` outputs, `ml_comparison.csv`, `demand_fit.png`, `results_*/`, `comparison_*`, `logs/`) are ignored, so runs for any season, elasticity or world stay out of the repo. The two rate config files (`rate_rules.csv`, `rate_occupancy.csv`) are tracked. The pipeline runs without warnings (the Excel export's openpyxl style warning is silenced in `load_raw`); keep it that way rather than running with `-W ignore`.

Two project rules: only the **summer season** is studied (arrivals 1 April – 31 October, `SUMMER` in `ml_demand_model.py`), and the **hotel stays anonymous** — never write its name (it appears in the data's property, website and member-programme columns) into code, configs, outputs or docs.

## Architecture

There are four files, forming one pipeline: **BAR reconstruction → ML demand model → simulator → RL agent**.

0. **[bar_rates.py](bar_rates.py)** reconstructs the hotel's double-occupancy Best Available Rate (BAR) from the recorded prices. The rate system builds a price as `occupancy BAR × stacked discount factors`; the script unwinds that per booking:
   - [rate_rules.csv](rate_rules.csv): one row per discount/commission (NR, Genius, Last Minute Deal, Expedia promos and 18% commission, website direct offer / member programme), with detection column + regex, season, booking-date window, factor, optional `round_to` (rates rounded after discounts) and `confirmed`/`inferred` status. A `@infer` row (`genius_unflagged`) is decided per booking against the price ladder.
   - [rate_occupancy.csv](rate_occupancy.csv): the occupancy grid. 2025: no single rate, triple at a fixed price (no BAR information). 2026: per-level single and triple prices. Children aged ≥ 4 count as adults; only Booking.com records ages, so unknown ages are inferred from the price.
   - Double price levels count only when seen on ≥ 2 channel groups. Multi-night stays are tested against, and split by, the one-night-stay BAR (never against themselves). The daily BAR is the weighted median per stay night (cancelled bookings included).
   - Outputs `bar_daily.csv` (BAR per night, `source` observed / interpolated / filled), `bar_bookings.csv` (per booking: rules, occupancy, inference method, `bar`, `bar_ref`), and `bar_validation.txt` / `bar_daily.png`.

1. **[ml_demand_model.py](ml_demand_model.py)** loads `BOOKINGS.xlsx`, drops PII columns, and keeps room type `STD`, seasons 2025/2026, summer arrivals and arrivals on or before the export date. It then builds a daily table of booking *requests* (cancelled bookings included), the hotel price (the daily BAR from `bar_daily.csv`, not smoothed), and calendar/holiday features. The holidays are Greek: Orthodox Easter, Whit Monday and national days. It trains RF and XGBoost (Poisson objective) with `GroupKFold` blocked by ISO week and compares them against a season-month mean baseline. For each model it stores:
   - `lambda_true`: the full-data fit, rescaled so that each season's total matches the real total. It drives the simulated "world".
   - `lambda_forecast`: out-of-fold predictions. The agent sees these as its forecast.
   - `dispersion_k`: the negative-binomial overdispersion estimated from the OOF residuals. `None` means pure Poisson.
   - Per-month pools of real booking profiles (rooms, nights, Total, orig_price, cancelled). `orig_price` is the booking's own BAR (`bar_ref`), so scaling `Total` to another BAR keeps the guest's occupancy, channel and plan factors. A month with fewer than 20 profiles falls back to the season pool.

   Everything goes into `sim_data.pkl`, keyed by `seasons[year]`.

2. **[hotel_season_env.py](hotel_season_env.py)**: `HotelEnv` is a Gymnasium env where one step is one arrival date and one episode is one season.
   - The action is one of 11 BAR multipliers (0.75–1.25×) of the season reference price `p_ref`, the median daily BAR.
   - Price response: `g(p) = exp(-elasticity·(p/p_ref − 1))`. `lambda_base = lambda_true / g(hotel_price)` removes the hotel's own price effect, so demand can be re-priced.
   - All randomness for the season (gamma shocks, uniforms for the counts and the profile picks) is drawn in `reset()`, before any price is set. Counts use inverse-CDF Poisson, so a given seed is monotone in price. This gives paired, common-random-number comparisons between policies. Keep this property when you change the env.
   - Each request samples a real booking profile. Cancelled profiles give no revenue, and profiles that don't fit under `capacity` (17 rooms) over their nights are lost. Revenue is the profile's real `Total` scaled by `price / orig_price`.
   - The observation has 9 dims in [-1, 1]: forecast, committed rooms today and tomorrow, cyclic month and day of week, holiday flag, and season progress.
   - Baselines use `step_price(price)` to pass an arbitrary price, for example the hotel's BAR.

3. **[rl_agent.py](rl_agent.py)** runs one pipeline per ML model, and each pipeline trains every RL algorithm:
   - RL algorithms live in the `ALGOS` registry (`key -> (SB3 class, params, display name)`). `train_agent` is generic: SB3, CPU, `VecNormalize` on reward only, which is also valid for DQN because off-policy SB3 normalises replay-buffer rewards at sample time. To add an algorithm, add a registry entry. The planned Double DQN is meant to be a `DQN` subclass that overrides `train()`, reusing `DQN_PARAMS` unchanged. Every DQN hyperparameter is set explicitly so that the TD target is the only difference between the two.
   - It validates the simulator: the hotel's prices in the sim are compared with the real season.
   - It finds the best fixed price once per ML model, then trains each algorithm for each seed.
   - It evaluates every policy on the same `N_EVAL` seasons, seeded from `EVAL_SEED0 = 100000` so they are never seen in training.
   - It reports the gain over the hotel split into a price-*level* effect (best fixed vs hotel, driven by the elasticity assumption) and a *day-by-day* effect per algorithm (RL vs best fixed). A negative day-by-day number means the agent hasn't converged.
   - It also sweeps elasticity for the best fixed price, with no training.
   - Per ML model it writes `results_*/` containing `results.csv`, `summary.txt`, `learning_curve_<algo>.png`, `price_path.png` and `<algo>_seed<n>.zip`. `comparison_<tag>.csv/.png` gets one row per ML × RL combination, with `rl_*` columns.

## Booking data and rates

`BOOKINGS.xlsx` is a private channel-manager export. Every row is in EUR, and `Total` is the room charge only (`Total == Stay`). `load_raw` keeps the recorded rate `price = Total / Room-Nights`; all rate normalisation lives in the BAR stage (`bar_rates.py` + the two rate CSVs). Change rates there, not in the ML/RL files. Rate structure (details and status per rule in the CSVs):
- Channel: `Source`, except website bookings (`Application == "WEBHOTELIER"`, `Source` empty or Google Free Booking Links), which form the "Direct" channel. Expedia Collect `Total` is net of 18% commission; all other channels are gross (Booking.com commission is not removed).
- Rate parity across channels. NR is −10% everywhere. Booking.com Genius (−10%, flagged "Genius Booker" in `Channel Notes for Hotelier`), the Booking.com Last Minute Deal (−5%, in `Room Type Name`), the Expedia promos in `Rate` (Blue Member −10% in 2025, Member / Mobile −7% in 2026) and every other promo stack by multiplication. 2025 website: permanent −12% direct offer. 2026 website: member programme (`SmartGuest`) −10%, or −14.5% for bookings made from 1 Sep 2026 (those rates are rounded to the euro).
- Occupancy pricing: see `rate_occupancy.csv`. BAR always means the double-occupancy BAR.
- Flexible bookings cancel at a rate of about 36%, NR bookings at about 2%. `Cancel Fee` is always 0. In the simulator a cancelled profile earns nothing and never holds capacity.
- The 2026 season is cut off at the export date (27 Sep), while 2025 runs to 31 Oct.
- The hotel's online booking systems opened late in 2025, on about 1 Apr. About a third of the 2025 demand is therefore unrecorded, including about three quarters of April. **Decision: 2025 is kept as recorded, with no correction.** Treat 2026 as the main RL season, and describe 2025 as a limitation rather than fixing it.

Cross-file contracts: `bar_rates.py` imports `load_raw`, `SEASONS` and `SUMMER` from `ml_demand_model.py`, which in turn reads `bar_daily.csv` / `bar_bookings.csv` (`load_bar` fails loudly if they are missing or don't cover every booking: re-run `bar_rates.py` after changing `BOOKINGS.xlsx`, the rate CSVs or the load filters). `rl_agent.py` imports `PRICE_MULTIPLIERS` from the env. The env depends on the exact `sim_data.pkl` schema produced by `build_sim_data`, so after changing that schema you must re-run `ml_demand_model.py`. Model keys `"rf"` / `"xgb"` are shared across the ML, env and RL files.
