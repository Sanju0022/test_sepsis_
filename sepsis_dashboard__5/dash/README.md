# Multi-Agent Healthcare Decision Support — Streamlit dashboard

Group Project 3 (PDF Task 5). Eight tabs, each with its own local filters — there is no global sidebar filter.

**Light and dark themes.** The app follows the viewer's system setting; anyone can switch in the menu (⋮) → Settings → Theme.
Both themes are defined in `.streamlit/config.toml`, and every chart uses one colour palette that was checked for colour-blind
separation, lightness and contrast on both the light and the dark background. Charts have transparent backgrounds and take their
fonts and grid colours from the active theme.

**Live panels.** *What if?* (POMDP tab) re-solves the POMDP for any cost of unnecessary antibiotics and of a missed case and
replays it on all 2,000 patients — with the default weights it reproduces the Step 5 results exactly. *Posterior beliefs* (Bandit tab)
shows the Beta posteriors that Thompson sampling uses. *Negotiation outcomes* (Collaboration tab) scores each agent when the team
disagreed. *Resource utilisation* (KPI tab) shows beds, nurses and the lab queue.

**Live ICU monitor.** A simulated 22-bed ICU admits synthetic patients one after another (one simulated hour = one row of a
patient's record). Every refresh the nurse (NEWS2 ≥ 5), the doctor (SIRS ≥ 2) and the POMDP agent (belief above the Step 5 treat
threshold) re-assess every bed. The true sepsis label stays hidden from the agents; a toggle reveals it for the demo. On Streamlit
Cloud the tab replays 600 synthetic patients inside the app; on your computer it can be fed live from Docker (see below).

**Interactive extras.** Play buttons replay the POMDP belief and the bandit learning curves; gauges compare a probability with its
treat threshold; the Bayesian tab redraws the network, the waterfall and the gauge as you tick signs.

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501.

## Deploy on Streamlit Community Cloud

Put these at the **root** of the GitHub repository (the same folder as `app.py`), then deploy with `app.py` as the main file:

```
app.py                  the dashboard
pomdp_core.py           the Step 5 POMDP, used by the what-if panel and the live monitor (must sit next to app.py)
icu_sim.py              the simulated ICU used by the live monitor and the Docker feeder (must sit next to app.py)
requirements.txt        packages Streamlit Cloud must install (streamlit>=1.55, plotly, pandas, numpy)
.streamlit/config.toml  theme
data/                   the CSV files produced by Steps 3–7 plus live_replay_sample.csv.gz (600 synthetic patients)
```

`feeder.py`, `Dockerfile`, `docker-compose.yml` and `.dockerignore` are only for the Docker demo; Streamlit Cloud ignores them.
Do **not** push `data_synth/synthetic_icu_20000_hourly.csv.gz` to GitHub — the app does not need it online.

If the app was deployed before `requirements.txt` was added, push the file and then use **Manage app → Reboot app** so the packages are installed.

## Live demo with Docker Desktop (on your own computer)

Two containers share one small SQLite database on a Docker volume:

```
feeder     reads the 20,000-patient synthetic file and writes one simulated ICU hour every TICK_SECONDS
dashboard  this Streamlit app; the Live ICU monitor tab reads the newest hour from the database
```

1. Install Docker Desktop (https://www.docker.com/products/docker-desktop/) and start it.
2. Unzip this folder and copy `synthetic_icu_20000_hourly.csv.gz` into its `data_synth/` folder.
   (Without it the feeder replays the 600-patient sample in `data/`.)
3. Open a terminal in the folder and run

   ```bash
   docker compose up --build
   ```

   The first run downloads `python:3.12-slim` and installs the packages, which takes a few minutes.
4. Open http://localhost:8501 and choose the **Live ICU monitor** tab. A green bar says *Connected to the Docker live feed*.
5. Stop with `Ctrl+C`, then `docker compose down` (add `-v` to also delete the database volume).

Settings are in `docker-compose.yml` under `feeder → environment`:

| Variable | Default | Meaning |
|---|---|---|
| TICK_SECONDS | 1.0 | real seconds per simulated ICU hour (0.5 = faster demo) |
| BEDS | 22 | number of ICU beds |
| START_HOUR | 48 | the feeder pre-loads this many hours so the ICU is already busy when you open the tab |
| LOOP | 1 | start again from the beginning when the file is used up |

After changing a setting run `docker compose up -d` again. `docker compose logs -f feeder` shows the feeder's progress.

## Tabs and where their data comes from

| Tab | Data file | Local filters |
|---|---|---|
| Patient Overview | sepsis_multiagent_dataset.csv (2,000 patients) | sepsis-only toggle, patient, hour range, which vital signs |
| POMDP Belief Tracker | pomdp_belief_traces.csv (200 patients), pomdp_policy.csv, pomdp_parameters.csv | patient group, patient, hours left for the threshold |
| Bandit Decision Explorer | bandit_treatment.csv, bandit_policy.csv, bandit_arm_shares.csv | patients seen, algorithms, bandit type, cost of unnecessary antibiotics |
| Bayesian Network | bayes_network.csv, bayes_fusion.csv | prior slider, the four SIRS signs, which measure to compare |
| MLE Parameter Insights | mle_parameters.csv | estimate group, parameters |
| Agent Collaboration Log | collaboration_log.csv (200 patients) | disagreement only, suggested action, patient, hour |
| Outcome & KPI Analysis | sepsis_multiagent_dataset.csv, pomdp_evaluation.csv, pomdp_sensitivity.csv | hospital set, age group, patient type |
| Live ICU monitor | Docker feed (synthetic_icu_20000_hourly.csv.gz) or live_replay_sample.csv.gz; mle_parameters.csv for the POMDP | refresh rate, hours left for the threshold, reveal true label, replay speed and pause (replay only), bed to follow |

## Notes on the data behind the charts

- Belief traces cover the first 200 of the 2,000 patients; the collaboration log covers the first 200 patients that have an hour with complete vitals and labs (6,175 hours). Both tabs say so on screen.
- The collaboration log compares the pooled opinion with the exact Step 5 treat threshold for the hours left (Step 5 writes `pomdp_thresholds.csv` for 1–72 hours left).
- The POMDP policy file holds three points in the stay (72, 24 and 1 hours left), so the threshold slider offers exactly those.
- Sepsis records in PhysioNet stop about 3 hours after onset, so sepsis stays look shorter and cheaper in the cost chart.
- Vital-sign panels are drawn one per sign rather than on one axis, because the scales differ.
- The live monitor uses synthetic patients with a 20% sepsis rate chosen for the demo; the real PhysioNet 2019 data it was learned from has 7.27% (2,932 of 40,336 patients).
- With the default threshold (24 hours left) POMDP alerts are rare; choose fewer hours left to see the policy act earlier.
- Where a treat threshold sits far above the plotted probabilities, it is written on the chart instead of being drawn, so the curve stays readable.
