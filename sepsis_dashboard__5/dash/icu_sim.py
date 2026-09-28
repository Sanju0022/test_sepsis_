"""
A simulated 22-bed ICU that admits synthetic patients one after another.
Shared by the Docker feeder (feeder.py) and by the dashboard's built-in replay, so both show the same thing.

One simulated hour = one row of a patient's record. When a patient's record ends the bed is cleaned and
stays empty for a random turnover time chosen so that average occupancy is about 68% (Wunsch et al., 2013).
"""
import numpy as np
import pandas as pd

LIVE_COLUMNS = ["patient_id", "ICULOS", "Age", "Gender", "HR", "O2Sat", "Temp", "SBP", "MAP", "Resp", "WBC",
                "HR_measured", "Temp_measured", "Resp_measured", "news2_score", "panels_ordered",
                "abx_started", "on_antibiotics", "SepsisLabel", "sepsis_patient", "sepsis_onset_hour"]
TARGET_OCCUPANCY = 0.682


def load_patients(path, max_patients=None):
    d = pd.read_csv(path, usecols=LIVE_COLUMNS)
    d = d.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
    if max_patients:
        keep = d["patient_id"].drop_duplicates().iloc[:max_patients]
        d = d[d["patient_id"].isin(keep)].reset_index(drop=True)
    return d


def build_schedule(d, beds=22, seed=7):
    """Admission hour and bed for every patient, in file order."""
    rng = np.random.default_rng(seed)
    stays = d.groupby("patient_id", sort=False).size()
    mean_gap = stays.mean() * (1 / TARGET_OCCUPANCY - 1)
    free_at = np.zeros(beds)                      # hour at which each bed can take the next patient
    rows = []
    for pid, n in stays.items():
        b = int(np.argmin(free_at))
        start = int(free_at[b])
        rows.append((pid, b + 1, start, start + int(n) - 1))
        free_at[b] = start + n + rng.geometric(1 / max(mean_gap, 1.0))
    return pd.DataFrame(rows, columns=["patient_id", "bed", "admit_hour", "last_hour"])


class Replay:
    """Everything that is visible in the ICU at a given simulated hour."""

    def __init__(self, d, beds=22, seed=7):
        self.d = d
        self.beds = beds
        self.sched = build_schedule(d, beds, seed)
        self.end_hour = int(self.sched["last_hour"].max())
        starts = d.groupby("patient_id", sort=False).cumcount()
        self.d = d.assign(offset=starts.to_numpy())
        s = self.sched.set_index("patient_id")
        self.d["bed"] = self.d["patient_id"].map(s["bed"])
        self.d["sim_hour"] = self.d["patient_id"].map(s["admit_hour"]) + self.d["offset"]

    def rows_at(self, h):
        """Rows released in simulated hour h (one per occupied bed)."""
        return self.d[self.d["sim_hour"] == h]

    def history(self, h):
        """For patients in a bed at hour h: all their rows released so far."""
        cur = self.sched[(self.sched["admit_hour"] <= h) & (self.sched["last_hour"] >= h)]["patient_id"]
        return self.d[self.d["patient_id"].isin(cur) & (self.d["sim_hour"] <= h)]

    def events(self, h, lookback=24):
        """Admissions, discharges, lab rounds, antibiotics and (for the demo) the true sepsis label, last `lookback` hours."""
        lo = max(0, h - lookback)
        ev = []
        s = self.sched
        for _, r in s[(s["admit_hour"] >= lo) & (s["admit_hour"] <= h)].iterrows():
            ev.append((int(r["admit_hour"]), int(r["bed"]), r["patient_id"], "admitted"))
        for _, r in s[(s["last_hour"] + 1 >= lo) & (s["last_hour"] + 1 <= h)].iterrows():
            ev.append((int(r["last_hour"]) + 1, int(r["bed"]), r["patient_id"], "left the ICU"))
        w = self.d[(self.d["sim_hour"] >= lo) & (self.d["sim_hour"] <= h)]
        for _, r in w[w["abx_started"] == 1].iterrows():
            ev.append((int(r["sim_hour"]), int(r["bed"]), r["patient_id"], "antibiotics started"))
        for _, r in w[w["panels_ordered"] > 0].iterrows():
            ev.append((int(r["sim_hour"]), int(r["bed"]), r["patient_id"], f"{int(r['panels_ordered'])} lab panel(s) drawn"))
        first_lab = w[w["SepsisLabel"] == 1].groupby("patient_id").head(1)
        prev = self.d[(self.d["sim_hour"] < lo) & (self.d["SepsisLabel"] == 1)]["patient_id"].unique()
        for _, r in first_lab[~first_lab["patient_id"].isin(prev)].iterrows():
            ev.append((int(r["sim_hour"]), int(r["bed"]), r["patient_id"], "true sepsis label switched on (hidden from agents)"))
        return pd.DataFrame(ev, columns=["sim_hour", "bed", "patient_id", "event"]).sort_values(["sim_hour", "bed"], ascending=[False, True])
