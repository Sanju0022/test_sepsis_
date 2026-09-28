"""
POMDP core used by the dashboard's live what-if panel.
It is the same model as Step 5 of the Colab notebook (Step5_POMDP.ipynb), so with the default
weights it reproduces pomdp_evaluation.csv and pomdp_sensitivity.csv exactly.
"""
import numpy as np
import pandas as pd

HORIZON = 72                       # hours of monitoring (modelling choice, Step 5)
W_SAT = 0.10                       # satisfaction weight for a lab wait (modelling choice, Step 5)
P_DEATH_BASE, OR_DELAY = 0.165, 1.04   # HCUP SB306 mortality; Seymour 2017 odds ratio per hour
ICU_HOUR = 4300 / 24               # SCCM ICU cost per hour
ACTIONS = ["monitor", "lab", "treat"]
G = np.linspace(0, 1, 1001)        # belief grid
POLICY_NAMES = {"pomdp": "POMDP policy", "never_treat": "Never treat",
                "news2_rule": "Treat when NEWS2 ≥ 5", "sirs_rule": "Treat when SIRS ≥ 2"}


def model_params(mle: pd.DataFrame, mean_lab_wait: float) -> dict:
    def get(item, parameter, group):
        r = mle[(mle["item"] == item) & (mle["parameter"] == parameter) & (mle["group"] == group)]
        assert len(r) == 1, (item, parameter, group)
        return float(r["mle"].iloc[0])

    sat_a = get("satisfaction_curve", "intercept_a", "logit")
    sat_b = get("satisfaction_curve", "slope_b_per_min", "logit")
    logistic = lambda z: 1 / (1 + np.exp(-z))
    odds = P_DEATH_BASE / (1 - P_DEATH_BASE) * OR_DELAY
    return {
        "P_TRANS": get("progression", "p_transition_0to1_per_hour", "all"),
        "P_INIT": get("progression", "p_initial_state1", "all"),
        "TESTS": {"news2": (get("test_accuracy", "sensitivity", "NEWS2 >= 5"), get("test_accuracy", "specificity", "NEWS2 >= 5")),
                  "sirs": (get("test_accuracy", "sensitivity", "SIRS >= 2"), get("test_accuracy", "specificity", "SIRS >= 2"))},
        "LAB_PENALTY": W_SAT * (logistic(sat_a) - logistic(sat_a + sat_b * mean_lab_wait)),
        "C_DELAY": odds / (1 + odds) - P_DEATH_BASE,
    }


def predict(b, p):
    return b + (1 - b) * p["P_TRANS"]


def update(b_pred, test, obs, p):
    if obs is None or (isinstance(obs, float) and np.isnan(obs)):
        return b_pred
    sens, spec = p["TESTS"][test]
    like1 = sens if obs == 1 else 1 - sens
    like0 = (1 - spec) if obs == 1 else spec
    num = b_pred * like1
    return num / (num + (1 - b_pred) * like0)


def _q_observe(v_next, test, cost, p):
    bp = predict(G, p)
    sens, spec = p["TESTS"][test]
    p_pos = bp * sens + (1 - bp) * (1 - spec)
    b_pos = np.divide(bp * sens, p_pos, out=np.zeros_like(bp), where=p_pos > 0)
    p_neg = 1 - p_pos
    b_neg = np.divide(bp * (1 - sens), p_neg, out=np.zeros_like(bp), where=p_neg > 0)
    return -cost + p_pos * np.interp(b_pos, G, v_next) + p_neg * np.interp(b_neg, G, v_next)


def solve(p, w_abx, w_miss):
    """Backward induction over the belief grid. Returns the best action index per (hour, belief)."""
    V = np.empty((HORIZON + 1, len(G)))
    A = np.zeros((HORIZON, len(G)), dtype=int)
    V[HORIZON] = -w_miss * G
    for t in range(HORIZON - 1, -1, -1):
        q = np.stack([_q_observe(V[t + 1], "news2", p["C_DELAY"] * G, p),
                      _q_observe(V[t + 1], "sirs", p["LAB_PENALTY"] + p["C_DELAY"] * G, p),
                      G - (1 - G) * w_abx])
        A[t] = q.argmax(0)
        V[t] = q.max(0)
    return A


def thresholds(A):
    """Lowest belief at which the policy treats, for every number of hours left (72 ... 1)."""
    out = {}
    for t in range(HORIZON):
        treat = G[A[t] == 2]
        out[HORIZON - t] = float(treat.min()) if len(treat) else np.nan
    return out


def observations(df: pd.DataFrame):
    """Per-patient arrays the policies read: NEWS2 >= 5, SIRS >= 2 (blank when an input is missing), ICU hour."""
    d = df.sort_values(["patient_id", "ICULOS"])
    sirs_pos = (((d["Temp"] > 38) | (d["Temp"] < 36)).astype(int) + (d["HR"] > 90).astype(int)
                + (d["Resp"] > 20).astype(int) + ((d["WBC"] > 12) | (d["WBC"] < 4)).astype(int))
    sirs_ok = ~d[["Temp", "HR", "Resp", "WBC"]].isna().any(axis=1)
    news2 = np.where(d["news2_score"].isna(), np.nan, (d["news2_score"] >= 5).astype(float))
    sirs = np.where(sirs_ok, (sirs_pos >= 2).astype(float), np.nan)
    d = d.assign(obs_news2=news2, obs_sirs=sirs)
    pats = []
    for pid, g in d.groupby("patient_id", sort=False):
        pats.append((pid, int(g["sepsis_patient"].iloc[0]), float(g["sepsis_onset_hour"].iloc[0]),
                     g["obs_news2"].to_numpy(), g["obs_sirs"].to_numpy(), g["ICULOS"].to_numpy()))
    return pats


def run(pats, p, strategy, A=None):
    rows = []
    for pid, sep, onset, news2, sirs, icu_hour in pats:
        L = len(icu_hour); b = p["P_INIT"]
        treated_at = np.nan; treated_row = np.nan; labs = 0
        for i in range(L):
            if strategy == "pomdp":
                t = max(0, HORIZON - (L - i))
                a = ACTIONS[A[t][np.abs(G - b).argmin()]]
            elif strategy == "never_treat":
                a = "monitor"
            elif strategy == "news2_rule":
                a = "treat" if news2[i] == 1 else "monitor"
            else:
                a = "treat" if sirs[i] == 1 else "monitor"
            if a == "treat":
                treated_at = float(icu_hour[i]); treated_row = i + 1
                break
            labs += (a == "lab")
            b = update(predict(b, p), "sirs" if a == "lab" else "news2", sirs[i] if a == "lab" else news2[i], p)
        rows.append((pid, sep, onset, L, treated_at, treated_row, labs, b))
    return pd.DataFrame(rows, columns=["patient_id", "sepsis", "onset", "stay_hours", "treated_at", "treated_row", "labs", "final_belief"])


def utility(r, p, w_abx, w_miss):
    u = np.where(r["treated_at"].notna(),
                 np.where(r["sepsis"] == 1, 1 - p["C_DELAY"] * np.clip(r["treated_at"] - r["onset"], 0, None), -w_abx),
                 np.where(r["sepsis"] == 1, -w_miss, 0.0))
    return u - p["LAB_PENALTY"] * r["labs"]


def compare(results: dict, p, w_abx, w_miss) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        sep, non = r[r["sepsis"] == 1], r[r["sepsis"] == 0]
        treated_sep = sep["treated_at"].notna()
        rows.append((name, POLICY_NAMES.get(name, name), float(np.mean(utility(r, p, w_abx, w_miss))),
                     treated_sep.mean() * 100,
                     ((sep["treated_at"] < sep["onset"]) & treated_sep).mean() * 100,
                     non["treated_at"].notna().mean() * 100,
                     float(r["labs"].mean()),
                     float((r["treated_row"].fillna(r["stay_hours"]) * ICU_HOUR).mean())))
    return pd.DataFrame(rows, columns=["policy", "label", "mean utility", "sepsis treated %", "treated before onset %",
                                       "non-sepsis treated %", "mean labs", "mean ICU cost $"])
