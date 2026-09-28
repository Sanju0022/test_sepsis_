"""
Live data feeder for the Docker demo.

Releases one simulated ICU hour every TICK_SECONDS into a SQLite database that the dashboard reads.
Environment variables (all optional):
  LIVE_DB       path of the SQLite file shared with the dashboard   (default /live/icu_live.db)
  SYNTH_FILE    synthetic hourly file to replay                    (default data_synth/synthetic_icu_20000_hourly.csv.gz,
                                                                    falls back to data/live_replay_sample.csv.gz)
  TICK_SECONDS  real seconds per simulated ICU hour                (default 1.0)
  BEDS          ICU beds                                           (default 22)
  START_HOUR    simulated hour to start from, so the ICU is already busy (default 48)
  LOOP          start again when the file is used up (1) or stop (0) (default 1)
"""
import os
import sqlite3
import time
from datetime import datetime, timezone

import icu_sim

LIVE_DB = os.environ.get("LIVE_DB", "/live/icu_live.db")
SYNTH_FILE = os.environ.get("SYNTH_FILE", "data_synth/synthetic_icu_20000_hourly.csv.gz")
FALLBACK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "live_replay_sample.csv.gz")
TICK = float(os.environ.get("TICK_SECONDS", "1.0"))
BEDS = int(os.environ.get("BEDS", "22"))
START_HOUR = int(os.environ.get("START_HOUR", "48"))
LOOP = os.environ.get("LOOP", "1") == "1"
COLS = ["sim_hour", "bed"] + icu_sim.LIVE_COLUMNS


def connect():
    os.makedirs(os.path.dirname(LIVE_DB) or ".", exist_ok=True)
    con = sqlite3.connect(LIVE_DB, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def reset(con, source):
    con.executescript("""
        DROP TABLE IF EXISTS hours; DROP TABLE IF EXISTS events; DROP TABLE IF EXISTS clock;
        CREATE TABLE hours (sim_hour INTEGER, bed INTEGER, patient_id TEXT, ICULOS INTEGER, Age REAL, Gender INTEGER,
            HR REAL, O2Sat REAL, Temp REAL, SBP REAL, MAP REAL, Resp REAL, WBC REAL,
            HR_measured INTEGER, Temp_measured INTEGER, Resp_measured INTEGER, news2_score REAL, panels_ordered INTEGER,
            abx_started INTEGER, on_antibiotics INTEGER, SepsisLabel INTEGER, sepsis_patient INTEGER, sepsis_onset_hour REAL);
        CREATE INDEX idx_hours_hour ON hours(sim_hour);
        CREATE INDEX idx_hours_patient ON hours(patient_id);
        CREATE TABLE events (sim_hour INTEGER, bed INTEGER, patient_id TEXT, event TEXT);
        CREATE TABLE clock (key TEXT PRIMARY KEY, value TEXT);
    """)
    set_clock(con, sim_hour=-1, status="starting", source=os.path.basename(source), tick_seconds=TICK, beds=BEDS)
    con.commit()


def set_clock(con, **kv):
    kv["updated_at"] = datetime.now(timezone.utc).isoformat()
    con.executemany("INSERT INTO clock(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    [(k, str(v)) for k, v in kv.items()])


def main():
    source = SYNTH_FILE if os.path.exists(SYNTH_FILE) else FALLBACK
    print(f"[feeder] replaying {source}  ->  {LIVE_DB}  ({TICK}s per ICU hour, {BEDS} beds)", flush=True)
    rep = icu_sim.Replay(icu_sim.load_patients(source), beds=BEDS)
    print(f"[feeder] {rep.sched.shape[0]:,} patients scheduled over {rep.end_hour:,} simulated hours", flush=True)
    con = connect()
    while True:
        reset(con, source)
        # load the hours before START_HOUR in one go, so the ICU is already running when the demo opens
        for h in range(0, min(START_HOUR, rep.end_hour + 1)):
            write_hour(con, rep, h)
        con.commit()
        for h in range(min(START_HOUR, rep.end_hour + 1), rep.end_hour + 1):
            t0 = time.time()
            write_hour(con, rep, h)
            set_clock(con, sim_hour=h, status="running")
            con.commit()
            if h % 25 == 0:
                print(f"[feeder] ICU hour {h:,}: {len(rep.rows_at(h))} beds occupied", flush=True)
            time.sleep(max(0.0, TICK - (time.time() - t0)))
        set_clock(con, status="finished")
        con.commit()
        if not LOOP:
            print("[feeder] finished", flush=True)
            break
        print("[feeder] reached the end of the file - starting again", flush=True)


def write_hour(con, rep, h):
    rows = rep.rows_at(h)
    if len(rows):
        con.executemany(f"INSERT INTO hours({','.join(COLS)}) VALUES({','.join('?' * len(COLS))})",
                        [tuple(None if (isinstance(v, float) and v != v) else (v.item() if hasattr(v, "item") else v) for v in r)
                         for r in rows[COLS].itertuples(index=False, name=None)])
    ev = rep.events(h, lookback=0)
    if len(ev):
        con.executemany("INSERT INTO events(sim_hour, bed, patient_id, event) VALUES(?, ?, ?, ?)",
                        [tuple(v.item() if hasattr(v, "item") else v for v in r) for r in ev.itertuples(index=False, name=None)])
    set_clock(con, sim_hour=h)


if __name__ == "__main__":
    main()
