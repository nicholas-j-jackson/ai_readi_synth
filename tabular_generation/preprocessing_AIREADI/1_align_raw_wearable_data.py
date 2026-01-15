import json, itertools, statistics
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines  as mlines
import matplotlib.patches as mpatch
import seaborn as sns
import numpy as np
import pickle
import os
import csv
from math import ceil 
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import statistics 
from typing import Tuple, List 
import warnings
import argparse
from itertools import cycle
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

INTERVAL =  None                
MAX_INTERPOLATE_WINDOW  = None
VALID_INTERPOLATE = None
WINDOW  = None          
START_HOUR = None   
CAL_RATE_INVALID_MAX = None 
INTERPOLATE_CALORIES = None  
CHECK_KCAL = None  

USE_OXYGEN_SATURATION: bool | None = None
REQ_COLS: list[str] | None = None
INVALID_VALS: dict[str, set] | None = None
SCALAR_COLS = None

ACTIVITY_MAP = {
    "walking": "act_walking",
    "running": "act_running",
    "sedentary": "act_sedentary",
    "generic": "act_generic",
    "": np.nan,
    None: np.nan,
}
SLEEP_MAP = {
    "light": "sleep_light",
    "deep": "sleep_deep",
    "rem": "sleep_rem",
    "awake": "sleep_awake",
    "": np.nan,
    None: np.nan,
}


#---------------------- Data loading functions ------------------------------- # 
def get_bloodglucose(path): 
    path = Path(path)
    with path.open() as f: 
        data = json.load(f) 
    times = [] 
    values = [] 
    for rec in data["body"]["cgm"]: 
        start_time = (rec["effective_time_frame"]["time_interval"]["start_date_time"])
        end_time = (rec["effective_time_frame"]["time_interval"]["end_date_time"])
        if start_time != end_time: 
            continue  # Skip inconsistent times
        v = rec["blood_glucose"]["value"]
        if v == 'Low': 
            v = float(-1.0)
        elif v == 'High': 
            v = float(-1.0)
        else: 
            v = float(rec["blood_glucose"]["value"])
        times.append(start_time)
        values.append(v) 
    return pd.DataFrame({"start_time": times, "blood_glucose": values}).sort_values(by='start_time')


def get_heartrate(path): 
    path = Path(path)
    with path.open() as f: 
        data = json.load(f) 
    times = [] 
    values = [] 
    for rec in data["body"]["heart_rate"]: 
        t = (rec["effective_time_frame"]["date_time"])
        v = float(rec["heart_rate"]["value"])
        times.append(t)
        values.append(v) 
    return pd.DataFrame({"date_time": times, "heart_rate": values}).sort_values(by='date_time')


def get_oxygensat(path): 
    path = Path(path)
    with path.open() as f: 
        data = json.load(f) 
    times = [] 
    values = [] 
    for rec in data["body"]["breathing"]: 
        t = (rec["effective_time_frame"]["date_time"])
        v = float(rec["oxygen_saturation"]["value"])
        times.append(t)
        values.append(v) 
    return pd.DataFrame({"date_time": times, "oxygen_saturation": values}).sort_values(by='date_time')


def get_activity(path): 
    path = Path(path)
    with path.open() as f: 
        data = json.load(f) 
    start_times = [] 
    end_times = [] 
    act_names = [] 
    act_values = [] 
    act_units = [] 
    for rec in data["body"]["activity"]: 
        start_time = (rec["effective_time_frame"]["time_interval"]["start_date_time"])
        end_time = (rec["effective_time_frame"]["time_interval"]["end_date_time"])
        act_name = rec["activity_name"]
        act_value_raw = rec["base_movement_quantity"]["value"]
        try:
            act_value = float(act_value_raw) if act_value_raw not in ("", None) else np.nan
        except ValueError:
            act_value = np.nan
        act_unit = rec["base_movement_quantity"]["unit"]
        start_times.append(start_time) 
        end_times.append(end_time)
        act_names.append(act_name)
        act_values.append(act_value)
        act_units.append(act_unit)
    return pd.DataFrame({"start_time": start_times, "end_time": end_times, "activity_name": act_names, "activity_value": act_values, "activity_units": act_units}).sort_values(by='start_time')


def get_calorie(path): 
    path = Path(path)
    with path.open() as f: 
        data = json.load(f) 
    times = [] 
    values = [] 
    for rec in data["body"]["activity"]: 
        t = (rec["effective_time_frame"]["date_time"])
        v = rec["calories_value"]["value"]
        times.append(t)
        values.append(v) 
    return pd.DataFrame({"date_time": times, "calories": values}).sort_values(by='date_time')

def get_respiratoryrate(path): 
    path = Path(path)
    with path.open() as f: 
        data = json.load(f) 
    times = [] 
    values = [] 
    for rec in data["body"]["breathing"]: 
        t = (rec["effective_time_frame"]["date_time"])
        v = rec["respiratory_rate"]["value"]
        times.append(t)
        values.append(v) 
    return pd.DataFrame({"date_time": times, "respiratory_rate": values}).sort_values(by='date_time')

def get_sleep(path): 
    #try:
    path = Path(path)
    with path.open() as f: 
        data = json.load(f) 
    start_times = [] 
    end_times = [] 
    values = [] 
    for rec in data["body"]["sleep"]: 
        start_time = (rec["effective_time_frame"]["time_interval"]["start_date_time"])
        end_time = (rec["effective_time_frame"]["time_interval"]["end_date_time"])
        v = rec["sleep_stage_state"]
        start_times.append(start_time)
        end_times.append(end_time)
        values.append(v) 
    return pd.DataFrame({"start_time": start_times, "end_times": end_times, "sleep_stage": values}).sort_values(by='start_time')

def get_stress(path): 
    path = Path(path)
    with path.open() as f: 
        data = json.load(f) 
    times = [] 
    values = [] 
    for rec in data["body"]["stress"]: 
        t = (rec["effective_time_frame"]["date_time"])
        v = rec["stress"]["value"]
        times.append(t)
        values.append(v) 
    return pd.DataFrame({"date_time": times, "stress": values}).sort_values(by='date_time')


def get_patient_data(patient_id, DATA_PATH): 
    root = Path(DATA_PATH) #Path("~/Desktop/synth_data/dataset").expanduser()
    patient = str(patient_id)
    files = {
        "blood_glucose"  : root / "wearable_blood_glucose"    / "continuous_glucose_monitoring" / "dexcom_g6" / patient / f"{patient}_DEX.json",
        "heart_rate"     : root / "wearable_activity_monitor" / "heart_rate" / "garmin_vivosmart5" / patient / f"{patient}_heartrate.json",
        "oxygen_sat"     : root / "wearable_activity_monitor" / "oxygen_saturation" / "garmin_vivosmart5" / patient / f"{patient}_oxygensaturation.json",
        "activity"       : root / "wearable_activity_monitor" / "physical_activity" / "garmin_vivosmart5" / patient / f"{patient}_activity.json",
        "calorie"        : root / "wearable_activity_monitor" / "physical_activity_calorie" / "garmin_vivosmart5" / patient / f"{patient}_calorie.json",
        "resp_rate"      : root / "wearable_activity_monitor" / "respiratory_rate" / "garmin_vivosmart5" / patient / f"{patient}_respiratoryrate.json",
        "sleep"          : root / "wearable_activity_monitor" / "sleep" / "garmin_vivosmart5" / patient / f"{patient}_sleep.json",
        "stress"         : root / "wearable_activity_monitor" / "stress" / "garmin_vivosmart5" / patient / f"{patient}_stress.json",
    }
    patient_dataframes = {
        'blood_glucose': get_bloodglucose(files["blood_glucose"]),
        'heart_rate': get_heartrate(files["heart_rate"]),
        'oxygen_sat': get_oxygensat(files["oxygen_sat"]),
        'activity': get_activity(files["activity"]),
        'calorie': get_calorie(files["calorie"]),
        'resp_rate': get_respiratoryrate(files["resp_rate"]),
        'sleep': get_sleep(files["sleep"]),
        'stress': get_stress(files["stress"])
    }

    return patient_id, patient_dataframes

def load_all_data(save_dir, DATA_PATH, PATIENT_IDS): 
    os.makedirs(save_dir, exist_ok=True)
    all_results = {}                 
    max_workers = os.cpu_count()  
    #process in parallel to go faster 
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(get_patient_data, pid, DATA_PATH): pid for pid in PATIENT_IDS}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Patients"):
            pid = futures[fut]     
            try:
                _, patient_dfs = fut.result()
                all_results[pid] = patient_dfs
            except Exception as e: 
                print(f"[WARN] patient {pid} raised {e!r}")
    #save to file 
    out_file = Path("population_analysis_final/all_patients.pkl")
    with out_file.open("wb") as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved to", out_file.resolve())
    return all_results


def clean_and_index(df: pd.DataFrame, time_col: str, value_col: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float, name=value_col)
    return (
        df[[time_col, value_col]]
        .rename(columns={time_col: "t"})
        .assign(t=lambda x: pd.to_datetime(x["t"], utc=True))
        .set_index("t")[value_col]
        .sort_index()
    )

def interpolate_series(raw: pd.Series, invalid_set: set, grid: pd.DatetimeIndex) -> Tuple[pd.Series, List[float]]:
    """
    One entry per interpolated point: the neighbour-to-neighbour distance
    used for that interpolation, in **minutes**.
    • For two-sided interpolation  → gap = next_valid – prev_valid.  
    • For the *first* bucket of the day (one-sided fill allowed)  
        → gap = |ts – next_valid|.
    If VALID_INTERPOLATE: all not filled entires are Nan 
    IF not VALID_INTERPOLATE: all not filled entries are - 1
    """
    if raw.empty:
        return pd.Series(index=grid, dtype=float), []
    raw_clean = raw.mask(raw.isin(invalid_set))      
    if raw_clean.index.has_duplicates:
        raw_clean = raw_clean[~raw_clean.index.duplicated(keep="first")]
    aug = raw_clean.reindex(raw_clean.index.union(grid)).sort_index()
    interp = aug.interpolate(method="time")
    aligned = interp.reindex(grid)

    gaps_min =  []
    need_check = aligned.index.difference(raw_clean.dropna().index)
    first_bucket = grid[0]
    for ts in need_check:
        if pd.isna(aligned.at[ts]):
            continue
        prev_idx = raw_clean[:ts].last_valid_index()
        next_idx = raw_clean[ts:].first_valid_index()
        #first bucket may copy just 1 next neighbour only
        if ts == first_bucket and prev_idx is None and next_idx is not None:
            gap = (next_idx - ts)
            if gap <= MAX_INTERPOLATE_WINDOW:
                aligned.at[ts] = raw_clean.at[next_idx]
                gaps_min.append(gap.total_seconds() / 60.0)
            else:
                aligned.at[ts] = np.nan if VALID_INTERPOLATE else -1
            continue
        # need both neighbours for all other buckets
        if prev_idx is None or next_idx is None:
            aligned.at[ts] = np.nan if VALID_INTERPOLATE else -1
            continue
        gap = next_idx - prev_idx
        if gap <= MAX_INTERPOLATE_WINDOW:
            gaps_min.append(gap.total_seconds() / 60.0)
        else:
            aligned.at[ts] = np.nan if VALID_INTERPOLATE else -1
    if not VALID_INTERPOLATE:
        aligned = aligned.fillna(-1)
    return aligned, gaps_min

#majority label vote per timestamp (and track mixed columns)
def interval_df_to_series(df, start_col, end_col, value_col, mapping, *, return_mixed=False):
    if df.empty:
        empty = pd.Series(dtype=object, name=value_col)
        return (empty, empty.astype(float)) if return_mixed else empty

    df = ( df[[start_col, end_col, value_col]]
          .rename(columns={start_col: "start", end_col: "end", value_col: "val"})
          .assign(
              start=lambda x: pd.to_datetime(x["start"], utc=True),
              end  =lambda x: pd.to_datetime(x["end"],   utc=True),
              val  =lambda x: x["val"].map(mapping).astype("object"),
          )
          .dropna(subset=["val"])
    )
    full_grid = pd.date_range(df["start"].min().floor(INTERVAL), df["end"].max().ceil(INTERVAL) - pd.Timedelta(seconds=1), freq=INTERVAL, tz="UTC")
    counts = defaultdict(Counter)
    for s, e, v in df.itertuples(index=False):
        for b in pd.date_range(s.floor(INTERVAL), (e - pd.Timedelta(seconds=1)).floor(INTERVAL), freq=INTERVAL, tz="UTC"):
            left  = max(s, b)
            right = min(e, b + pd.Timedelta(INTERVAL))
            counts[b][v] += (right - left).total_seconds()
    labels, mixed = [], []
    for b in full_grid:
        if b in counts:
            # majority label
            label = max(counts[b].items(), key=lambda t: (t[1], -list(counts[b]).index(t[0])))[0]
            labels.append(label)
            # mixed if >1 labels present
            mixed.append(len(counts[b]) > 1)        
        else:
            labels.append(np.nan)
            mixed.append(False)
    series_labels  = pd.Series(labels, index=full_grid, name=value_col)
    series_mix  = pd.Series(mixed,  index=full_grid, name=f"{value_col}_mixed").astype(float)
    return (series_labels, series_mix) if return_mixed else series_labels

#Average steps per min for each interval (and flag mixed if we had to average)
def interval_steps_to_series(df, start_col, end_col, value_col, *, return_mixed=False):
    if df.empty:
        empty = pd.Series(dtype=float, name="steps_per_min")
        return (empty, empty.astype(float)) if return_mixed else empty
    df = (df[[start_col, end_col, value_col]]
          .rename(columns={start_col: "start", end_col: "end", value_col: "steps"})
          .assign(
              start=lambda x: pd.to_datetime(x["start"], utc=True),
              end  =lambda x: pd.to_datetime(x["end"],   utc=True),
              steps=lambda x: pd.to_numeric(x["steps"], errors="coerce"),
          )
          .dropna(subset=["steps"])
    )
    #make grid 
    bucket_secs = pd.Timedelta(INTERVAL).total_seconds()
    full_grid  = pd.date_range(df["start"].min().floor(INTERVAL), df["end"].max().ceil(INTERVAL) - pd.Timedelta(seconds=1), freq=INTERVAL, tz="UTC")
    tot_steps  = defaultdict(float)
    cov_seconds = defaultdict(float)
    #align to buckets 
    for s, e, st in df.itertuples(index=False, name=None):
        dur = (e - s).total_seconds()
        if dur == 0:
            continue
        # steps per second
        rate = st / dur                   
        for b in pd.date_range(s.floor(INTERVAL), (e - pd.Timedelta(seconds=1)).floor(INTERVAL), freq=INTERVAL, tz="UTC"):
            left = max(s, b)
            right = min(e, b + pd.Timedelta(INTERVAL))
            overlap = (right - left).total_seconds()
            tot_steps[b]  += rate * overlap
            cov_seconds[b] += overlap
    #get steps per min for each interval (and mixed flags if needed)
    series_val  = pd.Series(np.nan,  index=full_grid, name="steps_per_min")
    series_mix  = pd.Series(False,   index=full_grid, name="steps_mixed")
    for b in full_grid:
        if b in tot_steps:
            series_val[b] = tot_steps[b] / (bucket_secs / 60)
            # mixed if not fully covered
            series_mix[b] = cov_seconds[b] < bucket_secs   
    return (series_val, series_mix.astype(float)) if return_mixed else series_val

def bin_counts(df: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series]:
    if not cols:
        return pd.Series(dtype=int), pd.Series(dtype=int)
    valid = df[cols].eq(1).sum()
    nan_  = df[cols].isna().sum()
    return valid, nan_


def interval_calories_to_series(df_cal, df_act, *, return_mixed=False):
    #Calories accumulate separately for each activity label and occasionally reset (could be glitch)
    #Sedentary buckets are always 0 – never interpolated or averaged
    #majority rule activity if overlap but average kcal / min and mark mixed 
    #interpolate within label only. never for sedentary. within window 
    #If VALID_INTERPOLATE, nan. if not, -1
    if df_cal.empty or df_act.empty:
        empty = pd.Series(dtype=float, name="kcal_per_min")
        return (empty, empty.astype(float)) if return_mixed else empty
    if df_act["start_time"].dtype == "O":  # object → likely strings
        df_act = df_act.assign(
            start_time=pd.to_datetime(df_act["start_time"], utc=True, errors="coerce"),
            end_time=pd.to_datetime(df_act["end_time"],   utc=True, errors="coerce"),
        ).sort_values("start_time")
    act_labels = interval_df_to_series( df_act, "start_time", "end_time", "activity_name", ACTIVITY_MAP)
    full_grid = pd.date_range(act_labels.index.min(), act_labels.index.max(), freq=INTERVAL, tz="UTC")
    cal = (df_cal[["date_time", "calories"]].rename(columns={"date_time": "t"}).assign( t=lambda x: pd.to_datetime(x["t"], utc=True), calories=lambda x: pd.to_numeric(x["calories"], errors="coerce")).set_index("t").sort_index())
    # map each calorie row to its activity label (majority rule) 
    # use floor(INTERVAL) here – that’s accurate because df_cal timestamps are inside the activity interval that dominates that minute bucket)
    cal["label"] = cal.index.floor(INTERVAL).map(act_labels)
    bucket_sec = pd.Timedelta(INTERVAL).total_seconds()
    rate  = pd.Series(np.nan,  index=full_grid, name="kcal_per_min")
    mixed = pd.Series(False,   index=full_grid, name="kcal_mixed")
    # sedentary is zero
    sed_mask = act_labels == "act_sedentary"
    rate.loc[sed_mask] = 0.0
    #loop through labels 
    gaps_min = []         
    for label, group in cal.groupby("label", sort=False):
        if pd.isna(label) or label == "act_sedentary":
            continue
        group = group[~group["calories"].isna()].copy()
        group = group.groupby(group.index).agg({'calories': 'mean'})
        # GLITCH / RESET DETECTION
        diff = group["calories"].diff()
        reset_idx = diff[diff < 0].index
        #glitch = false reset --> rebounds 
        glitch_mask = pd.Series(False, index=group.index)
        for idx in reset_idx:
            i = group.index.get_loc(idx)
            prev = group["calories"].iloc[i - 1]
            nxt1 = group["calories"].iloc[i]
            nxt2 = group["calories"].iloc[i + 1] if i + 1 < len(group) else prev
            nxt3 = group["calories"].iloc[i + 2] if i + 2 < len(group) else prev
            if (nxt2 <= prev) and (nxt3 <= prev):
                glitch_mask.iloc[i] = True  
        group.loc[glitch_mask, "calories"] = np.nan
        group["is_reset"] = (diff < 0) & ~glitch_mask
        #iterate through cumulative rows for this label and reset to 0 after each reset
        last_val = 0.0           
        # previous calorie timestamp *for this label* 
        last_ts  = None   
        #start of current activity interval        
        activity_start = None     
        for ts, row in group.iterrows():
            cur_val   = row["calories"]
            if pd.isna(cur_val):
                continue
            bucket_ts = ts.floor(INTERVAL)
            if (activity_start is None) or (act_labels.get(bucket_ts) != label):
                activity_start = df_act.loc[(df_act["start_time"] <= ts) & (df_act["end_time"] > ts)]
                if activity_start.empty: 
                    last_ts = ts 
                    continue 
                else: 
                    activity_start = activity_start.iloc[0]["start_time"].tz_convert("UTC")
                last_val = 0.0
                last_ts  = activity_start
            #explicit reset
            if row["is_reset"]:
                last_val = 0.0
                last_ts  = ts
                continue
            #delta kcal / delta sec relative to last sample (or activity start)
            delta_kcal = cur_val - last_val
            delta_sec  = (ts - last_ts).total_seconds()
            if delta_kcal < 0 or delta_sec == 0:
                last_val, last_ts = cur_val, ts
                continue
            slope = delta_kcal / delta_sec   
            #distribute into INTERVAL buckets between last_ts and ts
            buckets = pd.date_range(last_ts.floor(INTERVAL), (ts - pd.Timedelta(seconds=1)).floor(INTERVAL), freq=INTERVAL, tz="UTC")
            for b in buckets:
                if act_labels.get(b) != label:
                    continue
                left = max(last_ts, b)
                right  = min(ts, b + pd.Timedelta(INTERVAL))
                ov_sec = (right - left).total_seconds()
                inc = (slope * ov_sec) / (bucket_sec / 60)  
                if pd.isna(rate.at[b]):
                    rate.at[b] = inc
                else:  
                    rate.at[b] = (rate.at[b] + inc) / 2
                    mixed.at[b] = True
            last_val, last_ts = cur_val, ts
    #clean up / mark invalids and missings
    if VALID_INTERPOLATE:
        rate.loc[rate > CAL_RATE_INVALID_MAX] = np.nan
    else:
        rate.loc[rate > CAL_RATE_INVALID_MAX] = -1
    if VALID_INTERPOLATE:
        rate.loc[act_labels.reindex(rate.index).isna()] = np.nan
    else:
        rate.loc[act_labels.reindex(rate.index).isna()] = -1
    # per label interpolation
    if INTERPOLATE_CALORIES:
        for label in ACTIVITY_MAP.values():
            if label in ("act_sedentary", np.nan):
                continue
            mask_label = act_labels == label
            seg = rate.where(mask_label)
            seg = seg.mask((seg == -1) | (seg.isna()))
            originally_missing = seg.isna()
            nan_locs = seg.index[seg.isna()]
            filled = seg.interpolate( "time", limit_area="inside", limit_direction="both")
            for ts in nan_locs:
                if pd.isna(filled.at[ts]):
                    continue
                prev_idx = seg[:ts].last_valid_index()
                next_idx = seg[ts:].first_valid_index()
                if prev_idx is None or next_idx is None:
                    gap = np.inf
                else:
                    gap = (next_idx - prev_idx).total_seconds() / 60
                if gap <= MAX_INTERPOLATE_WINDOW.total_seconds() / 60:
                    gaps_min.append(gap)
            rate.update(filled[mask_label])
            still_missing = originally_missing & filled.isna() 
            if not VALID_INTERPOLATE: 
                rate.loc[still_missing & mask_label]
    if not VALID_INTERPOLATE:
        rate = rate.fillna(-1)
        rate.loc[rate > CAL_RATE_INVALID_MAX] = -1. 
    if VALID_INTERPOLATE: 
        rate.loc[rate > CAL_RATE_INVALID_MAX] = np.nan 
        rate.loc[rate < 0] = np.nan 
    return (rate, mixed.astype(float), gaps_min) if return_mixed else rate


def load_and_prepare(pid: int, DATA_PATH: str) -> dict[str, pd.Series]:
    _, dfs = get_patient_data(pid, DATA_PATH)
    #scalar data 
    data = {
        "blood_glucose": clean_and_index(dfs["blood_glucose"], "start_time", "blood_glucose"),
        "heart_rate":    clean_and_index(dfs["heart_rate"],    "date_time",  "heart_rate"),
        "resp_rate":     clean_and_index(dfs["resp_rate"],     "date_time",  "respiratory_rate"),
        "stress":        clean_and_index(dfs["stress"],        "date_time",  "stress"),
    }
    if USE_OXYGEN_SATURATION: 
        data["oxygen_sat"] = clean_and_index(dfs["oxygen_sat"],    "date_time",  "oxygen_saturation")
    # interval data
    data["steps_per_min"], data["steps_mixed"] = interval_steps_to_series(dfs["activity"], "start_time", "end_time", "activity_value", return_mixed=True)
    data["activity"], data["act_mixed"] = interval_df_to_series(dfs["activity"], "start_time", "end_time", "activity_name", ACTIVITY_MAP, return_mixed=True)
    data["sleep"] = interval_df_to_series(dfs["sleep"], "start_time", "end_times", "sleep_stage", SLEEP_MAP)
    
    #print(dfs['sleep'])
    
    data["kcal_per_min"], data["kcal_mixed"], data["_kcal_gaps"] = interval_calories_to_series(dfs["calorie"], dfs["activity"], return_mixed=True)
    # one hot encode
    for key, mapping in (("activity", ACTIVITY_MAP), ("sleep", SLEEP_MAP)):
        dummies = pd.get_dummies(data[key], prefix=None, dtype=float)
        for col in mapping.values():
            if col not in dummies.columns:
                dummies[col] = 0.0
        dummies = dummies[mapping.values()]
        all_zero = dummies.sum(axis=1) == 0
        dummies.loc[all_zero, :] = -1.0
        for col in dummies.columns:
            data[col] = dummies[col]
    return data

def dump_failed_slice(out_dir: Path, day_start: pd.Timestamp, aligned: pd.DataFrame, scalar_cols: list[str]): 
    #goes to skipped directory 
    mask = aligned[scalar_cols].isna().any(axis=1)
    fail_df = aligned.loc[mask, scalar_cols]
    if fail_df.empty:
        return
    skip_dir = out_dir / "skipped"
    skip_dir.mkdir(exist_ok=True)
    fname = skip_dir / f"FAIL_{day_start:%Y%m%d}.csv"
    fail_df.to_csv(fname, index_label="timestamp")
    pretty_txt = fail_df.to_string(index=True, col_space=10, justify="right", float_format="%.3f", na_rep="NaN")
    (fname.with_suffix(".txt").write_text(pretty_txt))
    #print(f"  › slice {day_start:%Y-m-d} rejected – {mask.sum()} NaN rows written to {fname.name}")

def align_patient(pid: int, DATA_PATH: str) -> None:
    # ---------- out directory -------------------------------------------------
    hours = int(MAX_INTERPOLATE_WINDOW.total_seconds() // 3600)
    if USE_OXYGEN_SATURATION: 
        msg_dir = Path(f"aligned_data/with_oxygensat/{INTERVAL}_interval_{hours}hr_maxinterp_{VALID_INTERPOLATE}_validinterp_{START_HOUR}start")
        out_dir = Path(f"aligned_data/with_oxygensat/{INTERVAL}_interval_{hours}hr_maxinterp_{VALID_INTERPOLATE}_validinterp_{START_HOUR}start/{pid}")
    else: 
        msg_dir = Path(f"aligned_data/no_oxygensat/{INTERVAL}_interval_{hours}hr_maxinterp_{VALID_INTERPOLATE}_validinterp_{START_HOUR}start")
        out_dir = Path(f"aligned_data/no_oxygensat/{INTERVAL}_interval_{hours}hr_maxinterp_{VALID_INTERPOLATE}_validinterp_{START_HOUR}start/{pid}")
    out_dir.mkdir(parents=True, exist_ok=True)
    #-----prep-----------------------
    #print(f"[{pid}] processing …")
    prepared = load_and_prepare(pid, DATA_PATH)

    if any(prepared[c].empty for c in REQ_COLS if c != "_kcal_gaps"):
        print(f"[{pid}] missing an entire channel – skipped")
        with (Path(f"{msg_dir}/errors.txt")).open("a") as f:
            f.write(f"Patient {pid} skipped – missing channels: {', '.join(missing_channels)}\n")
        return
    '''
    spans = [] 
    for key, s in prepared.items(): 
        if key != "_kcal_gaps": 
            spans.append((s.index.min(), s.index.max()))
    '''
    spans = [ (s.index.min(), s.index.max()) for key, s in prepared.items()  if key != "_kcal_gaps"]
    #print(spans) 

    start_global = max(s[0] for s in spans)
    end_global   = min(s[1] for s in spans)
    t0_floor = start_global.normalize() + pd.Timedelta(hours=START_HOUR)
    t0 = t0_floor if start_global < t0_floor else t0_floor + pd.Timedelta(days=1)
    #print(start_global, end_global)
    # ---------- header -----------------------------------------------------
    with (out_dir / "info.txt").open("w") as f:
        f.write("============================================================\n")
        f.write(f" Patient {pid} | interval {INTERVAL}\n")
        f.write("============================================================\n")
        f.write(f"Max-interpolate window : {MAX_INTERPOLATE_WINDOW}\n")
        f.write(f"Interpolation mode     : {'valid-only' if VALID_INTERPOLATE else 'fill −1'}\n")
        f.write(f"Slice window           : {WINDOW}\n")
        f.write(f"Slice start hour (UTC) : {START_HOUR:02d}:00\n")
        f.write(f"Required cols          : {', '.join(REQ_COLS)}\n")
        f.write("============================================================\n\n")
    # ------------ helpers & accumulators ----------------------------------
    interval_min   = int(pd.Timedelta(INTERVAL).total_seconds() // 60)
    bins_per_day   = int(WINDOW / pd.Timedelta(INTERVAL))

    activity_cols  = [c for c in REQ_COLS if c.startswith("act_")]
    sleep_cols     = [c for c in REQ_COLS if c.startswith("sleep_")]
    scalar_cols    = [c for c in REQ_COLS if c not in activity_cols + sleep_cols + ["steps_per_min"] + ["kcal_per_min"]]

    tot_gaps       = defaultdict(list)
    tot_interp     = Counter({c: 0 for c in REQ_COLS})
    tot_nan        = Counter({c: 0 for c in REQ_COLS})
    tot_invalid    = Counter({c: 0 for c in REQ_COLS})

    tot_valid_act  = Counter({c: 0 for c in activity_cols})
    tot_valid_slp  = Counter({c: 0 for c in sleep_cols})
    tot_mixed_act  = Counter({c: 0 for c in activity_cols})
    tot_mixed_step = 0
    tot_mixed_kcal = 0 

    overall_steps        = []
    overall_steps_by_act = {c: [] for c in activity_cols}
    day_counter = 0
    tot_sleep_all_neg1 = 0 
    # ======================================================================
    if t0 + WINDOW > end_global: 
        with (Path(f"{msg_dir}/errors.txt")).open("a") as f:
            f.write(f"Not a full day of data from start time for patient {pid} under settings interval = {INTERVAL}, max_interpolate = {MAX_INTERPOLATE_WINDOW}, valid_interpolate = {VALID_INTERPOLATE} start_hour = {START_HOUR}, use_oxygen_sat = {USE_OXYGEN_SATURATION}, interpolate_calories = {INTERPOLATE_CALORIES}, check_kcal = {CHECK_KCAL}\n")
        return
    while t0 + WINDOW <= end_global:
        t1   = t0 + WINDOW
        grid = pd.date_range(t0, t1 - pd.Timedelta(seconds=1), freq=INTERVAL, tz="UTC")
        aligned = pd.DataFrame(index=grid)
        gaps_day   = defaultdict(list)
        nan_day    = Counter({c: 0 for c in REQ_COLS})
        invalid_day= Counter({c: 0 for c in REQ_COLS})
        #interpolate scalar data 
        for col in scalar_cols:
            seg = prepared[col].loc[t0 - MAX_INTERPOLATE_WINDOW : t1 + MAX_INTERPOLATE_WINDOW]
            series, gaps = interpolate_series(seg, INVALID_VALS[col], grid)
            aligned[col] = series
            gaps_day[col].extend(gaps)
        gaps_day["kcal_per_min"].extend(prepared["_kcal_gaps"])
        # ---------- copy steps + one-hots ----------------------------------
        aligned["steps_per_min"] = prepared["steps_per_min"].reindex(grid)
        aligned["steps_mixed"]   = prepared["steps_mixed"].reindex(grid)
        aligned["act_mixed"]     = prepared["act_mixed"].reindex(grid)
        aligned["kcal_per_min"] = prepared["kcal_per_min"].reindex(grid)
        aligned["kcal_mixed"]   = prepared["kcal_mixed"].reindex(grid)
        for col in activity_cols + sleep_cols:
            aligned[col] = prepared[col].reindex(grid)
        # ---------- acceptance test ----------------------------------------
        #exclude oxygen sat from data 
        cols_to_check = [c for c in aligned.columns if c not in sleep_cols + ["steps_mixed", "act_mixed", "kcal_mixed", "kcal_per_min", "oxygen_sat"]]
        if CHECK_KCAL: 
            cols_to_check = cols_to_check + ["kcal_per_min"]
        no_activity_mask = (aligned[activity_cols] == -1).all(axis=1)
        if USE_OXYGEN_SATURATION: 
            aligned["oxygen_sat"] = aligned["oxygen_sat"].fillna(-1)
            count_oxygen_sat_invalids = (aligned["oxygen_sat"] == -1).sum()
            all_oxygen_invalids = (count_oxygen_sat_invalids >= (WINDOW.total_seconds() / (int(INTERVAL.split("min")[0])*60)))
        #always na coming out of interpolate if valid_interpolate == true 
        if VALID_INTERPOLATE: 
            aligned.loc[no_activity_mask, "steps_per_min"] = np.nan 
            if aligned[cols_to_check].isna().any().any() or no_activity_mask.any() or (USE_OXYGEN_SATURATION and all_oxygen_invalids):
                if USE_OXYGEN_SATURATION: 
                    dump_failed_slice(out_dir, t0, aligned, cols_to_check + ["oxygen_sat"])
                else: 
                    dump_failed_slice(out_dir, t0, aligned, cols_to_check) 
                t0 += pd.Timedelta("1d")
                continue
        aligned.loc[no_activity_mask, "steps_per_min"] = -1
        #--------export if good------------
        day_counter += 1
        csv_name = out_dir / f"{pid}_{t0:%Y%m%dT%H%M}.csv"
        aligned_csv = aligned.copy() 
        aligned_csv = aligned_csv.drop(columns=["steps_mixed", "act_mixed", "kcal_mixed"])
        aligned_csv.to_csv(csv_name, index_label="timestamp")
        pretty_txt = aligned_csv.to_string(index=True, col_space=10, justify="right", float_format="%.3f", na_rep="NaN")
        (csv_name.with_suffix(".txt").write_text(pretty_txt))
        kcal_only = aligned_csv.copy()
        kcal_only = kcal_only.drop(columns=["heart_rate", "resp_rate", "stress", "blood_glucose", "sleep_awake", "sleep_rem", "sleep_deep", "sleep_light"])
        #kcal_only = kcal_only[kcal_only["act_walking"] == 1].copy()
        pretty_2 = kcal_only.to_string(index=True, col_space=10, justify="right", float_format="%.3f", na_rep="NaN")
        cal_csv = out_dir / f"{pid}_{t0:%Y%m%dT%H%M}_kcal.csv"
        cal_csv.with_suffix(".txt").write_text(pretty_2)
        # ---------- per-day stats ------------------------------------------
        for c in REQ_COLS:
            nan_day[c]     = int(aligned[c].isna().sum())
            invalid_day[c] = int(aligned[c].isin(INVALID_VALS.get(c, set())).sum())

        sleep_all_neg1_day = int((aligned[sleep_cols].eq(-1).all(axis=1)).sum())
        tot_sleep_all_neg1 += sleep_all_neg1_day

        valid_act, _ = bin_counts(aligned, activity_cols)
        valid_slp, _ = bin_counts(aligned, sleep_cols)

        mixed_act_mask  = aligned["act_mixed"] == 1
        mixed_step_mask = aligned["steps_mixed"] == 1
        day_mixed_step  = int(mixed_step_mask.sum())
        day_mixed_act   = Counter({c: int((mixed_act_mask & (aligned[c] == 1)).sum()) for c in activity_cols})
        step_vals   = aligned["steps_per_min"].dropna()
        steps_by_act= {c: aligned.loc[aligned[c] == 1, "steps_per_min"].dropna() for c in activity_cols}
        mixed_kcal_mask = aligned["kcal_mixed"] == 1
        day_mixed_kcal  = int(mixed_kcal_mask.sum())
        tot_mixed_kcal += day_mixed_kcal
        # ---------- write day report ---------------------------------------
        with (out_dir / "info.txt").open("a") as f:
            f.write(f"──────── Day {day_counter} ({t0:%Y-%m-%d}) ────────\n")
            # interpolation + gaps
            all_gaps = list(itertools.chain.from_iterable(gaps_day[c] for c in scalar_cols))
            f.write(f"Interpolated points   : {len(all_gaps)}\n")
            if all_gaps:
                f.write(f"  gap min/med/mean/max "
                        f"{min(all_gaps):4.1f}/{statistics.median(all_gaps):4.1f}/"
                        f"{statistics.mean(all_gaps):4.1f}/{max(all_gaps):4.1f} \n")
            for c in scalar_cols + ["kcal_per_min"]:
                if gaps_day[c]:
                    f.write(f"  {c:<15}: {len(gaps_day[c]):5d} "
                            f"(min {min(gaps_day[c]):4.1f}  med {statistics.median(gaps_day[c]):4.1f}  "
                            f"mean {statistics.mean(gaps_day[c]):4.1f}  max {max(gaps_day[c]):4.1f})\n")
            # NaN / invalid breakdown
            f.write("\nNaN / invalid counts\n")
            for c in REQ_COLS:
                total = aligned.shape[0]
                nan_pct     = nan_day[c]     / total * 100
                inv_pct     = invalid_day[c] / total * 100
                f.write(f"  {c:<15}: NaN {nan_day[c]:4d} ({nan_pct:5.1f}%)   "
                        f"inv {invalid_day[c]:4d} ({inv_pct:5.1f}%)\n")
            # activity
            act_hours_tot = sum(valid_act[c] for c in activity_cols) * interval_min / 60
            f.write("\nActivity (hrs | %-act | %-day)\n")
            for c in activity_cols:
                hrs = valid_act[c] * interval_min / 60
                f.write(f"  {c:<15}: {hrs:5.2f}  "
                        f"{hrs/act_hours_tot*100 if act_hours_tot else 0:6.1f}%  "
                        f"{valid_act[c]/bins_per_day*100:6.1f}%\n")
            # mixed activity / steps
            mixed_total = sum(day_mixed_act.values())
            f.write(f"\nMixed-activity buckets : {mixed_total} ({mixed_total/bins_per_day*100:4.1f}% rows)\n")
            for c, v in day_mixed_act.items():
                if v:
                    f.write(f"  {c:<15}: {v:5d} "
                            f"{v/mixed_total*100 if mixed_total else 0:6.1f}%-mix "
                            f"{v/valid_act[c]*100 if valid_act[c] else 0:6.1f}%-{c} "
                            f"{v/bins_per_day*100:6.1f}%-rows\n")
            f.write(f"\nMixed step buckets     : {day_mixed_step} ({day_mixed_step/bins_per_day*100:4.1f}% rows)\n")
            # steps
            if not step_vals.empty:
                f.write("\nSteps/min (all rows)   "
                        f"mean {step_vals.mean():.2f}  med {step_vals.median():.2f}  "
                        f"min {step_vals.min():.2f}  max {step_vals.max():.2f}\n")
            f.write("  by activity:\n")
            for c, vals in steps_by_act.items():
                if not vals.empty:
                    f.write(f"    {c:<13} mean {vals.mean():.2f}  med {vals.median():.2f}  "
                            f"min {vals.min():.2f}  max {vals.max():.2f}\n")
            f.write(f"\nMixed calorie buckets     : {day_mixed_kcal} ({day_mixed_kcal/bins_per_day*100:4.1f}% rows)\n")
            # sleep
            sleep_hours_tot = sum(valid_slp[c] for c in sleep_cols) * interval_min / 60
            f.write("\nSleep stages (hrs | %-sleep | %-day)\n")
            for c in sleep_cols:
                hrs = valid_slp[c] * interval_min / 60
                f.write(f"  {c:<15}: {hrs:5.2f}  "
                        f"{hrs/sleep_hours_tot*100 if sleep_hours_tot else 0:6.1f}%  "
                        f"{valid_slp[c]/bins_per_day*100:6.1f}%\n")
            f.write("──────────────────────────────────────────────\n\n")
        # ---------- accumulate overall totals --------------------------------
        for c in scalar_cols + ["kcal_per_min"]:
            tot_gaps[c].extend(gaps_day[c])
            tot_interp[c]  += len(gaps_day[c])
        for c in REQ_COLS:
            tot_nan[c]     += nan_day[c]
            tot_invalid[c] += invalid_day[c]
        for c in activity_cols:
            tot_valid_act[c]  += valid_act[c]
            tot_mixed_act[c]  += day_mixed_act[c]
        for c in sleep_cols:
            tot_valid_slp[c]  += valid_slp[c]
        tot_mixed_step += day_mixed_step
        overall_steps.extend(step_vals)
        for c, vals in steps_by_act.items():
            overall_steps_by_act[c].extend(vals)
        t0 += pd.Timedelta("1d")

    # overall footer
    total_rows  = day_counter * bins_per_day                    
    non_sleep   = [c for c in REQ_COLS if not c.startswith("sleep_")]
    with (out_dir / "info.txt").open("a") as f:
        f.write("============================================================\n")
        f.write(f" Summary for patient {pid}\n")
        f.write("============================================================\n")
        f.write(f"Days saved : {day_counter}\n\n")
        if day_counter > 0:
            # ---------- interpolation + gaps ----------------------------
            all_gaps = list(itertools.chain.from_iterable(tot_gaps[c] for c in scalar_cols))
            f.write(f"Total interpolated     : {len(all_gaps)}\n")
            if all_gaps:
                f.write(f"Gap min/med/mean/max   : "
                        f"{min(all_gaps):4.1f}/"
                        f"{statistics.median(all_gaps):4.1f}/"
                        f"{statistics.mean(all_gaps):4.1f}/"
                        f"{max(all_gaps):4.1f} min\n")
            f.write(" • by column\n")
            for c in scalar_cols + ["kcal_per_min"]:
                if tot_gaps[c]:
                    f.write(f"   {c:<15}: {len(tot_gaps[c]):8d}  "
                            f"min {min(tot_gaps[c]):4.1f}  "
                            f"med {statistics.median(tot_gaps[c]):4.1f}  "
                            f"mean {statistics.mean(tot_gaps[c]):4.1f}  "
                            f"max {max(tot_gaps[c]):4.1f}\n")
            # ---------- NaN / invalid overall ----------------------------
            f.write("\nNaN / invalid overall\n")
            for c in REQ_COLS:
                nan_pct = tot_nan[c]     / total_rows * 100
                inv_pct = tot_invalid[c] / total_rows * 100
                f.write(f"  {c:<15}: NaN {tot_nan[c]:6d} ({nan_pct:5.2f}%)   "
                        f"inv {tot_invalid[c]:6d} ({inv_pct:5.2f}%)\n")
            # ---------- activity summary ---------------------------------
            act_hours_tot = (sum(tot_valid_act[c] for c in activity_cols) * interval_min / 60)
            f.write("\nActivity time (hrs | %-act | %-rows)\n")
            for c in activity_cols:
                hrs = tot_valid_act[c] * interval_min / 60
                f.write(f"  {c:<15}: {hrs:6.2f}  "
                        f"{hrs/act_hours_tot*100 if act_hours_tot else 0:6.1f}%  "
                        f"{tot_valid_act[c]/total_rows*100:6.1f}%\n")
            mixed_tot = sum(tot_mixed_act.values())
            f.write(f"\nMixed-activity buckets : {mixed_tot} "
                    f"({mixed_tot/total_rows*100:4.1f}% rows)\n")
            for c, v in tot_mixed_act.items():
                if v:
                    f.write(f"  {c:<15}: {v:6d}  "
                            f"{v/mixed_tot*100 if mixed_tot else 0:6.1f}%-mix  "
                            f"{v/tot_valid_act[c]*100 if tot_valid_act[c] else 0:6.1f}%-{c}  "
                            f"{v/total_rows*100:6.1f}%-rows\n")
            f.write(f"\nMixed step buckets     : {tot_mixed_step} "
                    f"({tot_mixed_step/total_rows*100:4.1f}% rows)\n")
            f.write(f"\nMixed calorie buckets     : {tot_mixed_kcal} "
                    f"({tot_mixed_kcal/total_rows*100:4.1f}% rows)\n")
            # ---------- steps --------------------------------------------
            if overall_steps:
                arr = np.asarray(overall_steps)
                f.write("\nSteps/min (all rows)\n")
                f.write(f"  mean {arr.mean():.2f}  med {np.median(arr):.2f}  "
                        f"min {arr.min():.2f}  max {arr.max():.2f}\n")
                f.write("  by activity:\n")
                for c, vals in overall_steps_by_act.items():
                    if vals:
                        a = np.asarray(vals)
                        f.write(f"    {c:<13} mean {a.mean():.2f}  med {np.median(a):.2f}  "
                                f"min {a.min():.2f}  max {a.max():.2f}\n")
            # ---------- sleep -------------------------------------------
            slp_hours_tot = (sum(tot_valid_slp[c] for c in sleep_cols)  * interval_min / 60)
            f.write("\nSleep stages (hrs | %-sleep | %-rows)\n")
            for c in sleep_cols:
                hrs = tot_valid_slp[c] * interval_min / 60
                f.write(f"  {c:<15}: {hrs:6.2f}  "
                        f"{hrs/slp_hours_tot*100 if slp_hours_tot else 0:6.1f}%  "
                        f"{tot_valid_slp[c]/total_rows*100:6.1f}%\n")
            f.write("============================================================\n")
    # CSV EVALUATION SUMMARY 
    csv_path = Path(f"{msg_dir}/evaluation_summary.csv")

    sleep_all_neg1_pct = (tot_sleep_all_neg1 / total_rows) * 100 if total_rows else 0
    mixed_act_pct = (mixed_tot / total_rows) * 100 if total_rows else 0


    scalar_cols = ["heart_rate", "resp_rate", "stress", "blood_glucose", "oxygen_sat" if USE_OXYGEN_SATURATION else None, "steps_per_min", "kcal_per_min" if CHECK_KCAL else None]
    scalar_cols = [c for c in scalar_cols if c is not None]
    gap_cols = ["heart_rate", "resp_rate", "stress", "blood_glucose", "oxygen_sat" if USE_OXYGEN_SATURATION else None, "kcal_per_min" if CHECK_KCAL else None]
    gap_cols = [c for c in gap_cols if c is not None]
    ALL_SCALARS = ["heart_rate", "resp_rate", "stress", "blood_glucose", "oxygen_sat", "steps_per_min", "kcal_per_min"]
    ALL_GAPS = ["heart_rate", "resp_rate", "stress", "blood_glucose", "oxygen_sat", "kcal_per_min"]
 
    
    total_rows = day_counter * bins_per_day
    total_gap_cells = total_rows * len(gap_cols)        
    total_scalar_cells = total_rows * len(scalar_cols)
    # interpolated / nan / invalid by column 
    if total_rows: 
        pct_interp_col  = {c: (tot_interp[c]  / total_rows * 100) for c in gap_cols}
        pct_nan_col     = {c: (tot_nan[c]     / total_rows * 100) for c in scalar_cols}
        pct_invalid_col = {c: (tot_invalid[c] / total_rows * 100) for c in scalar_cols}
        if total_scalar_cells: 
            pct_interp_all  = sum(tot_interp[c]  for c in gap_cols) / total_gap_cells * 100
            pct_nan_all     = sum(tot_nan[c]     for c in scalar_cols) / total_scalar_cells * 100
            pct_invalid_all = sum(tot_invalid[c] for c in scalar_cols) / total_scalar_cells * 100
        else:
            pct_interp_all  = 0
            pct_nan_all     = 0
            pct_invalid_all = 0
        if not USE_OXYGEN_SATURATION: 
            pct_interp_col["oxygen_sat"] = 0 
            pct_nan_col["oxygen_sat"] = 0 
            pct_invalid_col["oxygen_sat"] = 0 
        if not CHECK_KCAL: 
            pct_interp_col["kcal_per_min"] = 0 
            pct_nan_col["kcal_per_min"] = 0 
            pct_invalid_col["kcal_per_min"] = 0 

    else: 
        pct_interp_col  = {c: 0 for c in ALL_SCALARS}
        pct_nan_col     = {c: 0 for c in ALL_SCALARS}
        pct_invalid_col = {c: 0 for c in ALL_SCALARS}
        pct_interp_all  = 0
        pct_nan_all     = 0
        pct_invalid_all = 0
    # CSV HEADER
    header_common = ["patient_id", "interval", "max_interpolate", "valid_interp", "window", "start_hour", "use_oxygen_sat", "interpolate_kcal", "check_kcal", "days_saved"]
    header_percol = (
        ["%_interpolated_scalar"] + 
        [f"%_{c}_interp"  for c in ALL_GAPS] +
        ["%_nan_scalar"] + 
        [f"%_nan_{c}"     for c in ALL_SCALARS] +
        ["%_invalid_scalar"] + 
        [f"%_invalid_{c}" for c in ALL_SCALARS] + 
        ["avg_gap_scalar"] +                   
        [f"avg_gap_{c}" for c in ALL_GAPS]
    )
    header_stats =  [ "%_no_sleep", "%_mixed_activity"]
    full_header = header_common + header_percol + header_stats 
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    #calculate gaps 
    gap_values = {} 
    if day_counter > 0 and all_gaps: 
        for c in gap_cols: 
            if tot_gaps[c]: 
                gap_values[c] = statistics.mean(tot_gaps[c])
        overall_gap_mean = statistics.mean(all_gaps)
        if not USE_OXYGEN_SATURATION: 
            gap_values["oxygen_sat"] = 0 
        if not CHECK_KCAL: 
            gap_values["kcal_per_min"] = 0 

    else: 
        for c in ALL_GAPS: 
            gap_values[c] = 0 
        overall_gap_mean = 0 
    #write 
    with csv_path.open("a", newline="") as csv_f:
        writer = csv.writer(csv_f)
        if write_header:
            writer.writerow(full_header)
        row_common = [
            pid,
            INTERVAL,
            MAX_INTERPOLATE_WINDOW.total_seconds() / 3600,
            VALID_INTERPOLATE,
            WINDOW.total_seconds() / 3600,
            START_HOUR,
            USE_OXYGEN_SATURATION,
            INTERPOLATE_CALORIES, 
            CHECK_KCAL, 
            day_counter,
        ]
        row_stats = (
            [f"{pct_interp_all:.3f}"] + 
            [f"{pct_interp_col[c]:.3f}" for c in ALL_GAPS] + 
            [f"{pct_nan_all:.3f}"] + 
            [f"{pct_nan_col[c]:.3f}"     for c in ALL_SCALARS] +
            [f"{pct_invalid_all:.3f}"] + 
            [f"{pct_invalid_col[c]:.3f}" for c in ALL_SCALARS] + 
            [f"{overall_gap_mean:.3f}"] + 
            [f"{gap_values[c]:.3f}" for c in ALL_GAPS] + 
            [f"{sleep_all_neg1_pct:.3f}"] + 
            [f"{mixed_act_pct:.3f}"]
        )
        writer.writerow(row_common + row_stats)
    try:
        df = pd.read_csv(csv_path)
        float_cols = df.select_dtypes(include="float").columns
        df[float_cols] = df[float_cols].applymap(lambda x: f"{x:.3f}")
        pretty_path = csv_path.with_suffix(".txt")
        pretty_path.write_text(df.to_string(index=False))
    except Exception as e:
        print(f"[WARN] pretty-format failed: {e}")


def init_config( *, interval: str = "5min", max_interpolate: str = "1h", valid_interpolate: bool = True, start_hour: int = 7, window: str = "24h", use_oxygen_sat: bool = False, interpolate_calories: bool = True, check_kcal: bool = True):
    global INTERVAL, MAX_INTERPOLATE_WINDOW, VALID_INTERPOLATE
    global START_HOUR, WINDOW, USE_OXYGEN_SATURATION, CAL_RATE_INVALID_MAX, INTERPOLATE_CALORIES, CHECK_KCAL
    global REQ_COLS, INVALID_VALS, SCALAR_COLS
    INTERVAL = interval
    MAX_INTERPOLATE_WINDOW = pd.Timedelta(max_interpolate)
    VALID_INTERPOLATE = valid_interpolate
    START_HOUR = start_hour
    WINDOW = pd.Timedelta(window)
    CAL_RATE_INVALID_MAX = 20 
    USE_OXYGEN_SATURATION = use_oxygen_sat
    INTERPOLATE_CALORIES = interpolate_calories
    CHECK_KCAL = check_kcal 
    REQ_COLS = ["heart_rate", "resp_rate", "stress", "blood_glucose", "steps_per_min", "kcal_per_min", "act_walking", "act_running", "act_sedentary", "act_generic","sleep_awake", "sleep_rem", "sleep_deep", "sleep_light"]
    if USE_OXYGEN_SATURATION:
        REQ_COLS.insert(2, "oxygen_sat") 
    SCALAR_COLS = [c for c in REQ_COLS if not (c.startswith("act_") or c.startswith("sleep_"))]
    INVALID_VALS = {
        "heart_rate": {-1, 0},
        "resp_rate": {-2, -1, 0},
        "oxygen_sat": {-1, 0},
        "blood_glucose": {-1, 0},
        "stress": {-2, -1},
        "steps_per_min": {-1},
        "kcal_per_min": {-1},
        "act_walking": {-1}, "act_running": {-1},
        "act_sedentary": {-1}, "act_generic": {-1},
        "sleep_light": {-1}, "sleep_deep": {-1},
        "sleep_rem": {-1}, "sleep_awake": {-1},
    }

def get_csv_all_patients(out_dir, PATIENT_IDS):
    all_patients_data = []
    for patient in PATIENT_IDS: 
        #print("hi")
        patient_dir = Path(f"{out_dir}/{patient}")
        patient_data = []
        for csv_file in patient_dir.glob("*.csv"):
            day_df = pd.read_csv(csv_file)
            day_df.insert(0, 'patient_id', patient)
            patient_data.append(day_df)
        if patient_data:
            patient_combined = pd.concat(patient_data, ignore_index=True)
            all_patients_data.append(patient_combined)
    if all_patients_data:

        final_df = pd.concat(all_patients_data, ignore_index=True)
        output_csv = Path(out_dir) / "all_patients.csv"
        final_df.to_csv(output_csv, index=False)
        txt_path = output_csv.with_suffix(".txt")
        txt_path.write_text(final_df.to_string(index=False, col_space=10, justify="right", float_format="%.3f", na_rep="NaN"))


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_PATH", type=str, help='location of the original AI-READI dataset')
    args = parser.parse_args()

    #all patients with everything! 
    PATIENT_IDS = pd.read_csv(f'{args.DATA_PATH}/participants.tsv', delimiter='\t')
    PATIENT_IDS = PATIENT_IDS[PATIENT_IDS[['cardiac_ecg', 'clinical_data', 'wearable_activity_monitor', 'wearable_blood_glucose']].all(axis=1)]['person_id'].values
    PATIENT_IDS = [x for x in PATIENT_IDS if x not in [1059, 1078, 1082, 1107, 1108, 1127, 1130, 1142, 1150, 1162, 1265, 1279, 1319, 1342, 1343, 1369, 1371, 1375, 1395, 1435, 1459, 1460, 1464, 1488, 1517, 1522, 1560, 1561, 1570, 1581, 1601, 1633, 1638, 1671, 1672, 1692, 1697, 1700, 1711, 1724, 1755, 1761, 1781, 1783, 1796, 4025, 4029, 4038, 4050, 4057, 4068, 4069, 4070, 4071, 4075, 4079, 4080, 4081, 4083, 4084, 4085, 4086, 4090, 4092, 4093, 4094, 4095, 4096, 4097, 4098, 4102, 4108, 4110, 4129, 4137, 4173, 4174, 4176, 4194, 4195, 4197, 4198, 4199, 4209, 4212, 4213, 4214, 4217, 4218, 4223, 4233, 4238, 4242, 4244, 4258, 4259, 4260, 4262, 4272, 4277, 4280, 4288, 4293, 4295, 4300, 4303, 4307, 4324, 4325, 4326, 4339, 4342, 4346, 4353, 4354, 4355, 4359, 4363, 4367, 4368, 4369, 4375, 4382, 4386, 4390, 4393, 4397, 4398, 4401, 4404, 4408, 4414, 4415, 4443, 4448, 4458, 4459, 4465, 4469, 4470, 4474, 4475, 4480, 4484, 4491, 4493, 4495, 4500, 4502, 4507, 4511, 4524, 4552, 4553, 4570, 4572, 4573, 4580, 4582, 4584, 4598, 4601, 4602, 4607, 4610, 4611, 4613, 4637, 4641, 4664, 4675, 4679, 4685, 7060, 7135, 7289, 7321, 7324, 7331, 7342, 7353, 7370, 7380, 7402, 7410, 7426, 7441, 7444, 7463, 7468, 7494, 7509, 7526, 7535, 7550, 7554, 7560, 7565, 7586, 7594, 7621, 7686, 7702]]


    init_config(interval="60min", max_interpolate = "3h", valid_interpolate = True, start_hour=8, use_oxygen_sat=False, interpolate_calories=True, check_kcal =False)
    error_dir = f"aligned_data/errors"
    os.makedirs(error_dir, exist_ok=True)
    with (Path(f"{error_dir}/errors.txt")).open("w") as f:
        f.write(f"Errors \n")

    for patient in PATIENT_IDS:  
        try: 
            align_patient(patient, args.DATA_PATH, PATIENT_IDS) 
        except Exception as e: 
            print(f"{patient},")
            #print(f"[{patient}] Error processing patient:", e)

        #    with (Path(f"{error_dir}/errors.txt")).open("a") as f:
        #        f.write(f"Error processing patient: {patient} under settings interval = 5 min, max_interpolate = {MAX_INTERPOLATE_WINDOW.total_seconds() // 3600}, valid_interpolate = {VALID_INTERPOLATE} start_hour = {START_HOUR}, use_oxygen_sat = {USE_OXYGEN_SATURATION}, interpolate_calories = True, check_kcal = False\n")
        #    continue 

    get_csv_all_patients("aligned_data/no_oxygensat/60min_interval_3hr_maxinterp_True_validinterp_8start", PATIENT_IDS)
    

if __name__ == "__main__":
    main()