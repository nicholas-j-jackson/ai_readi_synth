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

#all patients with everything! 
plt.style.use('default')
sns.set_palette("husl")
pd.set_option("display.max_columns", None)   
pd.set_option("display.max_rows",    None)   
pd.set_option("display.width",       None)   

START_HOUR = None 
WINDOW = None
USE_OXYGEN_SATURATION = None
CALCULATE_OXYGEN_SATURATION = None 
USE_KCAL = None 
MAX_GAP = None 
INVALID_VALS = None 

ACTIVITY_MAP = {
    "walking": "act_walking_hrs",
    "running": "act_running_hrs",
    "sedentary": "act_sedentary_hrs",
    "generic": "act_generic_hrs",
    "": np.nan,
    None: np.nan,
}
SLEEP_MAP = {
    "light": "sleep_light_hrs",
    "deep": "sleep_deep_hrs",
    "rem": "sleep_rem_hrs",
    "awake": "sleep_awake_hrs",
    "": np.nan,
    None: np.nan,
}
#---------------------- Data loading functions ------------------------------- # 
def get_bloodglucose(path): 
    try:
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
    except Exception as e:
        return pd.DataFrame()

def get_heartrate(path): 
    try:
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
    except Exception as e:
        return pd.DataFrame()

def get_oxygensat(path): 
    try:
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
    except Exception as e:
        return pd.DataFrame()

def get_activity(path): 
    try:
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
    except Exception as e:
        return pd.DataFrame()

def get_calorie(path): 
    try:
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
    except Exception as e:
        return pd.DataFrame()

def get_respiratoryrate(path): 
    try:
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
    except Exception as e:
        return pd.DataFrame()

def get_sleep(path): 
    try:
        path = Path(path)
        with path.open() as f: 
            data = json.load(f) 
        start_times = [] 
        end_times = [] 
        values = [] 
        for rec in data["body"]["sleep"]: 
            start_time = (rec["sleep_stage_time_frame"]["time_interval"]["start_date_time"])
            end_time = (rec["sleep_stage_time_frame"]["time_interval"]["end_date_time"])
            v = rec["sleep_stage_state"]
            start_times.append(start_time)
            end_times.append(end_time)
            values.append(v) 
        return pd.DataFrame({"start_time": start_times, "end_time": end_times, "sleep_stage": values}).sort_values(by='start_time')
    except Exception as e:
        return pd.DataFrame()

def get_stress(path): 
    try:
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
    except Exception as e:
        return pd.DataFrame()

def get_patient_data(patient_id): 
    root = Path("~/Desktop/synth_data/dataset").expanduser()
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


def has_large_gap(ts: pd.Series):
    if ts.empty:
        return True
    ts = pd.to_datetime(ts).sort_values()
    return (ts.diff().dropna() >= MAX_GAP).any()

def activity_gap_ok(act_df: pd.DataFrame, s: pd.Timestamp, e: pd.Timestamp):
    if act_df.empty:
        return False
    act_intervals = act_df.copy()
    act_intervals["start"] = pd.to_datetime(act_intervals["start_time"], utc=True)
    act_intervals["end"]   = pd.to_datetime(act_intervals["end_time"],   utc=True)
    act_intervals = act_intervals[(act_intervals["end"] > s) & (act_intervals["start"] < e)].sort_values("start")
    if act_intervals.empty:
        return False
    # calculate gap between consecutive intervals
    gaps = (act_intervals["start"].iloc[1:].reset_index(drop=True) - act_intervals["end"].iloc[:-1].reset_index(drop=True))
    return not (gaps >= MAX_GAP).any()

def valid_times(df, ts_col: str, val_col: str, invalid_key: str):
    bad = INVALID_VALS[invalid_key]
    ok  = (~df[val_col].isin(bad)) & (~df[val_col].isna())
    return pd.to_datetime(df.loc[ok, ts_col], utc=True)

def interval_total(interval_df: pd.DataFrame, s: pd.Timestamp, e: pd.Timestamp, name_map: dict):
    #make col totals 
    out = {v: 0.0 for v in name_map.values() if pd.notna(v)}
    if interval_df.empty:
        return out
    df = interval_df.copy()
    df["start"] = pd.to_datetime(df["start_time"], utc=True)
    df["end"]   = pd.to_datetime(df["end_time"], utc=True)
    df = df[(df["end"] > s) & (df["start"] < e)]
    for _, row in df.iterrows():
        key = "activity_name" if "activity_name" in row else "sleep_stage"
        label = name_map.get(row[key], np.nan)
        if pd.isna(label):
            continue
        overlap = max(
            0, (min(row["end"], e) - max(row["start"], s)).total_seconds())
        out[label] += overlap / 3600.0
    return out

def total_steps(act_df: pd.DataFrame, s: pd.Timestamp, e: pd.Timestamp):
    if act_df.empty:
        return np.nan
    subset = act_df.copy()
    subset["start"] = pd.to_datetime(subset["start_time"])
    subset = subset[(subset["start"] >= s) & (subset["start"] < e) & (subset["activity_units"] == "steps")]
    return subset["activity_value"].dropna().sum()

def resting_hr(hr_df: pd.DataFrame, act_df: pd.DataFrame, s: pd.Timestamp, e: pd.Timestamp) -> float:
    sed = act_df[act_df["activity_name"] == "sedentary"]
    if sed.empty:
        return np.nan
    sed_intervals = pd.IntervalIndex.from_arrays(pd.to_datetime(sed["start_time"]), pd.to_datetime(sed["end_time"]), closed="left")
    hr_subset = hr_df.copy()
    hr_subset["t"] = pd.to_datetime(hr_subset["date_time"])
    hr_subset = hr_subset[(hr_subset["t"] >= s) & (hr_subset["t"] < e)]
    if hr_subset.empty:
        return np.nan
    # returns -1 for points outside every interval
    in_sed = sed_intervals.get_indexer(hr_subset["t"]) >= 0
    vals = hr_subset.loc[in_sed, "heart_rate"].replace(list(INVALID_VALS["heart_rate"]), np.nan).dropna()
    return vals.mean() if not vals.empty else np.nan

def calories_total(cal_df: pd.DataFrame, act_df: pd.DataFrame,
                   s: pd.Timestamp, e: pd.Timestamp) -> float:
    if cal_df.empty:
        return np.nan
    # activity interval set up 
    act_subset = act_df.copy()
    act_subset["start"] = pd.to_datetime(act_subset["start_time"], utc=True)
    act_subset["end"]   = pd.to_datetime(act_subset["end_time"],   utc=True)
    # calories inside the window 
    cal = cal_df.copy()
    cal["t"] = pd.to_datetime(cal["date_time"], utc=True)
    cal = cal[(cal["t"] >= s) & (cal["t"] < e)]
    if cal.empty:
        return np.nan
    # map kcals to activity label s
    labels = []
    for t in cal["t"]:
        row = act_subset[(act_subset["start"] <= t) & (act_subset["end"] > t)]
        # no matching interval → drop later
        if row.empty:             
            labels.append(np.nan)
        else:
            labels.append(row["activity_name"].iloc[0])
    cal["label"] = labels
    # keep only samples that found a label
    cal = cal.dropna(subset=["label"])
    if cal.empty:
        return np.nan
    total = 0.0
    for label, group in cal.groupby("label"):
        group = group.sort_values("t")
        last = None
        for i, v in enumerate(group["calories"]):
            if last is None:
                last = v
                continue
            # monotone up
            if v >= last:               
                last = v
            # downward spike - glitch or reset? 
            else:                       
                nxt = group["calories"].iloc[i + 1] if i + 1 < len(group) else None
                # glitch ⇒ ignore
                if nxt is not None and nxt >= last:  
                    continue
                # real reset
                total += last            
                last = v
        total += last if last is not None else 0.0
    return total

def get_scalar_data_from_raw_data(patient_id: int):
    pid, dfs = get_patient_data(patient_id)
    out_rows, bad_days = [], []
    # use heart-rate span to anchor calendar
    if dfs["heart_rate"].empty:
        warnings.warn(f"No heart-rate data for patient {pid}; nothing written.")
        return
    all_ts = pd.to_datetime(dfs["heart_rate"]["date_time"])
    t0, t1 = all_ts.min(), all_ts.max()
    # anchor at the first full START_HOUR boundary that is *after* the first sample
    t0_floor = t0.normalize() + pd.Timedelta(hours=START_HOUR)
    if t0 >= t0_floor:
        anchor = t0_floor + pd.Timedelta(days=1)   # skip same-day partial window
    else:
        anchor = t0_floor                          # earliest sample is before START_HOUR
    day_start = anchor
    while day_start + WINDOW <= t1:
        day_end = day_start + WINDOW
        # check for gaps 
        bad = False
        checks = {"heart_rate": valid_times(dfs["heart_rate"], "date_time", "heart_rate", "heart_rate"),
            "resp_rate": valid_times(dfs["resp_rate"], "date_time", "respiratory_rate", "respiratory_rate"),
            "stress": valid_times(dfs["stress"], "date_time", "stress", "stress"),
            "blood_glucose": valid_times(dfs["blood_glucose"], "start_time", "blood_glucose", "blood_glucose"),
        }
        for name, series in checks.items():
            if has_large_gap(series[(pd.to_datetime(series) >= day_start) & (pd.to_datetime(series) < day_end)]):
                bad = True
                break
        if not bad and not activity_gap_ok(dfs["activity"], day_start, day_end):
            bad = True
        if bad:
            bad_days.append(day_start.isoformat(timespec="seconds"))
            day_start += pd.Timedelta(days=1)
            continue
        # scalar calculations 
        row = {"patient_id": pid, "day": day_start.isoformat(timespec="seconds")}
        for key, col, tcol in [("heart_rate", "heart_rate", "date_time"), ("resp_rate", "respiratory_rate", "date_time"), ("stress", "stress", "date_time"), ("blood_glucose", "blood_glucose", "start_time")]:  
            df = dfs["heart_rate"] if col == "heart_rate" else \
                dfs["resp_rate"] if col == "respiratory_rate" else \
                dfs["stress"] if col == "stress" else \
                dfs["blood_glucose"]
            sub = df.copy()
            sub["t"] = pd.to_datetime(sub[tcol], utc=True)
            sub = sub[(sub["t"] >= day_start) & (sub["t"] < day_end)]
            vals = sub[col].replace(list(INVALID_VALS[col]), np.nan).dropna()
            for stat, func in [("min", np.min), ("median", np.median), ("mean", np.mean), ("max", np.max), ("std",    np.std)]:
                row[f"{key}_{stat}"] = func(vals) if not vals.empty else np.nan
        # oxygen saturation
        if USE_OXYGEN_SATURATION:
            sub = dfs["oxygen_sat"].copy()
            sub["t"] = pd.to_datetime(sub["date_time"])
            sub = sub[(sub["t"] >= day_start) & (sub["t"] < day_end)]
            vals = sub["oxygen_saturation"].replace(
                list(INVALID_VALS["oxygen_sat"]), np.nan).dropna()
            for stat, func in [("min", np.min), ("median", np.median), ("mean", np.mean), ("max", np.max), ("std",    np.std)]:
                row[f"oxygen_sat_{stat}"] = func(vals) if not vals.empty else np.nan
        # activity hours 
        row.update(interval_total(dfs["activity"], day_start, day_end, ACTIVITY_MAP))
        # activity event breakdown 
        sub = dfs["activity"].copy()
        sub["start"] = pd.to_datetime(sub["start_time"], utc=True)
        sub["end"] = pd.to_datetime(sub["end_time"], utc=True)
        sub = sub[(sub["end"] > day_start) & (sub["start"] < day_end)]
        sub = sub.sort_values("start")
        # only keep known activity names
        valid_activities = ["generic", "walking", "running", "sedentary"]
        sub = sub[sub["activity_name"].isin(valid_activities)]
        # detect transitions between intervals
        sub["prev_activity"] = sub["activity_name"].shift(1)
        sub["new_event"] = sub["activity_name"] != sub["prev_activity"]
        # count total events per activity type
        event_counts = sub[sub["new_event"]].groupby("activity_name").size().to_dict()
        # store in row
        for activity in valid_activities:
            row[f"act_{activity}_total_events"] = event_counts.get(activity, 0)
        #activity total active hours 
        row["total_active_hrs"] = sum(v for k, v in row.items() if (k.startswith("act_") and k.endswith("_hrs")) and ("sedentary" not in k ))
        # resting heart-rate
        row["resting_heart_rate"] = resting_hr(dfs["heart_rate"], dfs["activity"], day_start, day_end)
        # step statistics
        row["total_steps"] = total_steps(dfs["activity"], day_start, day_end)
        # calories
        if USE_KCAL:
            row["total_kcal"] = calories_total(dfs["calorie"], dfs["activity"], day_start, day_end)
        #sleep hours 
        row.update(interval_total(dfs["sleep"], day_start, day_end, SLEEP_MAP))
        #sleep events 
        sub = dfs["sleep"].copy()
        sub["start"] = pd.to_datetime(sub["start_time"], utc=True)
        sub["end"] = pd.to_datetime(sub["end_time"], utc=True)
        sub = sub[(sub["end"] > day_start) & (sub["start"] < day_end)]
        sub = sub.sort_values("start")
        valid_sleep_stages = ["light", "rem", "deep", "awake"]
        sub = sub[sub["sleep_stage"].isin(valid_sleep_stages)]
        sub["prev_stage"] = sub["sleep_stage"].shift(1)
        sub["new_event"] = sub["sleep_stage"] != sub["prev_stage"]
        event_counts = sub[sub["new_event"]].groupby("sleep_stage").size().to_dict()
        for stage in valid_sleep_stages:
            row[f"sleep_{stage}_total_events"] = event_counts.get(stage, 0)
        #sleep totals 
        row["total_sleep_monitor_hrs"]    = sum(v for k, v in row.items() if k in ["sleep_light_hrs", "sleep_rem_hrs", "sleep_awake_hrs", "sleep_deep_hrs"])
        if row["total_sleep_monitor_hrs"] > 0: 
            row["percent_of_sleep_sleeping"] = sum(v for k, v in row.items() if k in ["sleep_light_hrs", "sleep_rem_hrs", "sleep_deep_hrs"]) / row["total_sleep_monitor_hrs"]
        else: 
            row["percent_of_sleep_sleeping"] = 0 
        out_rows.append(row)
        day_start += pd.Timedelta(days=1)
    #write to file 
    out_dir = f"scalar_data/scalar_from_raw/{START_HOUR}_{WINDOW.total_seconds() / 3600}h_{MAX_GAP.total_seconds() / 3600}h_{USE_OXYGEN_SATURATION}_{USE_KCAL}/{pid}"
    os.makedirs(out_dir, exist_ok=True)
    if out_rows:
        df_out = pd.DataFrame(out_rows)
        csv_path = Path(f"{out_dir}/{pid}.csv")
        df_out.to_csv(csv_path, index=False)
        txt_path = csv_path.with_suffix(".txt")
        txt_path.write_text(df_out.to_string(index=False, col_space=10, justify="right", float_format="%.3f", na_rep="NaN"))
    if bad_days:
        fail_path = Path(f"{out_dir}/{pid}_failed.txt")
        with fail_path.open("a") as f:
            for d in bad_days:
                f.write(f"{d}\n")

def get_scalar_data_from_aligned_data(patient): 
    if USE_OXYGEN_SATURATION: 
        patient_dir = Path(f"aligned_data/with_oxygensat/60min_interval_{int(MAX_GAP.total_seconds()//3600)}hr_maxinterp_True_validinterp_{START_HOUR}start/{patient}")
    else: 
        patient_dir = Path(f"aligned_data/no_oxygensat/60min_interval_{int(MAX_GAP.total_seconds()//3600)}hr_maxinterp_True_validinterp_{START_HOUR}start/{patient}")
    all_daily_data = []
    #loop through days 
    for csv_file in patient_dir.glob("*.csv"):
        day_df = pd.read_csv(csv_file)
        scalar_values = {} 
        scalar_values["patient_id"] = patient
        scalar_values["day"] = day_df["timestamp"][0]

        #heart rate, resp rate, stress, blood glucose statistics
        df = day_df.copy() 
        for key in ["heart_rate", "resp_rate", "stress", "blood_glucose"]: 
            vals = df[key].replace(list(INVALID_VALS[key]), np.nan).dropna()
            for stat, func in [("min", np.min), ("median", np.median), ("mean", np.mean), ("max", np.max), ("std", np.std)]:
                scalar_values[f"{key}_{stat}"] = round(func(vals), 2) if not vals.empty else np.nan
        
        #oxygen saturation - to do 
        if USE_OXYGEN_SATURATION: 
            df = day_df.copy() 
            vals = df["oxygen_sat"].replace(list(INVALID_VALS["oxygen_sat"]), np.nan).dropna()
            for stat, func in [("min", np.min), ("median", np.median), ("mean", np.mean), ("max", np.max), ("std", np.std)]:
                scalar_values[f"oxygen_sat_{stat}"] = round(func(vals), 2) if not vals.empty else np.nan

        #activity breakdowns 
        df = day_df.copy()
        total_active_hours = 0 
        events = {} 
        for key in ["act_generic", "act_walking", "act_running", "act_sedentary"]: 
            vals = df[key].replace(list(INVALID_VALS[key]), np.nan).dropna()
            count_ones = (vals == 1).sum()
            scalar_values[f"{key}_hrs"] = round(count_ones * 5 / 60, 2)
            events[f"{key}_total_events"] = ((vals == 1) & (vals.shift(1) != 1)).sum()
            if "sedentary" not in key: 
                total_active_hours += round(count_ones * 5 / 60, 2)
        for key in events: 
            scalar_values[key] = events[key]
        scalar_values["total_active_hrs"] = total_active_hours 
        #resting heart rate 
        df = day_df.copy() 
        scalar_values["resting_heart_rate"] = round(df[df['act_sedentary'] == 1]['heart_rate'].replace(list(INVALID_VALS['heart_rate']), np.nan).dropna().mean(), 2)

        #total steps 
        df = day_df.copy() 
        vals = df["steps_per_min"].replace(list(INVALID_VALS["steps_per_min"]), np.nan).dropna()
        scalar_values["total_steps"] =  vals.sum() * 5 

        #total kcal 
        if USE_KCAL: 
            df = day_df.copy() 
            vals = df["kcal_per_min"].replace(list(INVALID_VALS["kcal_per_min"]), np.nan).dropna()
            scalar_values["total_kcal"] =  vals.sum() * 5 

        #sleep breakdown 
        df = day_df.copy()
        total_actually_asleep_hrs = 0 
        total_sleep_hours = 0 
        sleep_events= {} 
        for key in ["sleep_light", "sleep_deep", "sleep_rem", "sleep_awake"]: 
            vals = df[key].replace(list(INVALID_VALS[key]), np.nan).dropna()
            count_ones = (vals == 1).sum()
            scalar_values[f"{key}_hrs"] = round(count_ones * 5 / 60, 2)
            total_sleep_hours += round(count_ones * 5 / 60, 2)
            sleep_events[f"{key}_total_events"] = ((vals == 1) & (vals.shift(1) != 1)).sum()
            if "awake" not in key: 
                total_actually_asleep_hrs += round(count_ones * 5 / 60, 2)
        for key in sleep_events: 
            scalar_values[key] = sleep_events[key]
        scalar_values["total_sleep_monitor_hrs"] = total_sleep_hours 
        if total_sleep_hours > 0: 
            scalar_values["percent_of_sleep_sleeping"] = round(total_actually_asleep_hrs / total_sleep_hours, 2)
        else: 
            scalar_values["percent_of_sleep_sleeping"] = 0 

        
        all_daily_data.append(scalar_values)
    
    #save 
    out_dir = f"scalar_data/scalar_from_aligned/{START_HOUR}_{WINDOW.total_seconds() / 3600}h_{MAX_GAP.total_seconds() / 3600}h_{USE_OXYGEN_SATURATION}_{USE_KCAL}/{patient}"
    os.makedirs(out_dir, exist_ok=True)
    if all_daily_data:  
        daily_df = pd.DataFrame(all_daily_data)
        cols = ['patient_id', 'day'] + [col for col in daily_df.columns if col not in ['patient_id', 'day']]
        daily_df = daily_df[cols].sort_values(by=["patient_id", "day"])
        output_path = Path(f"{out_dir}/{patient}.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(output_path, index=False)
        txt_path = output_path.with_suffix(".txt")
        txt_path.write_text(daily_df.to_string(index=False, col_space=10, justify="right", float_format="%.3f", na_rep="NaN"))

        
def init_config( *, start_hour: int = 7, window: str = "24h", max_gap = "3h", use_oxygen_saturation: bool = True, use_kcal: bool = True):
    global START_HOUR, WINDOW, USE_OXYGEN_SATURATION, USE_KCAL, MAX_GAP
    global REQ_COLS, INVALID_VALS, SCALAR_COLS

    START_HOUR = start_hour 
    WINDOW = pd.Timedelta(window)
    USE_OXYGEN_SATURATION = use_oxygen_saturation
    USE_KCAL = use_kcal 
    MAX_GAP = pd.Timedelta(max_gap)
    CALCULATE_OXYGEN_SATURATION = True 

    INVALID_VALS = {
        "heart_rate": {np.nan, -1, 0},
        "respiratory_rate": {np.nan, -2, -1, 0}, 
        "resp_rate": {np.nan, -2, -1, 0},
        "oxygen_sat": {np.nan, -1, 0},
        "blood_glucose": {np.nan, -1, 0},
        "stress": {np.nan, -2, -1},
        "steps": {np.nan, -1},
        "steps_per_min": {np.nan, -1}, 
        "kcal": {np.nan, -1},
        "kcal_per_min": {np.nan, -1},
        "act_walking": {np.nan, -1}, "act_running": {np.nan, -1},
        "act_sedentary": {np.nan, -1}, "act_generic": {np.nan, -1},
        "sleep_light": {np.nan, -1}, "sleep_deep": {np.nan, -1},
        "sleep_rem": {np.nan, -1}, "sleep_awake": {np.nan, -1},
    }

def read_and_combine_csvs(patients, if_from_raw, start_hour, window, max_gap, use_oxygen_saturation, use_kcal): 
    all_patients_df = [] 
    for patient in patients: 
        if if_from_raw: 
            patient_path = Path(f"scalar_data/scalar_from_raw/{start_hour}_{window}_{max_gap}_{use_oxygen_saturation}_{use_kcal}/{patient}/{patient}.csv")
            combined_path = Path(f"scalar_data/all_patients/scalar_from_raw/all_patients_{start_hour}_{window}_{max_gap}_{use_oxygen_saturation}_{use_kcal}.csv")
            os.makedirs("scalar_data/all_patients/scalar_from_raw", exist_ok = True)
        else: 
            patient_path = Path(f"scalar_data/scalar_from_aligned/{start_hour}_{window}_{max_gap}_{use_oxygen_saturation}_{use_kcal}/{patient}/{patient}.csv")
            combined_path = Path(f"scalar_data/all_patients/scalar_from_aligned/all_patients_{start_hour}_{window}_{max_gap}_{use_oxygen_saturation}_{use_kcal}.csv")
            os.makedirs("scalar_data/all_patients/scalar_from_aligned", exist_ok = True)
        if os.path.exists(patient_path): 
            df = pd.read_csv(patient_path)
            all_patients_df.append(df) 
    if all_patients_df: 
        combined_df = pd.concat(all_patients_df)
        combined_df.to_csv(combined_path)
        txt_path = combined_path.with_suffix(".txt")
        txt_path.write_text(combined_df.to_string(index=False, col_space=10, justify="right", float_format="%.3f", na_rep="NaN"))
        return combined_df
    else: 
        with (Path(f"scalar_data/all_patients/{start_hour}_{window}_{max_gap}_{use_oxygen_saturation}_{use_kcal}_errors.txt")).open("a") as f:
            if if_from_raw: 
                f.write(f"No RAW csv files for settings: if_raw = {if_from_raw}, start_hour: {start_hour}, window: {window}, max_gap: {max_gap}, use_oxygensat: {use_oxygen_saturation}, use_kcal: {use_kcal}\n")
            else: 
                f.write(f"No aligned csv files for settings: if_raw = {if_from_raw}, start_hour: {start_hour}, window: {window}, max_gap: {max_gap}, use_oxygensat: {use_oxygen_saturation}, use_kcal: {use_kcal}\n")



def main(): 

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_PATH", type=str, help='location of the original AI-READI dataset')
    args = parser.parse_args()

    PATIENT_IDS = pd.read_csv(f'{args.DATA_PATH}/participants.tsv', delimiter='\t')
    PATIENT_IDS = PATIENT_IDS[PATIENT_IDS[['cardiac_ecg', 'clinical_data', 'wearable_activity_monitor', 'wearable_blood_glucose']].all(axis=1)]['person_id'].values
    PATIENT_IDS = [x for x in PATIENT_IDS if x not in [1059, 1078, 1082, 1107, 1108, 1127, 1130, 1142, 1150, 1162, 1265, 1279, 1319, 1342, 1343, 1369, 1371, 1375, 1395, 1435, 1459, 1460, 1464, 1488, 1517, 1522, 1560, 1561, 1570, 1581, 1601, 1633, 1638, 1671, 1672, 1692, 1697, 1700, 1711, 1724, 1755, 1761, 1781, 1783, 1796, 4025, 4029, 4038, 4050, 4057, 4068, 4069, 4070, 4071, 4075, 4079, 4080, 4081, 4083, 4084, 4085, 4086, 4090, 4092, 4093, 4094, 4095, 4096, 4097, 4098, 4102, 4108, 4110, 4129, 4137, 4173, 4174, 4176, 4194, 4195, 4197, 4198, 4199, 4209, 4212, 4213, 4214, 4217, 4218, 4223, 4233, 4238, 4242, 4244, 4258, 4259, 4260, 4262, 4272, 4277, 4280, 4288, 4293, 4295, 4300, 4303, 4307, 4324, 4325, 4326, 4339, 4342, 4346, 4353, 4354, 4355, 4359, 4363, 4367, 4368, 4369, 4375, 4382, 4386, 4390, 4393, 4397, 4398, 4401, 4404, 4408, 4414, 4415, 4443, 4448, 4458, 4459, 4465, 4469, 4470, 4474, 4475, 4480, 4484, 4491, 4493, 4495, 4500, 4502, 4507, 4511, 4524, 4552, 4553, 4570, 4572, 4573, 4580, 4582, 4584, 4598, 4601, 4602, 4607, 4610, 4611, 4613, 4637, 4641, 4664, 4675, 4679, 4685, 7060, 7135, 7289, 7321, 7324, 7331, 7342, 7353, 7370, 7380, 7402, 7410, 7426, 7441, 7444, 7463, 7468, 7494, 7509, 7526, 7535, 7550, 7554, 7560, 7565, 7586, 7594, 7621, 7686, 7702]]


    init_config(start_hour=8, window="24h", max_gap="3h", use_oxygen_saturation=False, use_kcal=True)
    for patient in PATIENT_IDS: 
        get_scalar_data_from_aligned_data(patient)
        #get_scalar_data_from_raw_data(patient) 

    init_config(start_hour=8, window="24h", max_gap="3h", use_oxygen_saturation=True, use_kcal=True)
    os.makedirs("scalar_data/all_patients", exist_ok=True)
    with (Path(f"scalar_data/all_patients/{START_HOUR}_{WINDOW.total_seconds() // 3600}h_{MAX_GAP.total_seconds() // 3600}_{USE_OXYGEN_SATURATION}_{USE_KCAL}_errors.txt")).open("w") as f:
        f.write("Errors\n")
    combined_df_raw = read_and_combine_csvs(PATIENT_IDS, True, 8, "24.0h", "3.0h", False, True)
    combined_df_aligned = read_and_combine_csvs(PATIENT_IDS, False, 8, "24.0h", "3.0h", False, True)


if __name__ == "__main__":
    main()