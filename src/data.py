"""Lecture des fichiers bruts et construction de la table journalière (une ligne par athlète et par jour)."""
import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data" / "raw"))
PARTICIPANTS = ["p01", "p03", "p05"]
TZ = "Europe/Oslo"  # pmsys et sleep_score sont en UTC, Fitbit est déjà en heure locale
START, END = "2019-11-01", "2020-03-31"


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def utc_to_day(s):
    return pd.to_datetime(s, utc=True).dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()


def fitbit_minutes(p, name):
    """calories / steps / distance par minute -> total par jour."""
    df = pd.DataFrame(read_json(DATA_DIR / p / "fitbit" / f"{name}.json"))
    df["date"] = pd.to_datetime(df["dateTime"]).dt.normalize()
    return df.assign(value=pd.to_numeric(df["value"])).groupby("date")["value"].sum()


def heart_rate(p):
    """FC toutes les ~5 s -> moyenne du jour et minutes avec une mesure (temps de port)."""
    path = DATA_DIR / p / "fitbit" / "heart_rate.json"
    if not path.exists():  # p03 n'a pas ce fichier
        return pd.DataFrame(columns=["hr_mean", "hr_wear_minutes"])
    raw = read_json(path)
    df = pd.DataFrame({"ts": pd.to_datetime([r["dateTime"] for r in raw]),
                       "bpm": [r["value"]["bpm"] for r in raw]})
    df["date"], df["minute"] = df["ts"].dt.normalize(), df["ts"].dt.floor("min")
    return df.groupby("date").agg(hr_mean=("bpm", "mean"), hr_wear_minutes=("minute", "nunique"))


def resting_hr(p):
    raw = read_json(DATA_DIR / p / "fitbit" / "resting_heart_rate.json")
    s = pd.Series({pd.Timestamp(r["dateTime"]).normalize(): r["value"]["value"] for r in raw})
    return s.replace(0, np.nan).rename("resting_hr")  # 0 = non calculé par Fitbit


def hr_zones(p):
    raw = read_json(DATA_DIR / p / "fitbit" / "time_in_heart_rate_zones.json")
    df = pd.DataFrame([{"date": pd.Timestamp(r["dateTime"]).normalize(), **r["value"]["valuesInZones"]} for r in raw])
    df = df.rename(columns={"IN_DEFAULT_ZONE_1": "zone1_min", "IN_DEFAULT_ZONE_2": "zone2_min",
                            "IN_DEFAULT_ZONE_3": "zone3_min"})
    return df.groupby("date")[["zone1_min", "zone2_min", "zone3_min"]].last()


def sleep(p):
    """Nuit principale, rattachée au jour du réveil."""
    rows = []
    for s in read_json(DATA_DIR / p / "fitbit" / "sleep.json"):
        if not s.get("mainSleep", True):
            continue
        summ = s["levels"]["summary"]
        rows.append({"date": pd.Timestamp(s["dateOfSleep"]), "sleep_minutes": s["minutesAsleep"],
                     "sleep_efficiency": s["efficiency"],
                     "sleep_deep_min": summ["deep"]["minutes"] if "deep" in summ else np.nan})
    return pd.DataFrame(rows).groupby("date").last()


def sleep_score(p):
    df = pd.read_csv(DATA_DIR / p / "fitbit" / "sleep_score.csv")
    df["date"] = utc_to_day(df["timestamp"])
    return df.sort_values("timestamp").groupby("date")["overall_score"].last().rename("sleep_score")


def wellness(p):
    df = pd.read_csv(DATA_DIR / p / "pmsys" / "wellness.csv")
    df["date"] = utc_to_day(df["effective_time_frame"])
    cols = ["fatigue", "mood", "readiness", "sleep_quality", "soreness", "stress"]
    df = df.sort_values("effective_time_frame").groupby("date")[cols].last()  # dernière réponse du jour
    return df.add_prefix("well_")


def srpe(p):
    df = pd.read_csv(DATA_DIR / p / "pmsys" / "srpe.csv")
    df["date"] = utc_to_day(df["end_date_time"])
    df["srpe_load"] = df["perceived_exertion"] * df["duration_min"]  # Foster 2001
    return df.groupby("date")["srpe_load"].sum()


def injury(p):
    df = pd.read_csv(DATA_DIR / p / "pmsys" / "injury.csv")
    df["date"] = utc_to_day(df["effective_time_frame"])
    df["injury"] = df["injuries"].apply(lambda s: int(len(ast.literal_eval(s)) > 0))
    return df.groupby("date")["injury"].max()


def reporting(p):
    # la colonne "timestamp" a un format incohérent (jour et mois inversés) : on garde "date"
    df = pd.read_csv(DATA_DIR / p / "googledocs" / "reporting.csv")
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df["n_meals"] = df["meals"].fillna("").str.count(",") + df["meals"].notna()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["alcohol"] = df["alcohol_consumed"].str.strip().str.lower().eq("yes").astype(int)
    return df.groupby("date")[["n_meals", "weight", "alcohol"]].last()


def participant_daily(p):
    days = pd.DataFrame(index=pd.date_range(START, END, name="date"))
    parts = [fitbit_minutes(p, "steps").rename("steps"), fitbit_minutes(p, "calories").rename("calories"),
             (fitbit_minutes(p, "distance") / 1e5).rename("distance_km"),  # cm -> km
             heart_rate(p), resting_hr(p), hr_zones(p), sleep(p), sleep_score(p), wellness(p),
             srpe(p), injury(p), reporting(p)]
    for part in parts:
        if len(part):  # p03 n'a pas de FC détaillée
            days = days.join(part, how="left")
    return days.reset_index().assign(participant=p)


def add_load(df):
    """Charge interne et indicateurs dérivés, par athlète.

    TRIMP d'Edwards (1993) adapté aux 3 zones Fitbit (poids 1,5 / 3,5 / 5), ACWR 7 j / 28 j
    (Gabbett 2016), monotonie de Foster (1998), écart de la FC de repos à la moyenne des 28 jours précédents.
    """
    df = df.sort_values(["participant", "date"]).copy()
    df["trimp"] = 1.5 * df["zone1_min"] + 3.5 * df["zone2_min"] + 5 * df["zone3_min"]
    g = df.groupby("participant")["trimp"]
    df["load_7d"] = g.transform(lambda s: s.rolling(7, min_periods=4).mean())
    df["load_28d"] = g.transform(lambda s: s.rolling(28, min_periods=14).mean())
    df["acwr"] = df["load_7d"] / df["load_28d"].replace(0, np.nan)
    df["monotony"] = df["load_7d"] / g.transform(lambda s: s.rolling(7, min_periods=4).std()).replace(0, np.nan)
    base = df.groupby("participant")["resting_hr"].transform(lambda s: s.shift(1).rolling(28, min_periods=7).mean())
    df["resting_hr_delta"] = df["resting_hr"] - base
    return df


def build_daily():
    df = pd.concat([participant_daily(p) for p in PARTICIPANTS], ignore_index=True)
    age = pd.read_excel(DATA_DIR / "participant-overview.xlsx", header=1).set_index("Participant ID")["Age"]
    df["age"] = df["participant"].map(age)
    return add_load(df)
