"""Nettoyage : codes invalides, bornes, jours sans montre, valeurs extrêmes, valeurs manquantes."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer

LIKERT = ["well_fatigue", "well_mood", "well_sleep_quality", "well_soreness", "well_stress"]
BOUNDS = {"resting_hr": (30, 120), "hr_mean": (30, 200), "steps": (0, 60000),
          "sleep_minutes": (0, 16 * 60), "weight": (30, 200), "well_readiness": (1, 10),
          **{c: (1, 5) for c in LIKERT}}
WATCH_COLS = ["steps", "distance_km", "calories", "zone1_min", "zone2_min", "zone3_min", "hr_mean", "trimp"]


def clean(daily):
    df = daily.copy()
    # 0 n'existe pas sur les échelles 1-5 ; readiness = 0 correspond à un curseur jamais déplacé (p05)
    df[LIKERT + ["well_readiness"]] = df[LIKERT + ["well_readiness"]].replace(0, np.nan)
    for col, (lo, hi) in BOUNDS.items():
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    # jour sans montre (moins de 10 h de FC, ou sans FC moins de 200 pas) : on ne le compte pas comme un repos
    no_watch = (df["hr_wear_minutes"] < 600) | (df["hr_wear_minutes"].isna() & (df["steps"].fillna(0) < 200))
    df["no_watch"] = no_watch.astype(int)
    df.loc[no_watch, WATCH_COLS] = np.nan

    # valeurs extrêmes : z-score robuste par athlète (Iglewicz & Hoaglin 1993), signalées mais gardées
    for col in ["steps", "trimp", "sleep_minutes"]:
        med = df.groupby("participant")[col].transform("median")
        mad = df.groupby("participant")[col].transform(lambda s: (s - s.median()).abs().median())
        df[f"{col}_outlier"] = (0.6745 * (df[col] - med) / mad.replace(0, np.nan)).abs().gt(3.5).astype(int)
    return df


def compare_imputers(df, cols, frac=0.1, seed=42):
    """On cache 10 % des valeurs connues, on les impute et on mesure l'erreur (RMSE / écart-type)."""
    rng = np.random.default_rng(seed)
    X = df[cols].astype(float)
    known = np.argwhere(X.notna().to_numpy())
    hide = known[rng.choice(len(known), int(frac * len(known)), replace=False)]
    masked = X.copy()
    masked.values[hide[:, 0], hide[:, 1]] = np.nan
    by_p = masked.groupby(df["participant"])
    mu, sd = masked.mean(), masked.std()
    methods = {
        "moyenne": lambda: masked.fillna(mu).to_numpy(),
        "médiane de l'athlète": lambda: by_p.transform(lambda s: s.fillna(s.median())).to_numpy(),
        "interpolation": lambda: by_p.transform(lambda s: s.interpolate(limit_direction="both")).to_numpy(),
        "KNN": lambda: KNNImputer(n_neighbors=5).fit_transform((masked - mu) / sd) * sd.to_numpy() + mu.to_numpy(),
        "MICE": lambda: IterativeImputer(RandomForestRegressor(n_estimators=30, random_state=seed),
                                         max_iter=5, random_state=seed).fit_transform(masked),
    }
    truth = X.values[hide[:, 0], hide[:, 1]]
    scale = X.std().to_numpy()[hide[:, 1]]
    res = {}
    for name, f in methods.items():
        est = np.asarray(f())[hide[:, 0], hide[:, 1]]
        ok = ~np.isnan(est)
        res[name] = np.sqrt(np.mean(((est[ok] - truth[ok]) / scale[ok]) ** 2))
    return pd.Series(res).sort_values().rename("erreur")


def impute(df, seed=42):
    """Version imputée exportée dans test_dataset.csv (MICE, la meilleure méthode de la comparaison).

    Pas d'imputation pour les blessures, la sRPE, les calories des photos et l'indice (la cible),
    ni pour une variable presque toujours absente chez un athlète (ex. la FC de p03).
    """
    out = df.copy()
    out["wellness_missing"] = out["well_readiness"].isna().astype(int)
    skip = {"injury", "srpe_load"}
    cols = [c for c in out.select_dtypes("number").columns
            if c not in skip and not c.startswith(("food_", "perf_")) and not c.endswith(("_outlier", "_missing"))
            and c != "no_watch"]
    rarely_seen = out[cols].isna().groupby(out["participant"]).mean() > 0.8
    X = pd.concat([out[cols].astype(float), pd.get_dummies(out["participant"], dtype=float)], axis=1)
    mice = IterativeImputer(RandomForestRegressor(n_estimators=30, random_state=seed, n_jobs=1),
                            max_iter=5, random_state=seed)
    filled = pd.DataFrame(np.asarray(mice.fit_transform(X))[:, :len(cols)], columns=cols, index=out.index)
    for p in out["participant"].unique():
        keep_nan = rarely_seen.columns[rarely_seen.loc[p]]
        filled.loc[out["participant"] == p, keep_nan] = np.nan
    out[cols] = out[cols].astype(float).fillna(filled)
    out[LIKERT] = out[LIKERT].round().clip(1, 5)
    out["well_readiness"] = out["well_readiness"].round().clip(1, 10)
    return out
