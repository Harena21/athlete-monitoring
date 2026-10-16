"""Indice de performance, cibles, variables et modèles."""
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import norm
from sklearn import set_config
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
set_config(transform_output="pandas")  # garde les noms de colonnes jusqu'à LightGBM

# ---------------------------------------------------------------- indice de performance (0-100)
# Blocs et sources : bien-être (Hooper 1995, McLean 2010, Saw 2016), sommeil (Fullagar 2015),
# FC de repos (Buchheit 2014), charge via l'ACWR (Gabbett 2016). Toutes les échelles pmsys vont
# dans le sens « plus haut = mieux » (vérifié : corrélations positives avec la readiness).
BLOCKS = {
    "wellness": {"well_fatigue": 1, "well_mood": 1, "well_sleep_quality": 1, "well_soreness": 1,
                 "well_stress": 1, "well_readiness": 1},
    "sleep": {"sleep_score": 1, "sleep_minutes": 1},
    "physio": {"resting_hr": -1},
}
WEIGHTS = {"wellness": 0.40, "sleep": 0.25, "physio": 0.20, "load": 0.15}


def zscore(s):
    sd = s.std()
    return ((s - s.mean()) / sd).clip(-3, 3) if sd else s * np.nan


def performance_index(df, weights=WEIGHTS):
    """Chaque athlète est comparé à sa propre saison ; 50 = état habituel.

    Une première version comparait chaque jour aux 28 jours précédents : la ligne de base suivait
    les périodes de fatigue et l'indice ne gardait que du bruit. La ligne de base sur la saison
    utilise des jours futurs, mais seulement pour définir la cible, pas pour les variables.
    """
    df = df.copy()
    g = df.groupby("participant")
    scores = {}
    for block, items in BLOCKS.items():
        scores[block] = pd.concat([sign * g[c].transform(lambda s: zscore(s.astype(float)))
                                   for c, sign in items.items()], axis=1).mean(axis=1)
    out_of_zone = np.maximum(0.8 - df["acwr"], 0) + np.maximum(df["acwr"] - 1.3, 0)
    scores["load"] = -(out_of_zone / 0.25).clip(upper=3)
    S = pd.DataFrame(scores)
    w = pd.Series(weights)
    total = (S.fillna(0) * w).sum(axis=1) / (S.notna() * w).sum(axis=1)
    total[S["wellness"].isna()] = np.nan  # pas d'indice sans questionnaire
    z = total.groupby(df["participant"]).transform(lambda s: (s - s.mean()) / s.std())
    df["perf_index"] = 100 * norm.cdf(z)
    return df


# ---------------------------------------------------------------- cibles et variables
FEATURES = ["steps", "trimp", "load_7d", "load_28d", "acwr", "monotony", "resting_hr", "resting_hr_delta",
            "hr_mean", "sleep_minutes", "sleep_efficiency", "sleep_score", "well_fatigue", "well_mood",
            "well_readiness", "well_soreness", "well_stress", "n_meals", "alcohol", "age",
            "readiness_7d", "fatigue_7d", "sleep_score_7d", "steps_7d", "answered_7d"]
PERF_FEATURES = ["perf_index", "perf_3d", "perf_7d", "readiness_7d", "fatigue_7d", "sleep_score_7d",
                 "resting_hr_delta", "acwr", "load_7d"]


def add_targets_and_features(df):
    df = df.sort_values(["participant", "date"]).copy()
    g = df.groupby("participant")
    roll = lambda col, w: g[col].transform(lambda s: s.astype(float).rolling(w, min_periods=1).mean())
    df["perf_3d"], df["perf_7d"] = roll("perf_index", 3), roll("perf_index", 7)
    df["readiness_7d"], df["fatigue_7d"] = roll("well_readiness", 7), roll("well_fatigue", 7)
    df["sleep_score_7d"], df["steps_7d"] = roll("sleep_score", 7), roll("steps", 7)
    # nombre de questionnaires remplis sur 7 jours : ne pas répondre est aussi une information
    df["answered_7d"] = g["well_readiness"].transform(lambda s: s.notna().astype(float).rolling(7, min_periods=1).sum())

    # performance : moyenne de l'indice sur les 3 jours suivants (un seul jour est trop bruité)
    df["y_perf"] = g["perf_index"].transform(lambda s: s[::-1].rolling(3, min_periods=2).mean()[::-1].shift(-1))

    # blessure : au moins un rapport dans les 7 jours suivants. Hypothèse : pas de rapport = pas de
    # blessure (le questionnaire est rarement rempli). Les 7 derniers jours n'ont pas de cible.
    inj = df["injury"].fillna(0)
    nxt = inj.groupby(df["participant"]).transform(lambda s: s[::-1].rolling(7, min_periods=1).sum()[::-1].shift(-1))
    df["y_injury"] = (nxt > 0).astype(float)
    df.loc[nxt.isna(), "y_injury"] = np.nan
    df["injured_today"] = inj.astype(int)
    return df


# ---------------------------------------------------------------- validation
def time_folds(dates, n=4):
    """On entraîne sur le passé et on teste sur la période suivante (jamais de K-fold aléatoire :
    deux jours voisins se ressemblent et l'information fuiterait)."""
    edges = pd.date_range(dates.min(), dates.max(), periods=n + 2)
    for k in range(1, n + 1):
        train = dates < edges[k]
        test = (dates >= edges[k]) & ((dates < edges[k + 1]) if k < n else True)
        yield np.where(train)[0], np.where(test)[0]


def injury_models():
    return {
        "logistique": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                    LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)),
        "LightGBM": make_pipeline(SimpleImputer(strategy="median"),
                                  LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=3, num_leaves=7,
                                                 min_child_samples=10, reg_lambda=5, class_weight="balanced",
                                                 random_state=SEED, verbose=-1)),
    }


def evaluate_injury(df):
    data = df[(df["injured_today"] == 0) & df["y_injury"].notna()].reset_index(drop=True)
    X, y = data[FEATURES].astype(float), data["y_injury"].astype(int)
    preds = {name: np.full(len(data), np.nan) for name in ["référence (prévalence)", *injury_models()]}
    for tr, te in time_folds(data["date"]):
        if y.iloc[tr].nunique() < 2:
            continue
        preds["référence (prévalence)"][te] = y.iloc[tr].mean()
        for name, model in injury_models().items():
            preds[name][te] = model.fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    rows = []
    for name, p in preds.items():
        m = ~np.isnan(p)
        rows.append({"modèle": name, "PR-AUC": average_precision_score(y[m], p[m]),
                     "ROC-AUC": roc_auc_score(y[m], p[m]), "n": int(m.sum()), "positifs": int(y[m].sum())})
    return pd.DataFrame(rows)


def evaluate_performance(df):
    data = df[df["y_perf"].notna()].reset_index(drop=True)
    X, y = data[PERF_FEATURES].astype(float), data["y_perf"]
    models = {
        "Ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10)),
        "LightGBM": make_pipeline(SimpleImputer(strategy="median"),
                                  LGBMRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, num_leaves=7,
                                                min_child_samples=10, reg_lambda=5, random_state=SEED, verbose=-1)),
    }
    preds = {name: np.full(len(data), np.nan) for name in ["moyenne", "persistance", *models]}
    for tr, te in time_folds(data["date"]):
        preds["moyenne"][te] = y.iloc[tr].mean()
        preds["persistance"][te] = data["perf_3d"].iloc[te]  # les 3 prochains jours = les 3 derniers
        for name, model in models.items():
            preds[name][te] = model.fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
    return pd.DataFrame([{"modèle": n, "MAE": mean_absolute_error(y[~np.isnan(p)], p[~np.isnan(p)])}
                         for n, p in preds.items()]).sort_values("MAE")
