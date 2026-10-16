import numpy as np
import pandas as pd

from src import cleaning, food, modeling


def toy_days(n=60):
    rng = np.random.default_rng(0)
    rows = []
    for p in ["p01", "p05"]:
        for i, d in enumerate(pd.date_range("2020-01-01", periods=n)):
            rows.append({"participant": p, "date": d, "steps": rng.integers(3000, 15000), "distance_km": 5.0,
                         "calories": 2500.0, "zone1_min": 60.0, "zone2_min": 10.0, "zone3_min": 2.0, "trimp": 100.0,
                         "hr_mean": 70.0, "hr_wear_minutes": 1400.0, "resting_hr": rng.uniform(50, 60),
                         "sleep_minutes": rng.uniform(350, 480), "sleep_efficiency": 90.0,
                         "sleep_score": rng.uniform(60, 90), "acwr": 1.0, "weight": 80.0,
                         **{c: rng.integers(1, 6) for c in cleaning.LIKERT}, "well_readiness": rng.integers(1, 11),
                         "injury": 1.0 if (p == "p05" and i == 30) else np.nan})
    return pd.DataFrame(rows)


def test_zero_answers_and_no_watch_days_become_missing():
    df = toy_days()
    df.loc[0, "well_fatigue"] = 0
    df.loc[1, "hr_wear_minutes"] = 60
    out = cleaning.clean(df)
    assert pd.isna(out.loc[0, "well_fatigue"])
    assert out.loc[1, "no_watch"] == 1 and pd.isna(out.loc[1, "steps"])


def test_index_between_0_and_100():
    out = modeling.performance_index(cleaning.clean(toy_days()))
    assert out["perf_index"].dropna().between(0, 100).all()


def test_injury_target_only_looks_ahead():
    df = modeling.add_targets_and_features(modeling.performance_index(cleaning.clean(toy_days())))
    p05 = df[df.participant == "p05"].reset_index(drop=True)
    assert p05.loc[23:29, "y_injury"].eq(1).all()   # la blessure du jour 30 est dans les 7 jours suivants
    assert p05.loc[22, "y_injury"] == 0
    assert p05.loc[30, "injured_today"] == 1


def test_time_folds_train_on_the_past():
    dates = pd.Series(pd.date_range("2020-01-01", periods=100).repeat(2))
    for tr, te in modeling.time_folds(dates):
        assert dates.iloc[tr].max() < dates.iloc[te].min()


def test_portion_calories_follow_usda_values():
    assert food.CLASSES["soft_drink"][1] == round(42 * 330 / 100)
    assert all(30 <= k <= 1000 for _, k in food.CLASSES.values())
