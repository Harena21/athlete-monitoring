"""Refait tout le traitement : données brutes -> data/processed/test_dataset.csv -> scores des modèles.

    python main.py           # jeu de données + modèles
    python main.py --food    # relance aussi CLIP sur les photos (GPU conseillé)
"""
import argparse
from pathlib import Path

import pandas as pd

from src import cleaning, data, food, modeling

OUT = Path(__file__).parent / "data"
FOOD_PRED = OUT / "external" / "food_predictions.csv"


def build_dataset(run_food=False):
    """Table journalière nettoyée avec l'indice ; écrit aussi la version imputée test_dataset.csv."""
    clean = modeling.performance_index(cleaning.clean(data.build_daily()))
    if run_food:
        FOOD_PRED.parent.mkdir(parents=True, exist_ok=True)
        food.classify(food.list_photos()).to_csv(FOOD_PRED, index=False)
    if FOOD_PRED.exists():
        pred = pd.read_csv(FOOD_PRED, parse_dates=["date"])
        clean = clean.merge(food.daily_calories(pred), on=["participant", "date"], how="left")
    (OUT / "processed").mkdir(parents=True, exist_ok=True)
    cleaning.impute(clean).to_csv(OUT / "processed" / "test_dataset.csv", index=False)
    return clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--food", action="store_true", help="relancer la classification des photos")
    clean = build_dataset(run_food=parser.parse_args().food)
    print("test_dataset.csv écrit")
    df = modeling.add_targets_and_features(clean)
    print("\nBlessure dans les 7 jours\n", modeling.evaluate_injury(df).round(3).to_string(index=False))
    print("\nPerformance sur 3 jours\n", modeling.evaluate_performance(df).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
