# Suivi d'athlètes : performance et risque de blessure

Les données suivent 3 athlètes pendant 152 jours (novembre 2019 à mars 2020) : montre Fitbit, questionnaires de bien-être et de blessure, rapports de repas et photos des repas.

Ce que fait le projet :
1. une table avec une ligne par athlète et par jour ;
2. la date des photos (EXIF) et une estimation des calories à partir des photos ;
3. le nettoyage des données et le traitement des valeurs manquantes ;
4. un indice de performance sur 100 ;
5. deux modèles, l'un pour le risque de blessure, l'autre pour la performance.

## Lancer le projet

```bash
pip install -r requirements.txt
python main.py          # écrit data/processed/test_dataset.csv et affiche les scores des modèles
pytest -q tests
```

Les données ne sont pas dans le dépôt. Il faut les placer dans `data/raw/` (dossiers `p01`, `p03`, `p05` et `participant-overview.xlsx`) ou indiquer leur dossier avec la variable `DATA_DIR`.

Sur Colab, la première cellule de chaque notebook monte Google Drive (données attendues dans `MyDrive/Colab Notebooks/athlete-data/data/raw/`), clone le dépôt et installe les dépendances. Comme le dépôt est privé, il faut un secret Colab `GITHUB_TOKEN` donnant accès à ce dépôt.

La classification des photos (`python main.py --food`) demande `torch` et `transformers` et un GPU. Ses résultats sont déjà dans `data/external/`, le reste tourne sans.

## Organisation

- `src/data.py` : lecture des fichiers et table journalière
- `src/cleaning.py` : nettoyage et valeurs manquantes
- `src/modeling.py` : indice de performance, cibles, modèles
- `src/food.py` : photos (dates, CLIP, calories)
- `notebooks/` : les 4 étapes avec les résultats, dans l'ordre
- `data/processed/test_dataset.csv` : le jeu de données nettoyé demandé

## Résultats

Données (notebook 1) : la charge déclarée (sRPE) n'est remplie que 31, 2 et 8 jours selon l'athlète, donc je calcule la charge à partir des zones de fréquence cardiaque. p03 n'a pas de FC détaillée et répond presque toujours la même chose au questionnaire. Les blessures sont rares (1, 0 et 10 rapports) et celles de p05 viennent surtout d'un même épisode.

Nettoyage (notebook 2) : les 0 du questionnaire et les valeurs impossibles deviennent des valeurs manquantes, les jours sans montre ne comptent pas comme du repos, les valeurs extrêmes sont signalées sans être supprimées. Sur cinq méthodes de remplissage testées en cachant 10 % des valeurs connues, MICE est la meilleure, c'est elle qui sert pour `test_dataset.csv`.

Modèles (notebook 3) :
- risque de blessure dans les 7 jours : LightGBM obtient une PR-AUC de 0,27 contre 0,16 pour la référence naïve. C'est un signal faible, qui repose surtout sur l'épisode de p05 : il reste exploratoire ;
- performance sur les 3 jours suivants : Ridge fait à peine mieux que la simple moyenne de l'athlète (erreur de 18,3 contre 18,6 sur 100), un écart trop faible pour en tirer quelque chose. Le ressenti quotidien varie beaucoup et n'a presque pas de lien avec les mesures de la montre.

Photos (notebook 4) : toutes les photos ont une date EXIF. Food-101 reconnaissait mal ces repas du quotidien, CLIP avec des classes adaptées reconnaît 21 photos de test sur 24. Les calories estimées restent bien en dessous de la dépense (photos manquantes, portions standard), ce sont des ordres de grandeur.

## Choix principaux

- Chaque athlète est comparé à lui-même (z-scores par athlète), les niveaux de base étant très différents.
- Validation temporelle (entraînement sur le passé, test sur la suite), jamais de mélange aléatoire des jours.
- Chaque modèle est comparé à une référence naïve.
- Hypothèse « pas de rapport = pas de blessure », faute de questionnaire rempli tous les jours.
- Choix sans source directe : les poids de l'indice (vérifiés en comparant avec des poids égaux), l'adaptation du TRIMP aux zones Fitbit, les seuils de port de la montre et les grammages des portions.

## Sources

- Charge : Foster 2001 (sRPE), Edwards 1993 (TRIMP), Gabbett 2016 et Hulin 2016 (ACWR), Impellizzeri 2020 (critique de l'ACWR), Foster 1998 (monotonie)
- Indice : Hooper 1995, McLean 2010, Saw 2016 (questionnaires de bien-être), Fullagar 2015 (sommeil), Buchheit 2014 (FC de repos)
- Données manquantes et valeurs extrêmes : Little et Rubin (2019), van Buuren 2011 (MICE), Iglewicz et Hoaglin 1993
- Évaluation : Saito et Rehmsmeier 2015 (PR-AUC sur classes rares), Bergmeir et Benítez 2012 (validation sur séries temporelles)
- Photos : Radford 2021 (CLIP), Bossard 2014 (Food-101), USDA FoodData Central (calories)

## Limites

Trois athlètes, c'est très peu pour généraliser. Les blessures sont rares et l'hypothèse sur les questionnaires non remplis est forte. L'indice de performance n'est validé que de façon indirecte, et les calories des photos sont approximatives.
