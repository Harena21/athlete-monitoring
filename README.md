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

Nettoyage (notebook 2) : les 0 du questionnaire et les valeurs impossibles deviennent des valeurs manquantes, les jours sans montre ne comptent pas comme du repos, les valeurs extrêmes sont signalées sans être supprimées. Pour les valeurs manquantes, j'ai comparé cinq méthodes, de la plus simple à la plus avancée :
1. moyenne de la colonne, tous athlètes confondus ;
2. médiane de l'athlète, qui tient compte des différences entre athlètes ;
3. interpolation linéaire dans le temps, athlète par athlète (la valeur d'un jour est tirée des jours voisins) ;
4. KNN (5 plus proches voisins), sur des variables standardisées pour qu'aucune ne domine la distance ;
5. MICE (`IterativeImputer` de scikit-learn avec une forêt aléatoire) : chaque variable est prédite à partir des autres, en plusieurs passes.

Le test : je cache 10 % des valeurs connues de 11 variables (pas, FC de repos, TRIMP, sommeil, questionnaire, poids), je les remplis avec chaque méthode et je mesure l'erreur (RMSE divisée par l'écart-type de la variable), sur 3 tirages. Erreur moyenne : MICE 0,90, interpolation 0,93, médiane 0,98, KNN 1,02, moyenne 1,07. MICE l'emporte de peu et l'ordre change d'un tirage à l'autre, mais c'est la méthode retenue pour `test_dataset.csv`. Je ne remplis ni les blessures, ni la sRPE, ni les calories des photos, ni l'indice, ni une variable absente plus de 80 % du temps chez un athlète (la FC de p03, par exemple).

Modèles (notebook 3) :
- risque de blessure dans les 7 jours : LightGBM obtient une PR-AUC de 0,27 contre 0,16 pour la référence naïve. C'est un signal faible, qui repose surtout sur l'épisode de p05 : il reste exploratoire ;
- performance sur les 3 jours suivants : Ridge fait à peine mieux que la simple moyenne de l'athlète (erreur de 18,3 contre 18,6 sur 100), un écart trop faible pour en tirer quelque chose. Le ressenti quotidien varie beaucoup et n'a presque pas de lien avec les mesures de la montre.

Photos (notebook 4) : toutes les photos ont une date EXIF. Food-101 reconnaissait mal ces repas du quotidien, CLIP avec des classes adaptées reconnaît 21 photos de test sur 24. Les calories estimées restent bien en dessous de la dépense (photos manquantes, portions standard), ce sont des ordres de grandeur.

## Méthode en détail

### Table journalière (`src/data.py`)
- Calendrier continu du 1er novembre 2019 au 31 mars 2020 pour chaque athlète : un jour sans donnée reste une ligne vide, il ne disparaît pas.
- Fitbit (pas, calories, distance) : somme des valeurs par minute. FC détaillée : moyenne du jour et nombre de minutes avec une mesure (temps de port). Sommeil : nuit principale seulement, rattachée au jour du réveil.
- pmsys et score de sommeil sont en UTC : je les convertis à l'heure d'Oslo avant de prendre le jour. S'il y a plusieurs questionnaires dans la journée, je garde le dernier.
- sRPE = RPE × durée en minutes, additionnée sur la journée (Foster 2001).
- Blessure = 1 si le rapport du jour contient au moins une blessure.

### Charge d'entraînement
- TRIMP = 1,5 × minutes en zone 1 + 3,5 × minutes en zone 2 + 5 × minutes en zone 3. Edwards (1993) découpe la FC en 5 zones de 10 % de la FC max, pondérées de 1 à 5. Fitbit n'en donne que 3 (50 à 69 %, 70 à 84 %, 85 % et plus de la FC max) : chacune reçoit le poids moyen des zones d'Edwards qu'elle recouvre. C'est une adaptation de ma part.
- Charge aiguë = moyenne du TRIMP sur 7 jours, charge chronique = moyenne sur 28 jours, ACWR = aiguë / chronique (Gabbett 2016).
- Monotonie = moyenne sur 7 jours / écart-type sur 7 jours (Foster 1998).
- Écart de FC de repos = FC de repos du jour moins la moyenne des 28 jours précédents (une hausse peut signaler de la fatigue).

### Nettoyage (`src/cleaning.py`)
- Codes invalides : un 0 sur une échelle de 1 à 5 ou une readiness à 0 devient une valeur manquante.
- Bornes physiologiques, en dehors desquelles la valeur devient manquante : FC de repos 30 à 120, FC moyenne 30 à 200, pas 0 à 60 000, sommeil 0 à 16 h, poids 30 à 200 kg.
- Jour sans montre : moins de 600 minutes de FC enregistrées, ou, pour p03 qui n'a pas de FC détaillée, moins de 200 pas. Les colonnes d'activité de ce jour passent à NaN et `no_watch` vaut 1.
- Valeurs extrêmes : z-score robuste par athlète, 0,6745 × (x − médiane) / MAD, seuil 3,5 (Iglewicz et Hoaglin 1993), sur les pas, le TRIMP et la durée de sommeil. Signalées dans `*_outlier`, gardées.

### Indice de performance (`src/modeling.py`)
1. Chaque variable est centrée-réduite sur la saison de l'athlète (z-score bridé à ±3). La FC de repos est inversée (plus basse = mieux).
2. Quatre blocs, moyenne des z-scores de leurs variables :
   - bien-être, 40 % : fatigue, humeur, qualité du sommeil, courbatures, stress, readiness ;
   - sommeil, 25 % : score Fitbit et durée ;
   - physiologie, 20 % : FC de repos ;
   - charge, 15 % : pénalité quand l'ACWR sort de la zone 0,8 à 1,3.
3. Moyenne pondérée des blocs disponibles. Sans questionnaire ce jour-là, pas d'indice.
4. Nouveau z-score par athlète, puis passage en percentile avec la loi normale : 50 = état habituel de l'athlète, 84 ≈ un écart-type au-dessus.

Contrôles : corrélation de 0,97 avec un indice à poids égaux, et de 0,22 avec la readiness déclarée seule.

### Modèles
- Validation : les dates sont coupées en 5 périodes égales (environ un mois chacune). On entraîne sur les premières et on teste sur la suivante, 4 fois, en agrandissant à chaque fois la période d'entraînement. Le remplissage des valeurs manquantes (médiane) est appris sur l'entraînement seulement.
- Blessure : cible = au moins un rapport de blessure dans les 7 jours suivants ; les jours déjà blessés sont exclus. 25 variables : activité, charge (TRIMP, 7 j, 28 j, ACWR, monotonie), FC, sommeil, questionnaire, repas, alcool, âge, moyennes sur 7 jours de la readiness, de la fatigue, du score de sommeil et des pas, nombre de questionnaires remplis sur 7 jours. Modèles : régression logistique (L2, C = 0,1) et LightGBM peu profond (200 arbres, profondeur 3), tous deux avec des classes pondérées. Référence : prédire la fréquence des blessures vue à l'entraînement. Mesure : PR-AUC (ROC-AUC aussi affichée).
- Performance : cible = moyenne de l'indice sur les 3 jours suivants. 9 variables : indice du jour, moyennes de l'indice sur 3 et 7 jours, readiness, fatigue et score de sommeil sur 7 jours, écart de FC de repos, ACWR, charge sur 7 jours. Modèles : Ridge (alpha = 10) et LightGBM (300 arbres, profondeur 3). Références : moyenne de l'entraînement et persistance (les 3 prochains jours = les 3 derniers). Mesure : erreur absolue moyenne sur 100.

### Photos (`src/food.py`)
- Date : `DateTimeOriginal` de l'EXIF, avec repli sur `DateTime`, puis la date dans le nom du fichier, puis la date du fichier.
- Reconnaissance : CLIP ViT-L/14 en zéro-shot, 20 classes décrites chacune par plusieurs phrases (la liste est dans `CLASSES`).
- Calories : chaque classe a une portion type calculée avec les valeurs USDA FoodData Central (pour 100 g, identifiants FDC dans le code). Par photo, moyenne des 3 classes les plus probables pondérée par leur probabilité, puis somme par jour.
- Évaluation : 24 photos annotées à la main, différentes de celles qui ont servi à choisir les classes.

## Colonnes de `test_dataset.csv`

456 lignes (3 athlètes × 152 jours) et 41 colonnes. Les valeurs manquantes sont remplies par MICE, sauf pour les colonnes indiquées « non remplie ».

| Colonnes | Contenu |
|---|---|
| `date`, `participant`, `age` | jour, athlète, âge |
| `steps`, `calories`, `distance_km` | pas, calories dépensées (Fitbit), distance en km |
| `hr_mean`, `hr_wear_minutes`, `resting_hr` | FC moyenne, minutes de port de la montre, FC de repos |
| `zone1_min`, `zone2_min`, `zone3_min` | minutes dans chaque zone de FC Fitbit |
| `sleep_minutes`, `sleep_efficiency`, `sleep_deep_min`, `sleep_score` | durée, efficacité (%), sommeil profond, score Fitbit |
| `well_fatigue`, `well_mood`, `well_sleep_quality`, `well_soreness`, `well_stress` | questionnaire pmsys, 1 à 5, plus haut = mieux |
| `well_readiness` | forme déclarée, 1 à 10 |
| `srpe_load` | charge déclarée, RPE × minutes (non remplie) |
| `injury` | 1 si un rapport de blessure ce jour-là (non remplie) |
| `n_meals`, `weight`, `alcohol` | nombre de repas, poids, alcool (1 = oui), d'après googledocs |
| `trimp`, `load_7d`, `load_28d`, `acwr`, `monotony` | charge et indicateurs dérivés (voir plus haut) |
| `resting_hr_delta` | FC de repos moins sa moyenne des 28 jours précédents |
| `no_watch` | 1 si la montre n'a pas été portée ce jour-là |
| `steps_outlier`, `trimp_outlier`, `sleep_minutes_outlier` | 1 si valeur extrême |
| `perf_index` | indice de performance de 0 à 100 (non rempli) |
| `food_n_photos`, `food_kcal` | photos de repas et calories estimées (février et mars, non remplies) |
| `wellness_missing` | 1 si le questionnaire n'a pas été rempli (avant imputation) |

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
