"""Photos de repas : date (EXIF) et calories estimées avec CLIP.

Premier essai : un classifieur Food-101 (101 plats de restaurant). En regardant les résultats, il
se trompait souvent avec une forte confiance (muesli pris pour « clam chowder », tartines de pâté
pour « foie gras »), car les repas photographiés sont surtout des tartines, du muesli, des fruits
et des boissons. D'où CLIP en zéro-shot, avec des classes choisies d'après les photos.

Calories par photo = portion type de la classe, calculée avec les kcal/100 g de l'USDA
(FoodData Central, base SR Legacy, identifiant FDC en commentaire). Les grammages sont des choix.
"""
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import ExifTags, Image, ImageOps

from src.data import DATA_DIR, PARTICIPANTS

CLIP_MODEL = "openai/clip-vit-large-patch14"

# kcal pour 100 g, USDA FoodData Central (SR Legacy)
KCAL_100G = {"bread": 252,        # 172688 Bread, whole-wheat
             "gouda": 356,        # 171241 Cheese, gouda
             "granola": 489,      # 171646 Cereals, granola, homemade
             "milk": 50,          # 171267 Milk, 2% milkfat
             "cola": 42,          # 174852 Beverages, carbonated, cola
             "orange_juice": 49,  # 169100 Orange juice, chilled
             "banana": 89,        # 173944 Bananas, raw
             "apple": 48,         # 171689 Apples, raw, without skin
             "chocolate": 535,    # 167587 Candies, milk chocolate
             "chips": 532,        # 169677 Snacks, potato chips
             "pizza": 268,        # 170317 Pizza, cheese topping, frozen, cooked
             "pasta": 158,        # 169737 Pasta, cooked
             "frankfurter": 322,  # 174614 Frankfurter, beef
             "roll": 310,         # 172793 Rolls, dinner
             "egg": 149,          # 172187 Egg, scrambled
             "coffee": 1}         # 171890 Coffee, brewed


def kcal(*parts):
    return round(sum(KCAL_100G[name] * grams / 100 for name, grams in parts))


# classe -> (descriptions pour CLIP, kcal d'une portion type)
CLASSES = {
    "open_sandwich": (["an open-faced sandwich", "a slice of bread with cheese", "a slice of bread with ham",
                       "a crispbread with a spread", "a bread roll with toppings"], kcal(("bread", 70), ("gouda", 20))),
    "cereal": (["a bowl of muesli with milk", "a bowl of cereal", "a bowl of oatmeal porridge", "granola with yogurt"],
               kcal(("granola", 50), ("milk", 150))),
    "soft_drink": (["a can of soda", "a glass of cola", "a soft drink bottle"], kcal(("cola", 330))),
    "juice": (["a glass of orange juice", "a glass of fruit juice"], kcal(("orange_juice", 250))),
    "fruit": (["a banana", "an apple", "an orange", "fresh fruit"], kcal(("banana", 60), ("apple", 90))),
    "chocolate": (["a bar of chocolate", "candy and sweets"], kcal(("chocolate", 40))),
    "chips": (["a bag of potato chips", "crisps"], kcal(("chips", 50))),
    "pizza": (["a pizza", "a slice of pizza"], kcal(("pizza", 200))),
    "hot_dog": (["a hot dog in a bun"], kcal(("roll", 50), ("frankfurter", 50))),
    "pasta_dish": (["a plate of pasta", "a bowl of noodles", "a rice dish"], kcal(("pasta", 300))),
    "omelette": (["an omelette", "scrambled eggs", "a pancake"], kcal(("egg", 150))),
    "coffee": (["a cup of coffee", "a cappuccino", "a cup of tea"], kcal(("coffee", 200), ("milk", 100))),
    "milk_drink": (["a bottle of milk drink", "a glass of milk"], kcal(("milk", 330))),
    "cheese": (["a piece of cheese"], kcal(("gouda", 40))),
    "pastry": (["a cake", "a sweet bun", "a cookie", "a pastry"], kcal(("roll", 90))),
    # plats composés : pas de référence USDA unique, valeurs estimées
    "hot_meal": (["a plate with meat, potatoes and vegetables", "a cooked dinner plate", "fish with potatoes"], 600),
    "wrap": (["a wrap", "a burrito"], 450),
    "yogurt": (["a cup of yogurt", "a yogurt dessert"], 140),
    "salad": (["a bowl of salad"], 200),
    "alcohol": (["a glass of wine", "a glass of beer"], 130),
}


def photo_date(path):
    """EXIF DateTimeOriginal, sinon DateTime, sinon date dans le nom, sinon date du fichier."""
    try:
        exif = Image.open(path).getexif()
        tags = {ExifTags.TAGS.get(k, k): v for k, v in {**exif, **exif.get_ifd(0x8769)}.items()}
        for key in ("DateTimeOriginal", "DateTime"):
            if tags.get(key):
                return datetime.strptime(str(tags[key]), "%Y:%m:%d %H:%M:%S"), "exif"
    except Exception:
        pass
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", path.stem)
    if m:
        return datetime(int(m[1]), int(m[2]), int(m[3])), "filename"
    return datetime.fromtimestamp(path.stat().st_mtime), "file_date"


def list_photos():
    rows = []
    for p in PARTICIPANTS:
        for path in sorted((DATA_DIR / p / "food-images").iterdir()):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                dt, src = photo_date(path)
                rows.append({"participant": p, "image": path.name, "datetime": dt, "date_source": src})
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
    return df


def classify(photos, batch=16):
    """CLIP zéro-shot : on compare l'image à la moyenne des descriptions de chaque classe."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    feats = lambda out: out if hasattr(out, "norm") else out.pooler_output  # transformers 4 ou 5
    names = list(CLASSES)
    with torch.no_grad():
        class_emb = []
        for name in names:
            t = feats(model.get_text_features(**proc(text=[f"a photo of {d}" for d in CLASSES[name][0]],
                                                         return_tensors="pt", padding=True).to(device)))
            t = (t / t.norm(dim=-1, keepdim=True)).mean(0)
            class_emb.append(t / t.norm())
        class_emb = torch.stack(class_emb)
    kcal_t = torch.tensor([CLASSES[n][1] for n in names], dtype=torch.float32)
    out = []
    for i in range(0, len(photos), batch):
        rows = photos.iloc[i:i + batch]
        imgs = [ImageOps.exif_transpose(Image.open(DATA_DIR / r.participant / "food-images" / r.image)).convert("RGB")
                for r in rows.itertuples()]
        with torch.no_grad():
            e = feats(model.get_image_features(**proc(images=imgs, return_tensors="pt").to(device)))
            probs = (100 * (e / e.norm(dim=-1, keepdim=True)) @ class_emb.T).softmax(-1).cpu()
        top_p, top_i = probs.topk(3, dim=-1)
        est = (top_p / top_p.sum(-1, keepdim=True) * kcal_t[top_i]).sum(-1)
        out += [{"label": names[int(top_i[j, 0])], "prob": float(top_p[j, 0]), "kcal": float(est[j])}
                for j in range(len(rows))]
    return pd.concat([photos.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def daily_calories(pred):
    return (pred.groupby(["participant", "date"])
            .agg(food_n_photos=("image", "size"), food_kcal=("kcal", "sum")).reset_index())
