# estimator.py
# Volume and weight estimates for only the item classes supported by detector.py.

import re

from pricing import get_truck_options


# Raw model labels from weights/household_v2_best.pt that are useful for moving inventory.
# Labels for fixed room parts, food, and tiny non-inventory clutter are ignored below.
SUPPORTED_MODEL_LABELS = {
    "AlarmClock", "ArmChair", "Bed", "Blinds", "Book", "Bottle", "Bowl", "Box",
    "Cabinet", "Candle", "CellPhone", "Chair", "Cloth", "CoffeeMachine",
    "CoffeeTable", "Cup", "Curtains", "Desk", "DeskLamp", "DiningTable",
    "Drawer", "Dresser", "FloorLamp", "Footstool", "Fork", "Fridge",
    "GarbageCan", "HandTowel", "HousePlant", "Kettle", "Knife", "Laptop",
    "LaundryHamper", "Microwave", "Mirror", "Mug", "Newspaper", "Ottoman",
    "Painting", "Pan", "Pen", "Pencil", "Pillow", "Plate", "Poster", "Pot",
    "RemoteControl", "Safe", "Shelf", "SideTable", "Sofa", "Spoon", "Statue",
    "TeddyBear", "Television", "TissueBox", "Toaster", "Towel", "Vase",
    "WateringCan", "WineBottle",
}


ITEM_DATA = {
    # Living room
    "sofa":             {"volume": 30, "weight": 60, "fragility": "Normal",  "category": "Living Room"},
    "armchair":         {"volume": 12, "weight": 18, "fragility": "Normal",  "category": "Living Room"},
    "chair":            {"volume": 10, "weight": 12, "fragility": "Normal",  "category": "Living Room"},
    "coffee table":     {"volume": 12, "weight": 20, "fragility": "Normal",  "category": "Living Room"},
    "side table":       {"volume": 6,  "weight": 10, "fragility": "Normal",  "category": "Living Room"},
    "ottoman":          {"volume": 8,  "weight": 10, "fragility": "Normal",  "category": "Living Room"},
    "footstool":        {"volume": 4,  "weight": 6,  "fragility": "Normal",  "category": "Living Room"},
    "tv":               {"volume": 8,  "weight": 15, "fragility": "Fragile", "category": "Living Room"},
    "remote":           {"volume": 1,  "weight": 1,  "fragility": "Fragile", "category": "Living Room"},
    "clock":            {"volume": 1,  "weight": 2,  "fragility": "Fragile", "category": "Living Room"},
    "lamp":             {"volume": 4,  "weight": 5,  "fragility": "Fragile", "category": "Living Room"},
    "potted plant":     {"volume": 2,  "weight": 5,  "fragility": "Normal",  "category": "Living Room"},
    "vase":             {"volume": 1,  "weight": 2,  "fragility": "Fragile", "category": "Living Room"},
    "painting":         {"volume": 3,  "weight": 4,  "fragility": "Fragile", "category": "Living Room"},
    "mirror":           {"volume": 4,  "weight": 8,  "fragility": "Fragile", "category": "Living Room"},
    "statue":           {"volume": 2,  "weight": 6,  "fragility": "Fragile", "category": "Living Room"},
    "candle":           {"volume": 1,  "weight": 1,  "fragility": "Fragile", "category": "Living Room"},
    "curtains":         {"volume": 2,  "weight": 3,  "fragility": "Normal",  "category": "Living Room"},
    "blinds":           {"volume": 2,  "weight": 4,  "fragility": "Normal",  "category": "Living Room"},

    # Bedroom and storage
    "bed":              {"volume": 55, "weight": 80, "fragility": "Normal",  "category": "Bedroom"},
    "cabinet":          {"volume": 20, "weight": 45, "fragility": "Heavy",   "category": "Bedroom"},
    "dresser":          {"volume": 25, "weight": 50, "fragility": "Heavy",   "category": "Bedroom"},
    "drawer":           {"volume": 18, "weight": 35, "fragility": "Heavy",   "category": "Bedroom"},
    "shelf":            {"volume": 18, "weight": 25, "fragility": "Normal",  "category": "Bedroom"},
    "pillow":           {"volume": 2,  "weight": 1,  "fragility": "Normal",  "category": "Bedroom"},
    "cloth":            {"volume": 1,  "weight": 1,  "fragility": "Normal",  "category": "Bedroom"},
    "towel":            {"volume": 1,  "weight": 1,  "fragility": "Normal",  "category": "Bedroom"},
    "laundry hamper":   {"volume": 5,  "weight": 3,  "fragility": "Normal",  "category": "Bedroom"},
    "teddy bear":       {"volume": 1,  "weight": 1,  "fragility": "Normal",  "category": "Bedroom"},
    "safe":             {"volume": 6,  "weight": 45, "fragility": "Heavy",   "category": "Bedroom"},
    "box":              {"volume": 4,  "weight": 8,  "fragility": "Normal",  "category": "Bedroom"},

    # Kitchen
    "refrigerator":     {"volume": 20, "weight": 65, "fragility": "Heavy",   "category": "Kitchen"},
    "microwave":        {"volume": 3,  "weight": 12, "fragility": "Fragile", "category": "Kitchen"},
    "toaster":          {"volume": 2,  "weight": 4,  "fragility": "Fragile", "category": "Kitchen"},
    "coffee machine":   {"volume": 2,  "weight": 5,  "fragility": "Fragile", "category": "Kitchen"},
    "kettle":           {"volume": 1,  "weight": 2,  "fragility": "Fragile", "category": "Kitchen"},
    "dining table":     {"volume": 25, "weight": 45, "fragility": "Normal",  "category": "Kitchen"},
    "plate":            {"volume": 1,  "weight": 1,  "fragility": "Fragile", "category": "Kitchen"},
    "pan":              {"volume": 1,  "weight": 2,  "fragility": "Normal",  "category": "Kitchen"},
    "pot":              {"volume": 2,  "weight": 3,  "fragility": "Normal",  "category": "Kitchen"},
    "mug":              {"volume": 1,  "weight": 1,  "fragility": "Fragile", "category": "Kitchen"},
    "bowl":             {"volume": 1,  "weight": 1,  "fragility": "Fragile", "category": "Kitchen"},
    "cup":              {"volume": 1,  "weight": 1,  "fragility": "Fragile", "category": "Kitchen"},
    "bottle":           {"volume": 1,  "weight": 1,  "fragility": "Fragile", "category": "Kitchen"},
    "fork":             {"volume": 1,  "weight": 1,  "fragility": "Normal",  "category": "Kitchen"},
    "knife":            {"volume": 1,  "weight": 1,  "fragility": "Normal",  "category": "Kitchen"},
    "spoon":            {"volume": 1,  "weight": 1,  "fragility": "Normal",  "category": "Kitchen"},

    # Utility and office
    "garbage can":      {"volume": 4,  "weight": 3,  "fragility": "Normal",  "category": "Utility"},
    "watering can":     {"volume": 1,  "weight": 1,  "fragility": "Normal",  "category": "Utility"},
    "tissue box":       {"volume": 1,  "weight": 1,  "fragility": "Normal",  "category": "Utility"},
    "desk":             {"volume": 22, "weight": 40, "fragility": "Normal",  "category": "Office"},
    "laptop":           {"volume": 1,  "weight": 2,  "fragility": "Fragile", "category": "Office"},
    "book":             {"volume": 1,  "weight": 2,  "fragility": "Normal",  "category": "Office"},
    "newspaper":        {"volume": 1,  "weight": 1,  "fragility": "Normal",  "category": "Office"},
    "pen":              {"volume": 1,  "weight": 1,  "fragility": "Normal",  "category": "Office"},
    "pencil":           {"volume": 1,  "weight": 1,  "fragility": "Normal",  "category": "Office"},
    "cell phone":       {"volume": 1,  "weight": 1,  "fragility": "Fragile", "category": "Office"},
}


MODEL_LABEL_TO_ITEM = {
    "AlarmClock": "clock",
    "ArmChair": "armchair",
    "Bed": "bed",
    "Blinds": "blinds",
    "Book": "book",
    "Bottle": "bottle",
    "Bowl": "bowl",
    "Box": "box",
    "Cabinet": "cabinet",
    "Candle": "candle",
    "CellPhone": "cell phone",
    "Chair": "chair",
    "Cloth": "cloth",
    "CoffeeMachine": "coffee machine",
    "CoffeeTable": "coffee table",
    "Cup": "cup",
    "Curtains": "curtains",
    "Desk": "desk",
    "DeskLamp": "lamp",
    "DiningTable": "dining table",
    "Drawer": "drawer",
    "Dresser": "dresser",
    "FloorLamp": "lamp",
    "Footstool": "footstool",
    "Fork": "fork",
    "Fridge": "refrigerator",
    "GarbageCan": "garbage can",
    "HandTowel": "towel",
    "HousePlant": "potted plant",
    "Kettle": "kettle",
    "Knife": "knife",
    "Laptop": "laptop",
    "LaundryHamper": "laundry hamper",
    "Microwave": "microwave",
    "Mirror": "mirror",
    "Mug": "mug",
    "Newspaper": "newspaper",
    "Ottoman": "ottoman",
    "Painting": "painting",
    "Pan": "pan",
    "Pen": "pen",
    "Pencil": "pencil",
    "Pillow": "pillow",
    "Plate": "plate",
    "Poster": "painting",
    "Pot": "pot",
    "RemoteControl": "remote",
    "Safe": "safe",
    "Shelf": "shelf",
    "SideTable": "side table",
    "Sofa": "sofa",
    "Spoon": "spoon",
    "Statue": "statue",
    "TeddyBear": "teddy bear",
    "Television": "tv",
    "TissueBox": "tissue box",
    "Toaster": "toaster",
    "Towel": "towel",
    "Vase": "vase",
    "WateringCan": "watering can",
    "WineBottle": "bottle",
}


IGNORED_MODEL_LABELS = {
    "Apple", "BaseballBat", "BasketBall", "Bathtub", "Boots", "Bread",
    "ButterKnife", "CounterTop", "CreditCard", "DishSponge", "Egg",
    "Faucet", "HandTowelHolder", "KeyChain", "Ladle", "Lettuce",
    "LightSwitch", "PepperShaker", "Plunger", "Potato", "SaltShaker",
    "ScrubBrush", "ShowerCurtain", "ShowerDoor", "ShowerGlass",
    "ShowerHead", "Sink", "SoapBar", "SoapBottle", "Spatula",
    "SprayBottle", "StoveBurner", "TennisRacket", "Toilet",
    "ToiletPaper", "ToiletPaperHanger", "Tomato", "TowelHolder",
    "Watch", "Window",
}


def normalize_item_name(item_name):
    """Normalize display text, aliases, and model labels into estimator keys."""
    if item_name in MODEL_LABEL_TO_ITEM:
        return MODEL_LABEL_TO_ITEM[item_name]

    name = str(item_name).strip()
    name = re.sub(r"(?<!^)(?=[A-Z])", " ", name)
    name = name.replace("_", " ").replace("-", " ").lower()
    name = re.sub(r"\s+", " ", name).strip()

    aliases = {
        "alarm clock": "clock",
        "arm chair": "armchair",
        "cellphone": "cell phone",
        "coffee machine": "coffee machine",
        "coffee table": "coffee table",
        "desk lamp": "lamp",
        "dining table": "dining table",
        "floor lamp": "lamp",
        "fridge": "refrigerator",
        "hand towel": "towel",
        "house plant": "potted plant",
        "houseplant": "potted plant",
        "remote control": "remote",
        "side table": "side table",
        "television": "tv",
        "wine bottle": "bottle",
    }
    return aliases.get(name, name)


def canonical_inventory_item(item_name):
    """Return a supported inventory item name, or None for ignored labels."""
    if item_name in IGNORED_MODEL_LABELS:
        return None

    normalized = normalize_item_name(item_name)
    if normalized in ITEM_DATA:
        return normalized
    return None


def get_item_data(item_name):
    """Return volume/weight/fragility data for a supported item."""
    name = normalize_item_name(item_name)
    if name not in ITEM_DATA:
        raise KeyError(f"Unsupported inventory item: {item_name}")
    return ITEM_DATA[name]


INVENTORY_ITEMS = sorted(
    {item for item in MODEL_LABEL_TO_ITEM.values() if item in ITEM_DATA}
)


TRUCK_OPTIONS = get_truck_options()


def get_truck_recommendation(total_volume):
    """Return the smallest truck that fits the total volume with a 20% buffer."""
    for truck in TRUCK_OPTIONS:
        if truck["capacity"] >= total_volume * 1.2:
            return truck
    return TRUCK_OPTIONS[-1]


def summarise_items(detected_items):
    """
    Convert {item_name: count} into item details, totals, and truck recommendation.
    """
    summary = []
    total_volume = 0
    total_weight = 0

    for item_name, count in detected_items.items():
        canonical_name = normalize_item_name(item_name)
        data = get_item_data(canonical_name)
        item_volume = data["volume"] * count
        item_weight = data["weight"] * count
        total_volume += item_volume
        total_weight += item_weight

        summary.append({
            "Item": canonical_name.title(),
            "Count": count,
            "Category": data["category"],
            "Fragility": data["fragility"],
            "Volume (cu ft)": item_volume,
            "Weight (kg)": item_weight,
        })

    return {
        "items": summary,
        "total_volume": round(total_volume, 1),
        "total_weight": round(total_weight, 1),
        "truck": get_truck_recommendation(total_volume),
    }


PACKING_TIPS = {
    "sofa": "Wrap in moving blankets. Remove legs if possible to save truck space.",
    "armchair": "Wrap in moving blankets or shrink wrap to protect upholstery.",
    "chair": "Stack chairs seat-to-seat. Wrap legs in bubble wrap to prevent scratches.",
    "coffee table": "Wrap corners in bubble wrap and protect glass surfaces with cardboard.",
    "side table": "Wrap in moving blankets. Remove drawers and pack separately.",
    "ottoman": "Use as extra packing space - fill hollow ottomans with small items.",
    "footstool": "Wrap in moving blanket. Light enough to stack on top of boxes.",
    "tv": "Wrap screen in bubble wrap, transport upright. Use original box if available.",
    "remote": "Pack in a ziplock bag and label. Keep with TV accessories.",
    "clock": "Remove batteries. Wrap in bubble wrap and pack in a small box.",
    "lamp": "Remove shade and bulb. Wrap base and shade separately in tissue paper.",
    "potted plant": "Water 2 days before. Wrap pot in plastic to prevent soil spills.",
    "vase": "Fill inside with packing paper. Wrap in bubble wrap and pack upright.",
    "painting": "Use corner protectors. Wrap in bubble wrap and transport upright.",
    "mirror": "Make an X with masking tape on glass. Wrap in bubble wrap and cardboard.",
    "statue": "Wrap in bubble wrap. Pack tightly in a box with crumpled paper fill.",
    "candle": "Pack flat in a cool box. Heat can cause warping during transport.",
    "curtains": "Fold neatly and pack in a large bag. Label rod hardware separately.",
    "blinds": "Remove and roll carefully. Secure with rubber bands. Label each window.",
    "bed": "Disassemble frame, label all bolts in a ziplock bag. Wrap mattress in plastic cover.",
    "cabinet": "Empty completely. Secure doors with tape. Transport upright.",
    "dresser": "Remove drawers and wrap separately. Use drawers to pack clothes.",
    "drawer": "Can be used to transport clothes as-is. Wrap with shrink wrap to keep closed.",
    "shelf": "Disassemble if possible. Wrap shelves in bubble wrap. Label all hardware.",
    "pillow": "Use garbage bags as pillow covers. Great for filling gaps in the truck.",
    "cloth": "Pack in vacuum bags to save space. Use as padding for fragile items.",
    "towel": "Use as padding for fragile items. Pack in garbage bags to keep clean.",
    "laundry hamper": "Use as a container - fill with soft items like clothes or linens.",
    "teddy bear": "Pack in a large garbage bag. Use to fill gaps between boxes.",
    "safe": "Do NOT pack on top of other items. Must be loaded first due to weight.",
    "box": "Seal with packing tape. Label contents and destination room clearly.",
    "refrigerator": "Defrost and drain 24 hrs before. Tape doors shut. Transport upright only.",
    "microwave": "Pack turntable separately in bubble wrap. Tape door shut.",
    "toaster": "Clean crumbs. Wrap in bubble wrap. Pack with kitchen appliances.",
    "coffee machine": "Drain water tank. Remove detachable parts and wrap separately.",
    "kettle": "Empty and dry completely. Wrap in bubble wrap. Pack with kitchen items.",
    "dining table": "Remove legs if possible. Wrap top in moving blankets. Protect edges.",
    "plate": "Pack vertically (like records) with paper between each. Never stack flat.",
    "pan": "Stack with paper between each. Wrap handles in bubble wrap.",
    "pot": "Nest smaller pots inside larger ones with paper padding between.",
    "mug": "Wrap each mug individually. Pack upside down in a divided box.",
    "bowl": "Pack vertically with paper between each. Use a sturdy box.",
    "cup": "Wrap each cup individually. Fill insides with crumpled paper.",
    "bottle": "Wrap in bubble wrap. Pack upright in a box with dividers.",
    "fork": "Bundle in groups with rubber bands. Wrap bundles in paper.",
    "knife": "Use blade guards or wrap blades in cardboard secured with tape.",
    "spoon": "Bundle in groups with rubber bands. Pack with cutlery.",
    "garbage can": "Clean thoroughly. Can be used as a container for long items.",
    "watering can": "Empty completely. Pack with garden items.",
    "tissue box": "No special packing needed. Use to fill empty spaces in boxes.",
    "desk": "Remove drawers. Disassemble legs if possible. Wrap top in blankets.",
    "laptop": "Back up data. Transport in padded laptop bag. Keep as carry-on.",
    "book": "Pack in small boxes (books are heavy). Alternate spine direction.",
    "newspaper": "Bundle and recycle, or use as packing paper for fragile items.",
    "pen": "Pack in a ziplock bag or pencil case with other stationery.",
    "pencil": "Pack in a ziplock bag or pencil case with other stationery.",
    "cell phone": "Keep with you. Do not pack in the truck.",
}


def get_packing_tip(item_name):
    """Return a packing tip for the given item, or a generic tip."""
    name = normalize_item_name(item_name)
    return PACKING_TIPS.get(name, "Wrap securely in packing paper and place in a labeled box.")
