from copy import deepcopy
from pathlib import Path

import yaml


DEFAULT_PRICING_CONFIG = {
    "currency": "INR",
    "trucks": {
        "Tata Ace": {
            "capacity_cuft": 150,
            "base_cost": 8000,
            "cost_per_km": 12,
        },
        "Tata 407": {
            "capacity_cuft": 400,
            "base_cost": 15000,
            "cost_per_km": 18,
        },
        "Eicher Pro 2049 Plus": {
            "capacity_cuft": 600,
            "base_cost": 22000,
            "cost_per_km": 25,
        },
        "Eicher Pro 2110 7S": {
            "capacity_cuft": 900,
            "base_cost": 30000,
            "cost_per_km": 32,
        },
    },
    "packing_cost_per_cuft": 12,
    "loading_base": 2000,
    "fragile_surcharge": 300,
    "floor_surcharge_per_floor": 500,
    "extra_box": {
        "cost": 200,
        "volume_cuft": 4,
        "weight_kg": {
            "light": 8,
            "heavy": 20,
        },
    },
    "quote_range": {
        "lower_multiplier": 0.90,
        "upper_multiplier": 1.15,
        "round_to": 100,
    },
    "city_distances_km": {
        "Delhi|Mumbai": 1400,
        "Delhi|Bangalore": 2150,
        "Delhi|Chennai": 2200,
        "Delhi|Hyderabad": 1600,
        "Delhi|Pune": 1450,
        "Delhi|Kolkata": 1500,
        "Mumbai|Bangalore": 980,
        "Mumbai|Chennai": 1330,
        "Mumbai|Hyderabad": 710,
        "Mumbai|Pune": 150,
        "Mumbai|Kolkata": 2050,
        "Bangalore|Chennai": 350,
        "Bangalore|Hyderabad": 570,
        "Chennai|Hyderabad": 630,
        "Kolkata|Hyderabad": 1500,
    },
}


def _deep_merge(base, override):
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_pricing_config(config_path=None):
    path = Path(config_path or Path(__file__).with_name("pricing_config.yaml"))
    if not path.exists():
        return deepcopy(DEFAULT_PRICING_CONFIG)

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return _deep_merge(DEFAULT_PRICING_CONFIG, loaded)


def get_truck_options(config_path=None):
    config = load_pricing_config(config_path)
    trucks = []
    for name, data in config["trucks"].items():
        trucks.append({
            "name": name,
            "capacity": data["capacity_cuft"],
            "base_cost": data["base_cost"],
        })
    return trucks


def get_truck_pricing(truck_name, config=None):
    pricing = config or load_pricing_config()
    trucks = pricing["trucks"]
    if truck_name in trucks:
        return trucks[truck_name]
    return next(iter(trucks.values()))


def get_distance(origin, destination, config_path=None):
    if origin == destination:
        return 1

    distances = load_pricing_config(config_path)["city_distances_km"]
    key1 = f"{origin}|{destination}"
    key2 = f"{destination}|{origin}"
    return distances.get(key1, distances.get(key2, 500))


def round_to_nearest(value, round_to):
    return round(value / round_to) * round_to
