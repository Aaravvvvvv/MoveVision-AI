# quote.py
# Calculates moving cost estimates from inventory summaries and pricing_config.yaml.

from pricing import (
    get_distance,
    get_truck_pricing,
    load_pricing_config,
    round_to_nearest,
)


def calculate_quote(
    summary,
    origin_city,
    destination_city,
    distance_km,
    floor_number=0,
    has_lift=True,
    extra_boxes=0,
    box_type="light",
    pricing_config_path=None,
):
    """
    Calculate a full moving cost estimate.

    Pricing values are loaded from pricing_config.yaml. If the config file is
    missing or incomplete, pricing.py supplies default fallback values.
    """
    pricing = load_pricing_config(pricing_config_path)

    truck = summary["truck"]
    truck_name = truck["name"]
    truck_pricing = get_truck_pricing(truck_name, pricing)

    base_cost = truck_pricing["base_cost"]
    transport_cost = distance_km * truck_pricing["cost_per_km"]
    packing_cost = summary["total_volume"] * pricing["packing_cost_per_cuft"]
    loading_cost = pricing["loading_base"]

    fragile_count = sum(
        item["Count"]
        for item in summary["items"]
        if item["Fragility"] == "Fragile"
    )
    fragile_cost = fragile_count * pricing["fragile_surcharge"]

    floor_cost = 0
    if not has_lift and floor_number > 0:
        floor_cost = floor_number * pricing["floor_surcharge_per_floor"]

    box_config = pricing["extra_box"]
    box_volume = extra_boxes * box_config["volume_cuft"]
    box_weight = extra_boxes * box_config["weight_kg"].get(
        box_type,
        box_config["weight_kg"]["light"],
    )
    box_cost = extra_boxes * box_config["cost"]

    total_volume = summary["total_volume"] + box_volume
    total_weight = summary["total_weight"] + box_weight

    subtotal = (
        base_cost
        + transport_cost
        + packing_cost
        + loading_cost
        + fragile_cost
        + floor_cost
        + box_cost
    )

    quote_range = pricing["quote_range"]
    round_to = quote_range["round_to"]
    lower_estimate = round_to_nearest(subtotal * quote_range["lower_multiplier"], round_to)
    upper_estimate = round_to_nearest(subtotal * quote_range["upper_multiplier"], round_to)

    return {
        "origin": origin_city,
        "destination": destination_city,
        "distance_km": distance_km,
        "truck": truck_name,
        "total_volume": round(total_volume, 1),
        "total_weight": round(total_weight, 1),
        "fragile_items": fragile_count,
        "extra_boxes": extra_boxes,
        "currency": pricing.get("currency", "INR"),
        "base_cost": base_cost,
        "transport_cost": round(transport_cost),
        "packing_cost": round(packing_cost),
        "loading_cost": loading_cost,
        "fragile_cost": fragile_cost,
        "floor_cost": floor_cost,
        "box_cost": box_cost,
        "lower_estimate": lower_estimate,
        "upper_estimate": upper_estimate,
        "midpoint": round_to_nearest((lower_estimate + upper_estimate) / 2, round_to),
    }
