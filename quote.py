# quote.py
# Calculates moving cost estimate based on detected items + move details

# Cost per km based on truck type
COST_PER_KM = {
    "Tata Ace":             12,
    "Tata 407":             18,
    "Eicher Pro 2049 Plus": 25,
    "Eicher Pro 2110 7S":   32,
}

# Packing material cost per cubic foot
PACKING_COST_PER_CUFT = 12

# Loading/unloading base cost
LOADING_BASE = 2000

# Extra cost per fragile item
FRAGILE_SURCHARGE = 300

# Floor surcharge if no lift (per floor above ground)
FLOOR_SURCHARGE = 500

# Extra carton box cost
COST_PER_EXTRA_BOX = 200

def calculate_quote(
    summary,
    origin_city,
    destination_city,
    distance_km,
    floor_number=0,
    has_lift=True,
    extra_boxes=0,
    box_type="light"
):
    """
    Calculate full moving cost estimate.

    Args:
        summary        : output from estimator.summarise_items()
        origin_city    : string
        destination_city: string
        distance_km    : distance between cities in km
        floor_number   : floor at origin (0 = ground)
        has_lift       : whether lift is available
        extra_boxes    : number of additional packed boxes user expects
        box_type       : 'light' or 'heavy' for extra boxes

    Returns:
        quote dict with itemised costs and total range
    """

    truck       = summary["truck"]
    truck_name  = truck["name"]
    base_cost   = truck["base_cost"]

    # 1. Transport cost based on distance
    transport_cost = distance_km * COST_PER_KM.get(truck_name, 18)

    # 2. Packing material cost
    packing_cost = summary["total_volume"] * PACKING_COST_PER_CUFT

    # 3. Loading and unloading
    loading_cost = LOADING_BASE

    # 4. Fragile item surcharge
    fragile_count = sum(
        item["Count"]
        for item in summary["items"]
        if item["Fragility"] == "Fragile"
    )
    fragile_cost = fragile_count * FRAGILE_SURCHARGE

    # 5. Floor surcharge if no lift
    floor_cost = 0
    if not has_lift and floor_number > 0:
        floor_cost = floor_number * FLOOR_SURCHARGE

    # 6. Extra boxes
    box_volume = extra_boxes * 4   # 4 cu ft per box
    box_weight = extra_boxes * (20 if box_type == "heavy" else 8)
    box_cost   = extra_boxes * COST_PER_EXTRA_BOX

    # Add extra box volume/weight to summary totals
    total_volume = summary["total_volume"] + box_volume
    total_weight = summary["total_weight"] + box_weight

    # 7. Total calculation
    subtotal = (
        base_cost +
        transport_cost +
        packing_cost +
        loading_cost +
        fragile_cost +
        floor_cost +
        box_cost
    )

    # Add 15% buffer for range (lower and upper estimate)
    lower_estimate = round(subtotal * 0.90 / 100) * 100   # round to nearest 100
    upper_estimate = round(subtotal * 1.15 / 100) * 100

    return {
        "origin":           origin_city,
        "destination":      destination_city,
        "distance_km":      distance_km,
        "truck":            truck_name,
        "total_volume":     round(total_volume, 1),
        "total_weight":     round(total_weight, 1),
        "fragile_items":    fragile_count,
        "extra_boxes":      extra_boxes,

        # Itemised costs
        "base_cost":        base_cost,
        "transport_cost":   round(transport_cost),
        "packing_cost":     round(packing_cost),
        "loading_cost":     loading_cost,
        "fragile_cost":     fragile_cost,
        "floor_cost":       floor_cost,
        "box_cost":         box_cost,

        # Final quote
        "lower_estimate":   lower_estimate,
        "upper_estimate":   upper_estimate,
        "midpoint":         round((lower_estimate + upper_estimate) / 2 / 100) * 100,
    }


# Common Indian city distances in km
CITY_DISTANCES = {
    ("Delhi",     "Mumbai"):    1400,
    ("Delhi",     "Bangalore"): 2150,
    ("Delhi",     "Chennai"):   2200,
    ("Delhi",     "Hyderabad"): 1600,
    ("Delhi",     "Pune"):      1450,
    ("Delhi",     "Kolkata"):   1500,
    ("Mumbai",    "Bangalore"): 980,
    ("Mumbai",    "Chennai"):   1330,
    ("Mumbai",    "Hyderabad"): 710,
    ("Mumbai",    "Pune"):      150,
    ("Mumbai",    "Kolkata"):   2050,
    ("Bangalore", "Chennai"):   350,
    ("Bangalore", "Hyderabad"): 570,
    ("Chennai",   "Hyderabad"): 630,
    ("Kolkata",   "Hyderabad"): 1500,
}

def get_distance(origin, destination):
    """Look up distance between two Indian cities."""
    key1 = (origin, destination)
    key2 = (destination, origin)

    if key1 in CITY_DISTANCES:
        return CITY_DISTANCES[key1]
    elif key2 in CITY_DISTANCES:
        return CITY_DISTANCES[key2]
    else:
        # Default fallback if city pair not in table
        return 500
