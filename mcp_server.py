# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import logging
from typing import List, Dict, Any, Optional
import httpx
from mcp.server.fastmcp import FastMCP

# Setup stderr logging to prevent interfering with JSON-RPC over stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("gba_gap_analyzer_mcp")

# Load .env file manually if it exists to ensure isolated subprocesses have credentials
current_dir = os.path.dirname(os.path.abspath(__file__))
for path in [".env", os.path.join(current_dir, ".env"), os.path.join(os.path.dirname(current_dir), ".env")]:
    if os.path.exists(path):
        try:
            logger.info(f"Manually loading environment variables from {path}")
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        val = v.strip().strip('"').strip("'")
                        os.environ[k.strip()] = val
            break
        except Exception as e:
            logger.error(f"Error reading .env at {path}: {str(e)}")

# Initialize the FastMCP server
mcp = FastMCP("GBA Gap Analyzer Places Server")

# Mock data definitions to support testing and offline development
MOCK_PLACE_ID = "mock_downtown_cafe"
MOCK_PLACE_DATA = {
    "id": MOCK_PLACE_ID,
    "displayName": {"text": "Downtown Cafe"},
    "primaryType": "coffee_shop",
    "types": ["coffee_shop", "cafe", "food", "establishment"],
    "outdoorSeating": False,
    "dineIn": True,
    "servesCoffee": True,
    "paymentOptions": {
        "acceptsCreditCards": True,
        "acceptsDebitCards": False,
        "acceptsCashOnly": False,
        "acceptsNfc": False,
    },
    "accessibilityOptions": {
        "wheelchairAccessibleEntrance": False,
        "wheelchairAccessibleParking": False,
        "wheelchairAccessibleRestroom": False,
        "wheelchairAccessibleSeating": False,
    },
}

# Predefined attributes that can be used for gap analysis per business type
# Maps normalized category names/types to their applicable possible attributes
POSSIBLE_ATTRIBUTES_MAP = {
    "coffee_shop": [
        "outdoorSeating",
        "wheelchairAccessibleEntrance",
        "wheelchairAccessibleParking",
        "wheelchairAccessibleRestroom",
        "wheelchairAccessibleSeating",
        "acceptsCreditCards",
        "acceptsDebitCards",
        "acceptsCashOnly",
        "acceptsNfc",
        "servesBreakfast",
        "servesBrunch",
        "servesCoffee",
        "servesDessert",
        "goodForKids",
        "dineIn",
    ],
    "cafe": [
        "outdoorSeating",
        "wheelchairAccessibleEntrance",
        "wheelchairAccessibleParking",
        "wheelchairAccessibleRestroom",
        "wheelchairAccessibleSeating",
        "acceptsCreditCards",
        "acceptsDebitCards",
        "acceptsCashOnly",
        "acceptsNfc",
        "servesBreakfast",
        "servesBrunch",
        "servesCoffee",
        "servesDessert",
        "goodForKids",
        "dineIn",
    ],
    "restaurant": [
        "outdoorSeating",
        "wheelchairAccessibleEntrance",
        "wheelchairAccessibleParking",
        "wheelchairAccessibleRestroom",
        "wheelchairAccessibleSeating",
        "acceptsCreditCards",
        "acceptsDebitCards",
        "acceptsCashOnly",
        "acceptsNfc",
        "servesBreakfast",
        "servesBrunch",
        "servesLunch",
        "servesDinner",
        "servesCoffee",
        "servesDessert",
        "servesBeer",
        "servesWine",
        "servesCocktails",
        "servesVegetarianFood",
        "goodForKids",
        "goodForGroups",
        "dineIn",
        "takeout",
        "delivery",
        "reservable",
    ],
}

# Default attributes used when a category is not recognized
DEFAULT_POSSIBLE_ATTRIBUTES = [
    "wheelchairAccessibleEntrance",
    "wheelchairAccessibleParking",
    "wheelchairAccessibleRestroom",
    "wheelchairAccessibleSeating",
    "acceptsCreditCards",
    "acceptsDebitCards",
    "acceptsCashOnly",
    "acceptsNfc",
]


def normalize_category(category: str) -> str:
    """Normalizes category/type string to match POSSIBLE_ATTRIBUTES_MAP keys."""
    normalized = category.lower().strip().replace(" ", "_").replace("-", "_")
    # Handle common synonyms
    if "coffee" in normalized or "cafe" in normalized:
        return "coffee_shop"
    return normalized


def extract_actual_attributes(data: dict) -> List[str]:
    """Helper to extract active boolean attributes from Google Places API response."""
    actual = []

    # Check top-level boolean fields
    boolean_fields = [
        "outdoorSeating",
        "goodForKids",
        "goodForGroups",
        "goodForWatchingSports",
        "liveMusic",
        "menuForChildren",
        "servesBeer",
        "servesBreakfast",
        "servesBrunch",
        "servesCocktails",
        "servesCoffee",
        "servesDessert",
        "servesDinner",
        "servesLunch",
        "servesVegetarianFood",
        "servesWine",
        "takeout",
        "delivery",
        "dineIn",
        "reservable",
    ]
    for field in boolean_fields:
        if data.get(field) is True:
            actual.append(field)

    # Check accessibilityOptions
    accessibility = data.get("accessibilityOptions", {})
    accessibility_fields = [
        "wheelchairAccessibleParking",
        "wheelchairAccessibleEntrance",
        "wheelchairAccessibleRestroom",
        "wheelchairAccessibleSeating",
    ]
    for field in accessibility_fields:
        if accessibility.get(field) is True:
            actual.append(field)

    # Check paymentOptions
    payment = data.get("paymentOptions", {})
    payment_fields = [
        "acceptsCreditCards",
        "acceptsDebitCards",
        "acceptsCashOnly",
        "acceptsNfc",
    ]
    for field in payment_fields:
        if payment.get(field) is True:
            actual.append(field)

    return actual


async def fetch_place_details_from_api(
    place_id: str, api_key: str
) -> Optional[dict]:
    """Makes a live call to the Google Places API (New) Place Details endpoint."""
    url = f"https://places.googleapis.com/v1/places/{place_id}"
    fields = [
        "id",
        "displayName",
        "primaryType",
        "types",
        "accessibilityOptions",
        "paymentOptions",
        "outdoorSeating",
        "goodForKids",
        "goodForGroups",
        "goodForWatchingSports",
        "liveMusic",
        "menuForChildren",
        "servesBeer",
        "servesBreakfast",
        "servesBrunch",
        "servesCocktails",
        "servesCoffee",
        "servesDessert",
        "servesDinner",
        "servesLunch",
        "servesVegetarianFood",
        "servesWine",
        "takeout",
        "delivery",
        "dineIn",
        "reservable",
    ]
    field_mask = ",".join(fields)

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": field_mask,
    }

    logger.info(f"Calling Google Places API for place_id: {place_id}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(
                    f"API returned status {response.status_code}: {response.text}"
                )
                return None
        except Exception as e:
            logger.error(f"Error querying Places API: {str(e)}")
            return None


@mcp.tool()
async def fetch_actual_attributes(
    place_id: str, api_key_override: Optional[str] = None
) -> List[str]:
    """Fetches the Google Place profile's current active attributes.

    Args:
        place_id: The Google Place ID for the business (e.g. 'ChIJ531sN3qGj4ARz8vbe3I9Hsw').
        api_key_override: Optional API key override if not using the environment variable.

    Returns:
        A list of strings representing the active attributes on the place's profile.
    """
    # 1. Use Mock Mode if explicitly requested or if it's the mock place ID
    if place_id == MOCK_PLACE_ID:
        logger.info("Serving mock data for Downtown Cafe")
        return extract_actual_attributes(MOCK_PLACE_DATA)

    # 2. Get API key from parameter or environment variables
    api_key = api_key_override or os.getenv("PLACES_API_KEY") or os.getenv("GOOGLE_PLACES_API_KEY")

    if not api_key or api_key.lower() == "mock":
        logger.warning(
            f"No Places API key configured. Falling back to mock data for Place ID: {place_id}"
        )
        # Create a basic mock listing if live credentials aren't available
        fallback_mock = dict(MOCK_PLACE_DATA)
        fallback_mock["id"] = place_id
        return extract_actual_attributes(fallback_mock)

    # 3. Live API call
    api_data = await fetch_place_details_from_api(place_id, api_key)
    if api_data:
        return extract_actual_attributes(api_data)
    else:
        logger.warning(
            f"Live API call failed or returned empty. Falling back to mock data."
        )
        return extract_actual_attributes(MOCK_PLACE_DATA)


@mcp.tool()
def fetch_possible_attributes(primary_category: str) -> List[str]:
    """Returns the list of possible attributes for a business category.

    Args:
        primary_category: The primary business category (e.g. 'Coffee Shop', 'Restaurant', 'Cafe').

    Returns:
        A list of strings representing all possible attributes for that category.
    """
    normalized = normalize_category(primary_category)

    # Check if category is hardcoded in map
    if normalized in POSSIBLE_ATTRIBUTES_MAP:
        attributes = POSSIBLE_ATTRIBUTES_MAP[normalized]
        logger.info(
            f"Fetched {len(attributes)} possible attributes for category '{primary_category}' (hardcoded map)"
        )
        return attributes

    # Start with default baseline (Accessibility + Payments)
    attributes = list(DEFAULT_POSSIBLE_ATTRIBUTES)

    # 1. Food, Beverage, and Dining establishments
    food_keywords = [
        "restaurant",
        "cafe",
        "coffee",
        "food",
        "bakery",
        "bar",
        "pub",
        "bistro",
        "diner",
        "eatery",
        "fast_food",
        "ice_cream",
        "juice",
        "sandwich",
        "pizza",
        "steakhouse",
        "sushi",
        "takeaway",
        "delivery",
        "buffet",
        "cafeteria",
    ]
    if any(kw in normalized for kw in food_keywords):
        dining_attributes = [
            "dineIn",
            "takeout",
            "delivery",
            "outdoorSeating",
            "servesCoffee",
            "servesDessert",
            "goodForKids",
            "goodForGroups",
            "reservable",
        ]
        for attr in dining_attributes:
            if attr not in attributes:
                attributes.append(attr)

    # 2. Specifically Bars, Pubs, Nightclubs, Lounges
    alcohol_keywords = [
        "bar",
        "pub",
        "nightclub",
        "lounge",
        "brewery",
        "winery",
        "tavern",
    ]
    if any(kw in normalized for kw in alcohol_keywords):
        alcohol_attributes = [
            "servesBeer",
            "servesWine",
            "servesCocktails",
            "liveMusic",
        ]
        for attr in alcohol_attributes:
            if attr not in attributes:
                attributes.append(attr)

    # 3. Retail Stores & Shops
    store_keywords = [
        "store",
        "shop",
        "market",
        "supermarket",
        "pharmacy",
        "grocer",
        "retail",
        "mall",
        "boutique",
        "bookstore",
        "florist",
        "gift",
        "clothing",
        "apparel",
    ]
    if any(kw in normalized for kw in store_keywords):
        retail_attributes = ["takeout"]  # represents store pickup / curbside pickup
        for attr in retail_attributes:
            if attr not in attributes:
                attributes.append(attr)

    logger.info(
        f"Fetched {len(attributes)} possible attributes dynamically for category '{primary_category}' (normalized: '{normalized}')"
    )
    return attributes


@mcp.tool()
async def fetch_attribute_gap(
    place_id: str,
    primary_category: str,
    api_key_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetches both actual and possible attributes, returning the attributes and the gap analysis.

    Args:
        place_id: The Google Place ID for the business.
        primary_category: The primary business category (e.g. 'Coffee Shop', 'Restaurant').
        api_key_override: Optional API key override if not using the environment variable.

    Returns:
        A dictionary containing actual attributes, possible attributes, and missing attributes.
    """
    actual = await fetch_actual_attributes(place_id, api_key_override)
    possible = fetch_possible_attributes(primary_category)

    # Determine missing opportunities
    missing = [attr for attr in possible if attr not in actual]

    return {
        "actual_attributes": actual,
        "possible_attributes": possible,
        "missing_attributes": missing,
    }


if __name__ == "__main__":
    # Start the MCP server (defaults to stdio)
    mcp.run()
