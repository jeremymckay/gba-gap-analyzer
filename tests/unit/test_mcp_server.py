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

import pytest
from mcp_server import (
    normalize_category,
    extract_actual_attributes,
    fetch_possible_attributes,
    fetch_actual_attributes,
    fetch_attribute_gap,
)


def test_normalize_category():
    assert normalize_category("Coffee Shop") == "coffee_shop"
    assert normalize_category("cafe") == "coffee_shop"
    assert normalize_category("Restaurant") == "restaurant"
    assert normalize_category("Unknown Category") == "unknown_category"


def test_extract_actual_attributes():
    test_data = {
        "dineIn": True,
        "servesCoffee": True,
        "paymentOptions": {
            "acceptsCreditCards": True,
            "acceptsDebitCards": False,
        },
        "accessibilityOptions": {
            "wheelchairAccessibleEntrance": True,
        },
    }
    attrs = extract_actual_attributes(test_data)
    assert "dineIn" in attrs
    assert "servesCoffee" in attrs
    assert "acceptsCreditCards" in attrs
    assert "wheelchairAccessibleEntrance" in attrs
    assert "acceptsDebitCards" not in attrs


def test_fetch_possible_attributes():
    coffee_attrs = fetch_possible_attributes("Coffee Shop")
    assert len(coffee_attrs) == 15
    assert "servesCoffee" in coffee_attrs

    unknown_attrs = fetch_possible_attributes("Unknown Category")
    assert "wheelchairAccessibleEntrance" in unknown_attrs


@pytest.mark.asyncio
async def test_fetch_actual_attributes_mock():
    # Test using the mock place id
    attrs = await fetch_actual_attributes("mock_downtown_cafe")
    assert len(attrs) == 3
    assert "servesCoffee" in attrs
    assert "acceptsCreditCards" in attrs
    assert "dineIn" in attrs


@pytest.mark.asyncio
async def test_fetch_attribute_gap_mock():
    gap = await fetch_attribute_gap("mock_downtown_cafe", "Coffee Shop")
    assert "actual_attributes" in gap
    assert "possible_attributes" in gap
    assert "missing_attributes" in gap

    assert len(gap["actual_attributes"]) == 3
    assert len(gap["possible_attributes"]) == 15
    assert len(gap["missing_attributes"]) == 12
    assert "outdoorSeating" in gap["missing_attributes"]
    assert "servesCoffee" not in gap["missing_attributes"]
