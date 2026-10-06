"""Size extraction cases taken from real retailer names. Run: python -m pytest tests"""

import pytest

from map_products import extract_size

CASES = [
    # weight / volume, unit spellings
    ("Dates Sukkari Appx 500g", (500.0, "g", None)),
    ("FOODWAY BROCCOLI FLORETS-250GM", (250.0, "g", None)),
    ("KARMA BLACK RAISIN 500GMS", (500.0, "g", None)),
    ("PEAS MANGETUOT 250 G PAK", (250.0, "g", None)),
    ("GARLIC 20X100 GRAM FL", (100.0, "g", 20.0)),
    ("POTATOES RUSSET ORGANIC BAGS 2.27 KG", (2.27, "kg", None)),
    ("Icld F/Fries Ctd Thn&Crsp1.25k", (1.25, "kg", None)),
    ("OKRA LADY FINGER .5 KG", (0.5, "kg", None)),
    ("Org.Baby Spinach Clamshell 5oz", (5.0, "oz", None)),
    ("Alx.Cut Fries W/SeaSalt 1.75lb", (1.75, "lb", None)),
    ("STRAWBERRY DRISCOLL USA 1PINT", (1.0, "pint", None)),
    ("APPLE JUICE 500 ML", (500.0, "ml", None)),
    ("CARROT FRESH JUICE 1 LTR", (1.0, "l", None)),
    # glued to words, abbreviation dots, retailer tags
    ("DriscollsBlueberryPortugal125g", (125.0, "g", None)),
    ("Plums Pkt Appx.500gm", (500.0, "g", None)),
    ("Sadia F/Fries6mm E.Crspy 1kgPO", (1.0, "kg", None)),
    ("ROYAL BABY TURNIP200GC", (200.0, "g", None)),
    ("STRAWBERRY GARIGUETTE 250GW", (250.0, "g", None)),
    # multipacks
    ("Lulu Cut Green Beans 3x400g PO", (400.0, "g", 3.0)),
    ("Fani Frz Mix Veg.4way 2x400gPO", (400.0, "g", 2.0)),
    ("Faani Frz.Tapioca SmlCut2x700g", (700.0, "g", 2.0)),
    ("Lulu Molokhia 400g 3s P/O", (400.0, "g", 3.0)),
    ("Sunbulah Sweet Corn 450gm 2's", (450.0, "g", 2.0)),
    ("Mima Molokhia Minced 400gX5 PO", (400.0, "g", 5.0)),
    ("MimaGdn Corn OnThe Cob 4s 950g", (950.0, "g", None)),
    # counts
    ("CAPSICUM MIX 3PCS", (3.0, "pcs", None)),
    ("APPLE ROCKIT (2 PCS)", (2.0, "pcs", None)),
    ("ORGANIC AVOCADO HASS 2PCC", (2.0, "pcs", None)),
    ("Rambutan 1pkt", (1.0, "pcs", None)),
    ("Sweet Corn Ready-to-Eat 1Cob", (1.0, "pcs", None)),
    ("Emborg Corn On The Cob-6 6's", (6.0, "pcs", None)),
    ("NECTARINE 1X6 IRAN", (6.0, "pcs", 1.0)),
    ("Apple Rockit Pack 5x2Pc", (2.0, "pcs", 5.0)),
    ("Little Gem Lettuce x2", (2.0, "pcs", None)),
    # numbers that are not sizes
    ("MANGO R2E2 THAILAND", (None, None, None)),
    ("CAPSICUMS RED 70/90 FL", (None, None, None)),
    ("AVOCADO HASS 14-16 READY TO EAT FL", (None, None, None)),
    ("ASPARAGUS GREEN JUMBO (20-28 MM) FL", (None, None, None)),
    ("Ltsa Steak Fries 10/18mm 1 kg", (1.0, "kg", None)),
    ("CAPSICUM 4 COLOR IMP", (None, None, None)),
    ("APRICOT DRIED 100% ORGANIC", (None, None, None)),
    ("TOMATO QATAR GRADE 1", (None, None, None)),
    ("POTATOES RUSSET WASHINGTON 70S", (None, None, None)),
    ("MUSHROOM CULTIVTED MIX 4X250", (None, None, None)),
    ("DATES 500 PACK TUNIS", (None, None, None)),
    ("GOLDEN APPLE CASE PRM W/COVER CAL.6", (None, None, None)),
    ("Fresh Green Almonds", (None, None, None)),
]


@pytest.mark.parametrize("name, expected", CASES)
def test_extract_size(name, expected):
    assert extract_size(name) == expected
