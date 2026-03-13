"""
Retail Brain — Sainsbury's Demo Data Generator
Generates a realistic Sainsbury's-branded dataset using real product names,
UK seasonal demand patterns, Nectar promotions, and Taste the Difference SKUs.

Data covers: Q4 2024 (Oct 1 – Dec 31) — peak UK trading period.
Run: python scripts/generate_sainsburys_data.py
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import date, timedelta

random.seed(2024)
np.random.seed(2024)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Real Sainsbury's Products Catalogue
# Format: (sku, name, category, tier, unit_price, base_demand, reorder_point, lead_days)
# Tiers: "So Good" (economy), "Sainsbury's" (standard), "Taste the Difference" (premium)
# base_demand = realistic daily unit sales for a mid-size UK store
# ─────────────────────────────────────────────────────────────────────────────
SAINSBURYS_PRODUCTS = [
    # ── Dairy & Eggs ──────────────────────────────────────────────────────────
    ("SAI-D001", "Sainsbury's British Whole Milk 6 Pints",         "Dairy & Eggs",    "Sainsbury's",          1.45,  85, 60, 1),
    ("SAI-D002", "Sainsbury's Semi-Skimmed Milk 4 Pints",          "Dairy & Eggs",    "Sainsbury's",          1.10,  70, 50, 1),
    ("SAI-D003", "Sainsbury's Skimmed Milk 2 Pints",               "Dairy & Eggs",    "Sainsbury's",          0.80,  40, 30, 1),
    ("SAI-D004", "Sainsbury's Large Free Range Eggs 12pk",         "Dairy & Eggs",    "Sainsbury's",          3.25,  55, 40, 1),
    ("SAI-D005", "Taste the Difference Burford Brown Eggs 6pk",    "Dairy & Eggs",    "Taste the Difference", 3.75,  22, 15, 2),
    ("SAI-D006", "Sainsbury's Salted Butter 250g",                 "Dairy & Eggs",    "Sainsbury's",          1.60,  38, 28, 2),
    ("SAI-D007", "Taste the Difference Normandy Butter 250g",      "Dairy & Eggs",    "Taste the Difference", 2.20,  18, 12, 2),
    ("SAI-D008", "Sainsbury's Mature Cheddar 400g",                "Dairy & Eggs",    "Sainsbury's",          3.50,  45, 32, 3),
    ("SAI-D009", "Taste the Difference West Country Cheddar 300g", "Dairy & Eggs",    "Taste the Difference", 4.00,  20, 14, 3),
    ("SAI-D010", "Sainsbury's Greek Style Yogurt 500g",            "Dairy & Eggs",    "Sainsbury's",          1.50,  30, 22, 2),

    # ── Fresh Bakery ──────────────────────────────────────────────────────────
    ("SAI-B001", "Sainsbury's Medium Sliced White Bread 800g",     "Fresh Bakery",    "Sainsbury's",          1.10,  90, 70, 1),
    ("SAI-B002", "Sainsbury's Wholemeal Bread 800g",               "Fresh Bakery",    "Sainsbury's",          1.20,  55, 42, 1),
    ("SAI-B003", "Taste the Difference Sourdough Loaf 400g",       "Fresh Bakery",    "Taste the Difference", 2.75,  30, 22, 1),
    ("SAI-B004", "Sainsbury's Croissants 4pk",                     "Fresh Bakery",    "Sainsbury's",          1.40,  40, 30, 1),
    ("SAI-B005", "Taste the Difference All Butter Croissants 4pk", "Fresh Bakery",    "Taste the Difference", 2.50,  25, 18, 1),
    ("SAI-B006", "Sainsbury's Bagels 5pk",                         "Fresh Bakery",    "Sainsbury's",          1.30,  28, 20, 1),
    ("SAI-B007", "Sainsbury's Hot Cross Buns 6pk",                 "Fresh Bakery",    "Sainsbury's",          1.25,  35, 25, 1),

    # ── Meat & Fish ───────────────────────────────────────────────────────────
    ("SAI-M001", "Sainsbury's British Chicken Breast Fillets 600g","Meat & Fish",     "Sainsbury's",          4.50,  52, 38, 2),
    ("SAI-M002", "Taste the Difference Corn Fed Chicken 1.5kg",    "Meat & Fish",     "Taste the Difference", 7.00,  18, 12, 2),
    ("SAI-M003", "Sainsbury's British Beef Mince 500g (20% fat)",  "Meat & Fish",     "Sainsbury's",          3.75,  45, 32, 2),
    ("SAI-M004", "Taste the Difference Scottish Beef Mince 500g",  "Meat & Fish",     "Taste the Difference", 5.00,  20, 14, 2),
    ("SAI-M005", "Sainsbury's Pork Sausages 8pk 454g",             "Meat & Fish",     "Sainsbury's",          2.50,  40, 28, 2),
    ("SAI-M006", "Taste the Difference Cumberland Sausages 400g",  "Meat & Fish",     "Taste the Difference", 3.50,  22, 15, 2),
    ("SAI-M007", "Sainsbury's Smoked Back Bacon 300g",             "Meat & Fish",     "Sainsbury's",          2.75,  35, 25, 2),
    ("SAI-M008", "Sainsbury's Atlantic Salmon Fillets 240g",       "Meat & Fish",     "Sainsbury's",          5.50,  25, 18, 2),
    ("SAI-M009", "Taste the Difference Scottish Salmon Fillet 2pk","Meat & Fish",     "Taste the Difference", 7.00,  15, 10, 2),
    ("SAI-M010", "Sainsbury's Cod Fillets 420g",                   "Meat & Fish",     "Sainsbury's",          4.75,  20, 14, 3),

    # ── Fresh Produce ─────────────────────────────────────────────────────────
    ("SAI-P001", "Sainsbury's Bananas Loose",                      "Fresh Produce",   "Sainsbury's",          0.99,  110, 80, 1),
    ("SAI-P002", "Sainsbury's Easy Peelers 600g",                  "Fresh Produce",   "Sainsbury's",          1.50,  60,  45, 1),
    ("SAI-P003", "Sainsbury's British Vine Tomatoes 500g",         "Fresh Produce",   "Sainsbury's",          1.20,  55,  40, 1),
    ("SAI-P004", "Sainsbury's Tenderstem Broccoli 200g",           "Fresh Produce",   "Sainsbury's",          1.50,  42,  30, 1),
    ("SAI-P005", "Sainsbury's Avocado",                            "Fresh Produce",   "Sainsbury's",          0.80,  48,  35, 2),
    ("SAI-P006", "Sainsbury's Baby Spinach 200g",                  "Fresh Produce",   "Sainsbury's",          1.60,  38,  28, 1),
    ("SAI-P007", "Sainsbury's Maris Piper Potatoes 2.5kg",         "Fresh Produce",   "Sainsbury's",          1.75,  65,  48, 2),
    ("SAI-P008", "Taste the Difference Chantenay Carrots 300g",    "Fresh Produce",   "Taste the Difference", 1.80,  30,  22, 2),
    ("SAI-P009", "Sainsbury's Iceberg Lettuce",                    "Fresh Produce",   "Sainsbury's",          0.59,  50,  36, 1),
    ("SAI-P010", "Sainsbury's British Strawberries 400g",          "Fresh Produce",   "Sainsbury's",          2.50,  35,  25, 1),

    # ── Drinks ────────────────────────────────────────────────────────────────
    ("SAI-DK01", "Sainsbury's Pure Orange Juice 1L",               "Drinks",          "Sainsbury's",          1.55,  52,  38, 2),
    ("SAI-DK02", "Innocent Orange Juice 900ml",                    "Drinks",          "Branded",              2.80,  30,  22, 3),
    ("SAI-DK03", "Sainsbury's Cola 2L",                            "Drinks",          "Sainsbury's",          1.15,  60,  45, 2),
    ("SAI-DK04", "Coca-Cola Original 6x330ml",                     "Drinks",          "Branded",              4.50,  42,  30, 3),
    ("SAI-DK05", "Sainsbury's Still Water 6x500ml",                "Drinks",          "Sainsbury's",          1.60,  48,  35, 2),
    ("SAI-DK06", "Highland Spring Sparkling Water 6x500ml",        "Drinks",          "Branded",              3.00,  28,  20, 3),
    ("SAI-DK07", "Taste the Difference Fairtrade Orange Juice 1L", "Drinks",          "Taste the Difference", 2.20,  20,  14, 2),
    ("SAI-DK08", "Sainsbury's Apple Juice 1L",                     "Drinks",          "Sainsbury's",          1.30,  38,  28, 2),

    # ── Ambient Grocery ───────────────────────────────────────────────────────
    ("SAI-G001", "Sainsbury's Long Grain White Rice 1kg",          "Ambient Grocery", "Sainsbury's",          1.25,  35,  25, 5),
    ("SAI-G002", "Sainsbury's Spaghetti 500g",                     "Ambient Grocery", "Sainsbury's",          0.90,  45,  32, 5),
    ("SAI-G003", "Heinz Baked Beans in Tomato Sauce 415g",         "Ambient Grocery", "Branded",              0.90,  70,  52, 5),
    ("SAI-G004", "Sainsbury's Chopped Tomatoes 400g",              "Ambient Grocery", "Sainsbury's",          0.65,  55,  40, 5),
    ("SAI-G005", "Taste the Difference Italian Passata 690g",      "Ambient Grocery", "Taste the Difference", 1.35,  25,  18, 5),
    ("SAI-G006", "Sainsbury's Plain Flour 1.5kg",                  "Ambient Grocery", "Sainsbury's",          1.10,  30,  22, 5),
    ("SAI-G007", "Kellogg's Cornflakes 500g",                      "Ambient Grocery", "Branded",              2.75,  35,  25, 5),
    ("SAI-G008", "Sainsbury's Porridge Oats 1kg",                  "Ambient Grocery", "Sainsbury's",          1.20,  42,  30, 5),
    ("SAI-G009", "Sainsbury's Baked Beans 4pk x 415g",             "Ambient Grocery", "Sainsbury's",          2.25,  40,  30, 5),
    ("SAI-G010", "Taste the Difference Crunchy Peanut Butter 280g","Ambient Grocery", "Taste the Difference", 2.50,  18,  12, 5),

    # ── Snacks & Confectionery ────────────────────────────────────────────────
    ("SAI-S001", "Walkers Ready Salted Crisps 6pk",                "Snacks",          "Branded",              2.00,  55,  40, 3),
    ("SAI-S002", "Doritos Chilli Heatwave 150g",                   "Snacks",          "Branded",              1.80,  45,  32, 3),
    ("SAI-S003", "Sainsbury's Salted Popcorn 100g",                "Snacks",          "Sainsbury's",          0.85,  38,  28, 4),
    ("SAI-S004", "Cadbury Dairy Milk 200g",                        "Snacks",          "Branded",              2.50,  60,  45, 3),
    ("SAI-S005", "Taste the Difference 70% Dark Chocolate 100g",   "Snacks",          "Taste the Difference", 1.75,  25,  18, 3),
    ("SAI-S006", "McVitie's Digestive Biscuits 400g",              "Snacks",          "Branded",              1.50,  48,  35, 4),
    ("SAI-S007", "Sainsbury's Houmous 200g",                       "Snacks",          "Sainsbury's",          1.00,  42,  30, 3),
    ("SAI-S008", "Pringles Original 200g",                         "Snacks",          "Branded",              2.00,  40,  28, 3),

    # ── Frozen ────────────────────────────────────────────────────────────────
    ("SAI-F001", "Sainsbury's Garden Peas 1kg",                    "Frozen",          "Sainsbury's",          1.25,  38,  28, 5),
    ("SAI-F002", "Sainsbury's Oven Chips 1.5kg",                   "Frozen",          "Sainsbury's",          1.45,  55,  40, 5),
    ("SAI-F003", "Taste the Difference Stone Baked Margherita 435g","Frozen",         "Taste the Difference", 4.50,  20,  14, 5),
    ("SAI-F004", "Sainsbury's Beef Lasagne 400g",                  "Frozen",          "Sainsbury's",          2.75,  30,  22, 5),
    ("SAI-F005", "Sainsbury's Vanilla Ice Cream 1L",               "Frozen",          "Sainsbury's",          2.00,  28,  20, 5),
    ("SAI-F006", "Taste the Difference Clotted Cream Ice Cream 1L","Frozen",          "Taste the Difference", 4.00,  15,  10, 5),
    ("SAI-F007", "Sainsbury's Fish Fingers 10pk",                  "Frozen",          "Sainsbury's",          2.50,  35,  25, 4),

    # ── Household & Health ────────────────────────────────────────────────────
    ("SAI-H001", "Sainsbury's Toilet Tissue 9 Rolls",              "Household",       "Sainsbury's",          3.50,  40,  30, 5),
    ("SAI-H002", "Andrex Gentle Clean Toilet Tissue 9 Rolls",      "Household",       "Branded",              5.00,  30,  22, 5),
    ("SAI-H003", "Sainsbury's Washing Up Liquid 500ml",            "Household",       "Sainsbury's",          0.89,  30,  22, 7),
    ("SAI-H004", "Fairy Platinum Plus Washing Up Liquid 650ml",    "Household",       "Branded",              2.75,  25,  18, 7),
    ("SAI-H005", "Sainsbury's Ibuprofen 200mg 16 Tablets",         "Health & Beauty", "Sainsbury's",          1.49,  22,  16, 7),
    ("SAI-H006", "Sainsbury's Vitamin D 1000IU 90 Tablets",        "Health & Beauty", "Sainsbury's",          3.50,  15,  10, 7),
]

# ─────────────────────────────────────────────────────────────────────────────
# Dynamically Generate 924 More Products to reach 1000 SKUs
# ─────────────────────────────────────────────────────────────────────────────
ADJECTIVES = ["Premium", "Organic", "British", "Scottish", "Fresh", "Finest", "Value", "Classic", "Spicy", "Sweet", "Smoked", "Roasted", "Free Range", "Corn Fed", "Stone Baked", "Salted", "Unsalted", "Mixed", "Chilled", "Handmade", "Whole", "Reduced Fat", "Extra Mature", "Slow Cooked", "Hand Stretched", "Gluten Free", "Vegan", "Plant Based", "Low Sugar", "High Protein", "Enriched", "Fortified", "Seasonal", "Limited Edition", "Locally Sourced", "Award Winning", "Sustainably Sourced", "Easy Cook"]
NOUNS = {
    "Dairy & Eggs": ["Milk 2 Pints", "Milk 4 Pints", "Milk 6 Pints", "Cheddar 400g", "Cheddar 750g", "Brie 200g", "Camembert 250g", "Stilton 200g", "Yogurt 500g", "Greek Yogurt 450g", "Butter 250g", "Clotted Cream 227g", "Eggs 6pk", "Eggs 12pk", "Cream 300ml", "Mozzarella 125g", "Feta 200g", "Cream Cheese 180g", "Custard 500g", "Soured Cream 300ml"],
    "Fresh Bakery": ["White Bread 800g", "Wholemeal Bread 800g", "Sourdough 400g", "Baguette", "Croissants 4pk", "Bagels 5pk", "Muffins 4pk", "Rolls 6pk", "Wraps 8pk", "Pitta Breads 6pk", "Crumpets 6pk", "English Muffins 4pk", "Pain au Chocolat 4pk", "Danish Pastries 4pk", "Focaccia 280g", "Naan Breads 2pk", "Tiger Bloomer 800g", "Seeded Batch 800g"],
    "Meat & Fish": ["Chicken Breast 600g", "Chicken Thighs 800g", "Whole Chicken 1.5kg", "Beef Mince 500g", "Beef Steak 300g", "Pork Sausages 8pk", "Pork Chops 500g", "Salmon Fillets 240g", "Salmon Side 400g", "Cod Fillets 400g", "Haddock Fillets 300g", "Bacon 300g", "Gammon Joint 750g", "Lamb Chops 400g", "Lamb Mince 500g", "Turkey Breast 500g", "Turkey Crown 2kg", "Prawns 200g", "Scallops 200g", "Tuna Steaks 240g"],
    "Fresh Produce": ["Bananas Loose", "Apples 6pk", "Oranges 5pk", "Potatoes 2.5kg", "Sweet Potatoes 1kg", "Carrots 1kg", "Broccoli 300g", "Cauliflower", "Spinach 200g", "Kale 200g", "Tomatoes 500g", "Cherry Tomatoes 250g", "Onions 1kg", "Red Onions 500g", "Grapes 500g", "Strawberries 400g", "Raspberries 150g", "Blueberries 150g", "Lemons 3pk", "Limes 5pk", "Avocado", "Cucumber", "Courgettes 3pk", "Peppers 3pk", "Mushrooms 400g", "Garlic Bulb", "Ginger Root 100g", "Asparagus 200g", "Tenderstem Broccoli 200g"],
    "Drinks": ["Orange Juice 1L", "Apple Juice 1L", "Cranberry Juice 1L", "Cola 2L", "Diet Cola 2L", "Sparkling Water 6x500ml", "Still Water 6x500ml", "Lemonade 2L", "Coffee Beans 200g", "Instant Coffee 200g", "Tea Bags 80pk", "Green Tea 40pk", "Squash 1L", "Energy Drink 500ml", "Iced Tea 500ml", "Kombucha 330ml", "Oat Milk 1L", "Almond Milk 1L", "Soy Milk 1L", "Prosecco 75cl"],
    "Ambient Grocery": ["Long Grain Rice 1kg", "Basmati Rice 1kg", "Arborio Rice 500g", "Spaghetti 500g", "Penne 500g", "Fusilli 500g", "Tagliatelle 375g", "Baked Beans 415g", "Chopped Tomatoes 400g", "Passata 500g", "Flour 1.5kg", "Self Raising Flour 1.5kg", "Sugar 1kg", "Caster Sugar 1kg", "Olive Oil 500ml", "Vegetable Oil 1L", "Cornflakes 500g", "Porridge Oats 1kg", "Peanut Butter 280g", "Almond Butter 170g", "Strawberry Jam 340g", "Marmalade 340g", "Tomato Ketchup 500g", "Mayonnaise 400g", "Soy Sauce 150ml", "Coconut Milk 400ml", "Chickpeas 400g", "Kidney Beans 400g", "Lentils 500g", "Vegetable Stock 500ml"],
    "Snacks": ["Ready Salted Crisps 6pk", "Cheese & Onion Crisps 6pk", "Tortilla Chips 200g", "Popcorn 100g", "Salted Popcorn 100g", "Milk Chocolate Bar 200g", "Dark Chocolate 100g", "White Chocolate 150g", "Digestive Biscuits 400g", "Shortbread Fingers 200g", "Hobnobs 300g", "Mixed Nuts 200g", "Cashew Nuts 200g", "Almonds 200g", "Crackers 200g", "Rice Cakes 130g", "Cereal Bars 6pk", "Flapjacks 5pk", "Jelly Babies 200g", "Wine Gums 190g"],
    "Frozen": ["Garden Peas 1kg", "Petits Pois 900g", "Oven Chips 1.5kg", "Sweet Potato Fries 750g", "Margherita Pizza 400g", "Pepperoni Pizza 400g", "Lasagne 400g", "Shepherd's Pie 400g", "Vanilla Ice Cream 1L", "Strawberry Ice Cream 1L", "Fish Fingers 10pk", "Battered Cod 2pk", "Chicken Nuggets 500g", "Chicken Kievs 2pk", "Mixed Vegetables 1kg", "Spinach 900g", "Hash Browns 700g", "Waffles 8pk", "Yorkshire Puddings 8pk", "Croissants 4pk"],
    "Household": ["Toilet Roll 9pk", "Toilet Roll 18pk", "Kitchen Roll 2pk", "Washing Up Liquid 500ml", "Laundry Pods 20pk", "Laundry Liquid 1.5L", "Fabric Conditioner 1L", "Bleach 750ml", "Antibacterial Spray 500ml", "Floor Cleaner 1L", "Bin Liners 20pk", "Freezer Bags 25pk", "Cling Film 30m", "Foil 10m", "Sponges 4pk", "Dishwasher Tablets 30pk", "Toilet Cleaner 500ml", "Air Freshener 300ml", "Washing Powder 1kg", "Dryer Sheets 40pk"],
    "Health & Beauty": ["Ibuprofen 200mg 16pk", "Paracetamol 500mg 16pk", "Vitamin C 1000mg 60pk", "Vitamin D 1000IU 90pk", "Omega 3 60 Capsules", "Toothpaste 75ml", "Electric Toothbrush Heads 3pk", "Shower Gel 250ml", "Bubble Bath 500ml", "Shampoo 250ml", "Conditioner 250ml", "Dry Shampoo 200ml", "Deodorant Spray 150ml", "Roll On Deodorant 50ml", "Hand Wash 250ml", "Hand Cream 75ml", "Cotton Pads 100pk", "Cotton Buds 200pk", "Plasters 40pk", "Razors 4pk"],
    "Wine & Spirits": ["Sauvignon Blanc 75cl", "Pinot Grigio 75cl", "Chardonnay 75cl", "Merlot 75cl", "Cabernet Sauvignon 75cl", "Shiraz 75cl", "Prosecco 75cl", "Champagne 75cl", "Rose Wine 75cl", "Gin 70cl", "Vodka 70cl", "Rum 70cl", "Whisky 70cl", "Baileys 70cl", "Tia Maria 70cl", "Pimms 70cl", "Sherry 75cl", "Port 75cl", "Beer 4x440ml", "Lager 12x330ml"],
    "Baby & Toddler": ["Infant Formula 800g", "Follow On Formula 800g", "Baby Wipes 72pk", "Nappies Size 3 40pk", "Nappies Size 4 34pk", "Nappies Size 5 28pk", "Baby Shampoo 250ml", "Baby Lotion 200ml", "Baby Food Pouch 100g", "Baby Porridge 125g", "Toddler Snacks 7pk", "Baby Rice 100g", "Sudocrem 125g", "Calpol 200ml", "Teething Gel 15g"],
    "Pet Food": ["Dog Food Pouches 12pk", "Cat Food Pouches 12pk", "Dry Dog Food 2kg", "Dry Cat Food 1.5kg", "Dog Treats 200g", "Cat Treats 60g", "Dog Biscuits 500g", "Cat Milk 200ml", "Hamster Food 1kg", "Bird Seed 1kg"],
    "Free From": ["Gluten Free Bread 400g", "Gluten Free Pasta 500g", "Dairy Free Milk 1L", "Dairy Free Butter 250g", "Vegan Cheese 200g", "Gluten Free Flour 1kg", "Oat Free Granola 400g", "Nut Free Cereal Bar 5pk", "Vegan Sausages 6pk", "Plant Based Mince 500g"],
    "World Foods": ["Basmati Rice 5kg", "Jasmine Rice 5kg", "Coconut Milk 400ml", "Soy Sauce 250ml", "Fish Sauce 200ml", "Oyster Sauce 255ml", "Miso Paste 200g", "Curry Paste 290g", "Naan Breads 4pk", "Chapattis 8pk", "Tortilla Chips 200g", "Tahini 300g", "Harissa Paste 120g", "Kimchi 300g"]
}
BRANDS = ["Heinz", "Kellogg's", "Walkers", "Cadbury", "Coca-Cola", "Fairy", "Andrex", "McVitie's", "Pringles", "Colgate", "Nivea", "Persil", "L'Oreal", "Nestle", "Innocent", "Alpro", "Oatly", "Warburtons", "Hovis", "Kenco", "PG Tips", "Yorkshire Tea", "Nescafe", "Muller", "Activia", "Lurpak", "Cathedral City", "Philadelphia", "Dairylea", "Cravendale", "Aptamil", "Pampers", "Huggies", "Sudocrem", "Pedigree", "Whiskas", "Felix", "Bakers", "PURINA", "Schweppes", "Robinsons", "Ribena", "Tropicana", "Copella"]

for i in range(len(SAINSBURYS_PRODUCTS) + 1, 5001):
    cat = random.choice(list(NOUNS.keys()))
    noun = random.choice(NOUNS[cat])
    adj = random.choice(ADJECTIVES)
    
    tier_rand = random.random()
    branded_cats = ["Household", "Health & Beauty", "Drinks", "Snacks", "Ambient Grocery", "Wine & Spirits", "Baby & Toddler", "Pet Food", "World Foods"]
    if cat in branded_cats:
        if tier_rand < 0.30:
            tier = "Branded"
            brand = random.choice(BRANDS)
            name = f"{brand} {noun}"
        elif tier_rand < 0.65:
            tier = "Sainsbury's"
            name = f"Sainsbury's {adj} {noun}"
        elif tier_rand < 0.85:
            tier = "Taste the Difference"
            name = f"Taste the Difference {adj} {noun}"
        else:
            tier = "So Good"
            name = f"Sainsbury's So Good {noun}"
    elif cat == "Free From":
        tier = random.choice(["Sainsbury's", "Taste the Difference", "Branded"])
        if tier == "Branded":
            name = f"{random.choice(BRANDS)} {noun}"
        elif tier == "Taste the Difference":
            name = f"Taste the Difference {noun}"
        else:
            name = f"Sainsbury's Free From {noun}"
    else:
        if tier_rand < 0.68:
             tier = "Sainsbury's"
             name = f"Sainsbury's {adj} {noun}"
        elif tier_rand < 0.88:
             tier = "Taste the Difference"
             name = f"Taste the Difference {adj} {noun}"
        else:
             tier = "So Good"
             name = f"Sainsbury's So Good {noun}"
             
    name = name.replace("  ", " ").strip()
    
    sku = f"SAI-GEN{str(i).zfill(4)}"
    unit_price = round(random.uniform(0.5, 8.0), 2)
    
    if tier == "Taste the Difference":
        unit_price = round(unit_price * 1.5, 2)
        base_demand = random.randint(15, 35)
    elif tier == "So Good":
        unit_price = round(unit_price * 0.7, 2)
        base_demand = random.randint(50, 90)
    elif tier == "Branded":
        unit_price = round(unit_price * 1.3, 2)
        base_demand = random.randint(30, 60)
    else:
        base_demand = random.randint(35, 75)
        
    reorder_point = int(base_demand * random.uniform(0.5, 0.8))
    lead_time = random.randint(1, 4) if cat not in ["Fresh Bakery", "Dairy & Eggs", "Fresh Produce"] else random.randint(1, 2)
    
    SAINSBURYS_PRODUCTS.append((sku, name, cat, tier, unit_price, base_demand, reorder_point, lead_time))

# ─────────────────────────────────────────────────────────────────────────────
# Build DataFrame
# ─────────────────────────────────────────────────────────────────────────────
products_df = pd.DataFrame(SAINSBURYS_PRODUCTS, columns=[
    "product_id", "product_name", "category", "tier",
    "unit_price", "base_demand", "reorder_point", "lead_time_days"
])
products_df.to_csv(os.path.join(OUTPUT_DIR, "products.csv"), index=False)
print(f"✅ products.csv — {len(products_df)} Sainsbury's products across {products_df['category'].nunique()} categories")

# ─────────────────────────────────────────────────────────────────────────────
# Calendar — Q4 2024 (Oct 1 – Dec 31) — UK Key Events
# ─────────────────────────────────────────────────────────────────────────────
start_date = date(2024, 10, 1)
end_date   = date(2024, 12, 31)
dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]

# UK Bank Holidays Q4 2024
uk_bank_holidays = {
    date(2024, 12, 25),  # Christmas Day
    date(2024, 12, 26),  # Boxing Day
}

# Key UK shopping events for demand multipliers
uk_events = {
    date(2024, 10, 31): ("Halloween",    1.5),
    date(2024, 11, 1):  ("Halloween",    1.2),
    date(2024, 11, 5):  ("Bonfire Night",1.3),
    date(2024, 11, 29): ("Black Friday", 1.6),
    date(2024, 11, 30): ("Black Friday", 1.4),
    date(2024, 12, 1):  ("Cyber Monday", 1.2),
    date(2024, 12, 20): ("Christmas Rush", 1.8),
    date(2024, 12, 21): ("Christmas Rush", 2.0),
    date(2024, 12, 22): ("Christmas Rush", 2.2),
    date(2024, 12, 23): ("Christmas Rush", 2.5),
    date(2024, 12, 24): ("Christmas Eve",  3.0),
    date(2024, 12, 26): ("Boxing Day",     1.8),
    date(2024, 12, 27): ("Post Christmas", 1.4),
    date(2024, 12, 28): ("Post Christmas", 1.3),
    date(2024, 12, 29): ("Post Christmas", 1.2),
}

# Nectar promotion weeks — realistic Sainsbury's promotional calendar
nectar_promo_weeks = {
    (2024, 40), (2024, 43), (2024, 45), (2024, 47),  # October-November promos
    (2024, 48), (2024, 49), (2024, 50),               # Pre-Christmas heavy promo
}

calendar_records = []
for d in dates:
    iso    = d.isocalendar()
    event, event_mult = uk_events.get(d, ("Normal", 1.0))
    is_nectar_week = (iso.year, iso.week) in nectar_promo_weeks
    calendar_records.append({
        "date":            d.isoformat(),
        "day_of_week":     d.weekday(),
        "day_name":        d.strftime("%A"),
        "week_of_year":    iso.week,
        "month":           d.month,
        "is_weekend":      int(d.weekday() >= 5),
        "is_bank_holiday": int(d in uk_bank_holidays),
        "is_month_end":    int(d.day >= 28),
        "uk_event":        event,
        "event_multiplier":event_mult,
        "is_nectar_week":  int(is_nectar_week),
        "is_christmas_period": int(d >= date(2024, 12, 15)),
    })

calendar_df = pd.DataFrame(calendar_records)
calendar_df.to_csv(os.path.join(OUTPUT_DIR, "calendar.csv"), index=False)
print(f"✅ calendar.csv — {len(calendar_df)} days (Q4 2024, UK events flagged)")

# ─────────────────────────────────────────────────────────────────────────────
# Daily Sales — Category-aware seasonality + UK events
# ─────────────────────────────────────────────────────────────────────────────
# Category seasonality adjustments for Q4 (Oct–Dec)
CATEGORY_SEASONALITY = {
    "Fresh Bakery":     {"Oct": 1.0,  "Nov": 1.1,  "Dec": 1.5},  # Christmas baking
    "Dairy & Eggs":     {"Oct": 1.0,  "Nov": 1.1,  "Dec": 1.4},  # Baking/cooking surge
    "Meat & Fish":      {"Oct": 1.0,  "Nov": 1.1,  "Dec": 1.8},  # Christmas feasting
    "Fresh Produce":    {"Oct": 1.0,  "Nov": 0.95, "Dec": 1.2},  # Slight dip in fresh veg
    "Drinks":           {"Oct": 1.0,  "Nov": 1.1,  "Dec": 1.6},  # Festive drinks
    "Ambient Grocery":  {"Oct": 1.0,  "Nov": 1.0,  "Dec": 1.3},  # Pantry stocking
    "Snacks":           {"Oct": 1.2,  "Nov": 1.1,  "Dec": 1.5},  # Halloween + Christmas
    "Frozen":           {"Oct": 1.0,  "Nov": 1.0,  "Dec": 1.4},  # Christmas ready meals
    "Household":        {"Oct": 1.0,  "Nov": 0.95, "Dec": 1.2},
    "Health & Beauty":  {"Oct": 1.0,  "Nov": 1.0,  "Dec": 1.1},
    "Wine & Spirits":   {"Oct": 1.1,  "Nov": 1.4,  "Dec": 2.2},  # Huge festive spike
    "Baby & Toddler":   {"Oct": 1.0,  "Nov": 1.0,  "Dec": 1.1},
    "Pet Food":         {"Oct": 1.0,  "Nov": 1.0,  "Dec": 1.2},
    "Free From":        {"Oct": 1.0,  "Nov": 1.0,  "Dec": 1.1},
    "World Foods":      {"Oct": 1.1,  "Nov": 1.0,  "Dec": 1.2},
}

MONTH_NAMES = {10: "Oct", 11: "Nov", 12: "Dec"}

# Nectar promotion products — % of products in each category on Nectar Price on a given promo week
NECTAR_ELIGIBLE_PIDS = set(
    products_df[products_df["tier"].isin(["Sainsbury's", "So Good"])]["product_id"]
)

cal_index = calendar_df.set_index("date")

sales_records = []
for _, prod in products_df.iterrows():
    base = prod["base_demand"]
    cat  = prod["category"]
    tier = prod["tier"]
    pid  = prod["product_id"]

    # Taste the Difference products sell less frequently but at higher margin
    if tier == "Taste the Difference":
        base *= 0.40
    elif tier == "Branded":
        base *= 0.70

    # Day-of-week effect (Sainsbury's real pattern: Fri/Sat peak)
    dow_mult = {0: 0.85, 1: 0.88, 2: 0.90, 3: 0.92, 4: 1.10, 5: 1.25, 6: 1.20}

    for d in dates:
        d_str  = d.isoformat()
        row    = cal_index.loc[d_str]
        month  = MONTH_NAMES[d.month]

        # Seasonal adjustment
        seasonal = CATEGORY_SEASONALITY.get(cat, {}).get(month, 1.0)

        # Day-of-week multiplier
        dow = dow_mult.get(d.weekday(), 1.0)

        # Event multiplier (Christmas, Halloween, Black Friday etc.)
        event_m = float(row["event_multiplier"])

        # Nectar price promotion boost (random 15% of eligible products in promo week)
        is_promo = 0
        promo_boost = 1.0
        if row["is_nectar_week"] and pid in NECTAR_ELIGIBLE_PIDS and random.random() < 0.15:
            is_promo   = 1
            promo_boost = random.uniform(1.35, 1.85)  # Sainsbury's avg promo uplift

        # Bank holiday near-closure effect (massive surge day before)
        bh_pre_mult = 1.0
        tomorrow = (d + timedelta(days=1))
        if tomorrow in uk_bank_holidays:
            bh_pre_mult = 2.2  # panic buying before bank holiday

        demand = base * seasonal * dow * event_m * promo_boost * bh_pre_mult
        noise  = np.random.normal(0, max(demand * 0.18, 0.01))
        demand = max(0.0, demand + noise)

        sales_records.append({
            "sale_id":       len(sales_records) + 1,
            "product_id":    pid,
            "store_id":      "SBY-LON-001",  # Sainsbury's London flagship store
            "date":          d_str,
            "units_sold":    round(demand, 2),
            "is_promotion":  is_promo,
            "promo_type":    "Nectar Price" if is_promo else "None",
            "uk_event":      row["uk_event"],
        })

sales_df = pd.DataFrame(sales_records)
sales_df.to_csv(os.path.join(OUTPUT_DIR, "daily_sales.csv"), index=False)
n_promo = sales_df["is_promotion"].sum()
print(f"✅ daily_sales.csv — {len(sales_df):,} records | {n_promo:,} Nectar promo events")

# ─────────────────────────────────────────────────────────────────────────────
# Inventory — Simulate realistic stock levels with Sainsbury's supply chain
# ─────────────────────────────────────────────────────────────────────────────
inventory_records = []
sales_pivot = sales_df.set_index(["product_id", "date"])["units_sold"]

for _, prod in products_df.iterrows():
    pid     = prod["product_id"]
    reorder = prod["reorder_point"]
    lead    = prod["lead_time_days"]

    # Start at generous stock level for a major supermarket
    stock = random.uniform(reorder * 3, reorder * 6)

    for d in dates:
        d_str = d.isoformat()
        sold  = sales_pivot.get((pid, d_str), 0)
        stock = max(0.0, stock - sold)

        # Sainsbury's automated replenishment trigger
        if stock < reorder:
            restock_chance = 0.85 if lead <= 2 else 0.70  # faster for perishables
            if random.random() < restock_chance:
                qty   = random.uniform(reorder * 2.5, reorder * 5)
                stock += qty

        inventory_records.append({
            "product_id":    pid,
            "store_id":      "SBY-LON-001",
            "date":          d_str,
            "stock_on_hand": round(stock, 2),
            "reorder_point": reorder,
        })

inventory_df = pd.DataFrame(inventory_records)
inventory_df.to_csv(os.path.join(OUTPUT_DIR, "inventory.csv"), index=False)
print(f"✅ inventory.csv — {len(inventory_df):,} records")

# ─────────────────────────────────────────────────────────────────────────────
# Replenishment — Sainsbury's rapid supply chain (dairy: 1-day, ambient: 3-5 days)
# ─────────────────────────────────────────────────────────────────────────────

def _get_supplier(tier: str, category: str) -> str:
    if tier == "Taste the Difference":
        return random.choice(["Sainsbury's Premium Suppliers Ltd", "UK Farmhouse Co-op"])
    elif category == "Meat & Fish":
        return random.choice(["ABP Food Group", "Morrisons Wholesale", "Cranswick PLC"])
    elif category == "Fresh Produce":
        return random.choice(["G's Fresh", "Berry World", "Fresca Group"])
    else:
        return random.choice(["Sainsbury's DC Weybridge", "Sainsbury's DC Emerald Park",
                               "Sainsbury's DC Hams Hall"])


replenishment_records = []
inv_pivot = inventory_df.set_index(["product_id", "date"])["stock_on_hand"]

for _, prod in products_df.iterrows():
    pid    = prod["product_id"]
    reorder = prod["reorder_point"]
    lead   = prod["lead_time_days"]
    last_order_date = None

    for d in dates:
        d_str = d.isoformat()
        stock = inv_pivot.get((pid, d_str), 999)

        # Don't double-order within lead time
        if last_order_date and (d - last_order_date).days < lead:
            continue

        if stock < reorder * 1.5 and random.random() < 0.55:
            ordered  = round(random.uniform(reorder * 2, reorder * 4))
            # Sainsbury's supply chain reliability: 96% on-time delivery
            received = ordered if random.random() < 0.96 else round(ordered * random.uniform(0.85, 0.98))
            replenishment_records.append({
                "product_id":     pid,
                "store_id":       "SBY-LON-001",
                "order_date":     d_str,
                "expected_date":  (d + timedelta(days=lead)).isoformat(),
                "units_ordered":  ordered,
                "units_received": received,
                "supplier":       _get_supplier(prod["tier"], prod["category"]),
                "status":         "received" if received == ordered else "partial",
            })
            last_order_date = d

replenishment_df = pd.DataFrame(replenishment_records)
replenishment_df.to_csv(os.path.join(OUTPUT_DIR, "replenishment.csv"), index=False)
print(f"✅ replenishment.csv — {len(replenishment_df):,} replenishment orders")
print("\n🎉 Sainsbury's Q4 2024 dataset ready in data/raw/")
print(f"   Store: SBY-LON-001 | Period: Oct–Dec 2024 | Products: {len(products_df):,}")
print(f"   Peak events: Christmas Eve (+200%), Black Friday (+60%), Halloween (+50%)")
