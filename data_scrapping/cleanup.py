import json
import os

F1 = open('stage_to_EFR_products.json',)
F2 = open('stage_to_WFR_products.json',)
F3 = open('non_hurricane_EFR_products.json',)
F4 = open('non_hurricane_WFR_products.json',)
F5 = open('hurricane_to_EFR_products.json',)
F6 = open('hurricane_to_WFR_products.json',)

source_dir = "H:\\MGSTR\\source_data"

stages_efr = json.load(F1)
stages_wfr = json.load(F2)
other_efr = json.load(F3)
other_wfr = json.load(F4)
hurricanes_efr = json.load(F5)
hurricanes_wfr = json.load(F6)

stages_efr["other"] = other_efr["other"]
stages_wfr["other"] = other_wfr["other"]

unique_efr_products = {}
unique_wfr_products = {}

stages = ["stage_1", "stage_2", "stage_3", "stage_4", "other"]

for stage in stages:
    stage_dir = os.path.join(source_dir, stage, "EFR")
    existing_products = [file for file in os.listdir(stage_dir) if file.endswith(".SEN3")]
    unique_efr_products[stage] = [product for product in stages_efr[stage] if product["name"] in existing_products]

    stage_dir = os.path.join(source_dir, stage, "WFR")
    existing_products = [file for file in os.listdir(stage_dir) if file.endswith(".SEN3")]
    unique_wfr_products[stage] = [product for product in stages_wfr[stage] if product["name"] in existing_products]

all_unique_efr_products = [item for sublist in list(unique_efr_products.values()) for item in sublist]
for hurricane in hurricanes_efr.keys():
    hurricanes_efr[hurricane] = [product for product in hurricanes_efr[hurricane] if product in all_unique_efr_products]

all_unique_wfr_products = [item for sublist in list(unique_efr_products.values()) for item in sublist]
for hurricane in hurricanes_wfr.keys():
    hurricanes_wfr[hurricane] = [product for product in hurricanes_wfr[hurricane] if product in all_unique_wfr_products]

unique_other_efr_products = {"other": unique_efr_products["other"]}
unique_other_wfr_products = {"other": unique_wfr_products["other"]}
del unique_efr_products["other"]
del unique_wfr_products["other"]

F1 = open('stage_to_EFR_products.json', 'w')
F2 = open('stage_to_WFR_products.json', 'w')
F3 = open('non_hurricane_EFR_products.json', 'w')
F4 = open('non_hurricane_WFR_products.json', 'w')
F5 = open('hurricane_to_EFR_products.json', 'w')
F6 = open('hurricane_to_WFR_products.json', 'w')

json.dump(unique_efr_products, F1)
json.dump(unique_wfr_products, F2)
json.dump(unique_other_efr_products, F3)
json.dump(unique_other_wfr_products, F4)
json.dump(hurricanes_efr, F5)
json.dump(hurricanes_wfr, F6)