import json
import os
import shutil

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


def compare_coords(a, other):
    other_coords = other["name"].split("_")[10:14]
    for i in range(len(a)):
        if abs(int(a[i]) - int(other_coords[i])) > 1:
            return False
    return True


def is_duplicate(item, elements, index):
    coords = item["name"].split("_")[10:14]
    prod_id = item["name"][:47]
    return any(prod_id in other_item["name"] and compare_coords(coords, other_item) for other_item in elements[:index])


for stage in stages:
    efr_elements = stages_efr[stage]
    wfr_elements = stages_wfr[stage]

    print(f"EFR duplicates for {stage}")
    efr_dupes = [x["name"] for n, x in enumerate(efr_elements) if is_duplicate(x, efr_elements, n)]
    print(len(efr_dupes))

    unique_efr_products[stage] = [x for n, x in enumerate(efr_elements) if not is_duplicate(x, efr_elements, n)]

    print(f"WFR duplicates for {stage}")
    wfr_dupes = [x["name"] for n, x in enumerate(wfr_elements) if is_duplicate(x, wfr_elements, n)]
    print(len(wfr_dupes))

    unique_wfr_products[stage] = [x for n, x in enumerate(wfr_elements) if not is_duplicate(x, wfr_elements, n)]

    for file_name in efr_dupes:
        try:
            shutil.rmtree(os.path.join(source_dir, stage, "EFR", file_name))
            print(f"removed {file_name}")
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    for file_name in wfr_dupes:
        try:
            shutil.rmtree(os.path.join(source_dir, stage, "WFR", file_name))
            print(f"removed {file_name}")
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

    for file_name in efr_dupes:
        file_name += ".png"
        if stage == "stage_4":
            file_dir = os.path.join(source_dir, "exported_images\\source\\train\\hurricane", file_name)
        elif stage == "other":
            file_dir = os.path.join(source_dir, "exported_images\\source\\val\\other", file_name)
        else:
            file_dir = os.path.join(source_dir, "exported_images\\source\\val\\hurricane", file_name)
        try:
            os.remove(file_dir)
            print(f"removed {file_name}")
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))

