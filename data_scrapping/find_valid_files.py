import glob
import json
import os


target_dir = "H:\\MGSTR\\source_data"

f2 = open('stage_to_EFR_products.json',)
stages = json.load(f2)


def is_download_incomplete(product, stage, existing_products):
    if product["name"] not in existing_products[stage]:
        return True
    print(product)
    print(existing_products[stage])
    files = os.listdir(os.path.join(target_dir, stage, product['name']))
    return "tie_meteo.nc" not in files or \
           "xfdumanifest.xml" not in files or \
           "geo_coordinates.nc" not in files


def export_valid_files():
    existing_products = dict()
    for (stage, products) in stages.items():
        existing_products[stage] = list(
            filter(lambda filename: ".zip" not in filename, os.listdir(os.path.join(target_dir, stage)))
        )
        for product in products:
            if not is_download_incomplete(product, stage, existing_products):
                file = glob.glob(os.path.join(target_dir, stage, product["name"]))
                file_destination = os.path.join(target_dir, product["name"])
                os.rename(file[0], file_destination)


export_valid_files()
