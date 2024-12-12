import _thread
import json
import requests
import os

target_dir = "H:/MGSTR/source_data"

f2 = open('data_scrapping/stage_to_EFR_products.json',)
stages = json.load(f2)

os.chdir(target_dir)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def retrieve_api_key():
    data = {
        "client_id": "CLOUDFERRO_PUBLIC",
        "username": "228097@student.pwr.edu.pl",
        "password": "Kurw@Mac#",
        "grant_type": "password"
    }
    auth_response = requests.post('https://auth.creodias.eu/auth/realms/DIAS/protocol/openid-connect/token', data)
    auth_response = auth_response.json()
    return auth_response["access_token"]


def get_storage_path(stage_name, product_name):
    return f"{target_dir}/{stage_name}/{product_name}.zip"
    # return f"{stage_name}/{product_name}.ico"


def get_url(download_url):
    return f"{download_url}?token={retrieve_api_key()}"


def download_product(stage_name, satellite_prod):
    path = get_storage_path(stage_name, satellite_prod["name"])
    print(f"Processing download request for {satellite_prod['name']}...")
    resp = requests.get(get_url(satellite_prod["download"]), allow_redirects=True)
    # resp = requests.get('https://www.facebook.com/favicon.ico', allow_redirects=True)
    print(f"Initiated download of {satellite_prod['name']} under {path}...")
    with open(path, 'wb') as file:
        file.write(resp.content)
    print(f"Download completed for {satellite_prod}!")


def filter_downloads(products):
    existing_products = dict()
    missing_products = dict()
    for stage in products.keys():
        existing_products[stage] = set([prod[:-4] if ".zip" in prod else prod for prod in os.listdir(stage)])
        missing_products[stage] = list(
            filter(lambda product: is_download_incomplete(product, stage, existing_products), products[stage])
        )
    return missing_products


def is_download_incomplete(product, stage, existing_products):
    if product["name"] not in existing_products[stage]:
        return True
    files = os.listdir(f"{stage}/{product['name']}")
    return "tie_meteo.nc" not in files or \
           "xfdumanifest.xml" not in files or \
           "geo_coordinates.nc" not in files


def run_download():
    products_to_download = filter_downloads(stages)
    print(products_to_download.values())
    for (stage, products) in products_to_download.items():
        products_batches = list(chunks(products, 5))
        for batch in products_batches:
            head_product = batch.pop(0)
            for prod in batch:
                _thread.start_new_thread(download_product, (stage, prod))
            download_product(stage, head_product)


run_download()
