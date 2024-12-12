import datetime
import requests
import json
import random


product_type = "WFR"
instrument = "OL"


url = "https://finder.creodias.eu/resto/api/collections/Sentinel3/search.json?maxRecords=10&startDate=2016-01-01T00%3A00%3A00Z&completionDate=2020-12-31T23%3A59%3A59Z&cloudCover=%5B0%2C10%5D&instrument=OL&geometry=POLYGON((-2.8298017771702013+44.78693312075771%2C-100.02734107997269+44.26062532356406%2C-98.30485304169517+15.316635023001155%2C-76.40464798359537+1.2302540539015325%2C-34.57279562542721+-6.629043159845139%2C-1.107313738892689+-6.13995747281416%2C-2.8298017771702013+44.78693312075771))&sortParam=startDate&sortOrder=descending&status=all&dataset=ESA-DATASET"


def simplify_long_lat(input):
    if input[-1] in ["S", "W"]:
        return "-" + input[:-1]
    return input[:-1]


# lat_range = (simplify_long_lat("6.8S"), simplify_long_lat("47.5N"))
# lon_range = (simplify_long_lat("95.6W"), simplify_long_lat("5.6W"))
# polygon = f"POLYGON (({lat_range[0]} {lon_range[0]}, {lat_range[1]} {lon_range[0]}, {lat_range[1]} {lon_range[1]}, {lat_range[0]} {lon_range[1]}, {lat_range[0]} {lon_range[0]}))"
date_range = (datetime.datetime(year=2016, month=1, day=1), datetime.datetime(year=2020, month=12, day=31))

random.seed(123)

products_amount = 400


def random_date():
    """Generate a random datetime between `start` and `end`"""
    return date_range[0] + datetime.timedelta(
        # Get a random amount of seconds between `start` and `end`
        seconds=random.randint(0, int((date_range[1] - date_range[0]).total_seconds())),
    )


def get_api_request_url(date):
    completion_date = date + datetime.timedelta(days=1)
    return "https://finder.creodias.eu/resto/api/collections/Sentinel3/search.json?maxRecords={}&startDate={}&completionDate={}&instrument=OL&productType={}&geometry=POLYGON((-66.68489405331512+48.33805220987668%2C-3.1989063568011087+51.50507801545518%2C-1.7224880382775458+43.01409203968416%2C-7.628161312371865+32.95787974921409%2C-12.057416267942621+19.541391224070736%2C-6.889952153110083+-8.093252063858003%2C-40.60150375939852+-7.849560208081385%2C-53.39712918660288+0%2C-60.7792207792208+6.139957472814146%2C-73.57484620642516+8.580189571047185%2C-82.43335611756666+8.336796621055228%2C-97.4436090225564+20.4662755944873%2C-98.18181818181819+30.65788790292109%2C-85.38619275461383+32.12815782413472%2C-79.23444976076557+38.73716865320793%2C-69.39166097060837+45.82528281946168%2C-66.68489405331512+48.33805220987668))&sortParam=startDate&sortOrder=descending".format(
        1,
        date.isoformat(),
        completion_date.isoformat(),
        product_type
    )


def get_all_hurricane_products():
    file = open(f'stage_to_{product_type}_products.json', )
    data = json.load(file)
    existing = []
    for prods in data.values():
        product_names = [prod["name"] for prod in prods]
        existing += product_names
    return existing


existing_products = get_all_hurricane_products()

print(f"{len(existing_products)} existing hurricane products found")

final_products = {"other": []}


counter = 0

while counter < products_amount:

    already_saved = [product["name"] for product in final_products["other"]]

    r_date = random_date()

    print(f"querying for day {r_date}")

    response = requests.get(get_api_request_url(r_date))
    response = response.json()
    print(f"{response['properties']['totalResults']} products returned")
    if response["properties"]["totalResults"] > 0:
        products = []
        index = 0
        print(f"Features length: {len(response['features'])}")
        while counter < products_amount and index < len(response["features"]):
            feature = response["features"][index]
            product = {
                "name": feature["properties"]["title"],
                "download": feature["properties"]["services"]["download"]["url"]
            }
            print(product)
            if product["name"] not in existing_products + already_saved:
                products.append(product)
                counter += 1
            index += 1

        final_products["other"] += products
        print(f"{len(products)} product(s) added")


print(f"{len(final_products)} pictures collected")

with open(f'non_hurricane_{product_type}_products.json', 'w') as f:
    json.dump(final_products, f)
