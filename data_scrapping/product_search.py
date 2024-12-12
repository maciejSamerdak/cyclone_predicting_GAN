import datetime
import requests
import json

year_threshold = 2016  # that's when sentinel 3 starts...

product_type = "EFR"
instrument = "OL"

stage_to_storm_type = {
    "stage_1": ["DB", "WV"],
    "stage_2": ["SD", "TD"],
    "stage_3": ["SS", "TS"],
    "stage_4": ["EX", "HU"],
}


def simplify_long_lat(input):
    if input[-1] in ["S", "W"]:
        return "-" + input[:-1]
    return input[:-1]


def get_api_request_url(date, lon, lat):
    completion_date = date + datetime.timedelta(hours=5, minutes=59)
    return "http://finder.creodias.eu/resto/api/collections/Sentinel3/search.json?startDate={}&completionDate={}&lon={}&lat={}&instrument={}&productType={}".format(
        date.isoformat(),
        completion_date.isoformat(),
        lon,
        lat,
        instrument,
        product_type
    )


with open("hurdat2-1851-2020-052921.txt", "r") as a_file:
    index = 0
    year = 1851
    hurricane_name = ""
    hurricane_to_data_packs = {}
    stage_to_data_packs = {}
    for line in a_file:
        stripped_line = line.strip()
        if index == 0:
            # Get hurricane metadata from header
            year = int(stripped_line[4:8])
            index = int(stripped_line[33:36]) + 1
            hurricane_name = stripped_line.split(",")[1].replace(" ", "")
            if hurricane_name == "UNNAMED":
                hurricane_name = stripped_line[:8]
            print(f"{hurricane_name} {year}")
        else:
            print(index)
            if year >= year_threshold:
                # retrieve data for query
                month = int(stripped_line[4:6])
                day = int(stripped_line[6:8])
                hour = int(stripped_line[10:12])
                minute = int(stripped_line[12:14])
                line_chunks = stripped_line[16:44].split(", ")
                storm_type = line_chunks[1]
                lat = simplify_long_lat(line_chunks[2].replace(" ", ""))
                lon = simplify_long_lat(line_chunks[3][1:].replace(" ", ""))
                date = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute)

                # find which stage
                stages = list(stage_to_storm_type.keys())
                target_stage = stage_to_storm_type[stages[0]]
                stages_index = 0
                while stages_index < len(stages) and storm_type not in stage_to_storm_type[stages[stages_index]]:
                    stages_index += 1

                if stages_index < len(stages):
                    target_stage = stages[stages_index]

                    # perform search request
                    response = requests.get(get_api_request_url(date, lon, lat))
                    response = response.json()
                    if response["properties"]["totalResults"] > 0:
                        products = []
                        if stage_to_data_packs.get(stages[stages_index], None) is None:
                            stage_to_data_packs[stages[stages_index]] = []
                        if hurricane_to_data_packs.get(hurricane_name, None) is None:
                            hurricane_to_data_packs[hurricane_name] = []
                        for feature in response["features"]:
                            product = {
                                "name": feature["properties"]["title"],
                                "download": feature["properties"]["services"]["download"]["url"]
                            }
                            if product not in stage_to_data_packs[stages[stages_index]]:
                                products.append(product)

                        stage_to_data_packs[stages[stages_index]] += products

                        hurricane_to_data_packs[hurricane_name] += products
                        print(products)
        index -= 1
    print(hurricane_to_data_packs)
    print(stage_to_data_packs)


with open(f'hurricane_to_{product_type}_products.json', 'w') as f:
    json.dump(hurricane_to_data_packs, f)


with open(f'stage_to_{product_type}_products.json', 'w') as f:
    json.dump(stage_to_data_packs, f)
# 20100803, 1800,  , DB, 15.3N,  52.2W,  35, 1009,   60,    0,    0,   45,    0,    0,    0,    0,    0,    0,    0,    0,
