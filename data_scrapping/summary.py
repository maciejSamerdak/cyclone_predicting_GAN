import json
import pandas as pd

f1 = open('hurricane_to_EFR_products.json',)
hurricanes = json.load(f1)

f2 = open('stage_to_EFR_products.json',)
stages = json.load(f2)

for key in hurricanes.keys():
    print(f"{key}: {len(hurricanes[key])}")
print(f"{len(hurricanes.keys())} hurricanes, {sum(len(lst) for lst in hurricanes.values())} pictures")

for key in stages.keys():
    print(f"{key}: {len(stages[key])}")

mapset = {}
for hurricane, hurr_product_list in hurricanes.items():
    hurr_product_names = [other_product["name"] for other_product in hurr_product_list]
    mapset[hurricane] = {}
    for stage, stage_product_list in stages.items():
        stage_products = [product for product in stage_product_list if product["name"] in hurr_product_names]
        mapset[hurricane][stage] = len(stage_products)

df = pd.DataFrame(mapset).transpose().sort_index()

# df = df.append(df.sum().rename('Total')).append(df.sum(axis=1).rename('Total'))

df.loc['Total'] = df.sum(numeric_only=True, axis=0)
df.loc[:, 'Total'] = df.sum(numeric_only=True, axis=1)

df = df.rename(columns={'stage_1': '1 - Fala', 'stage_2':  '2 - Depresja', 'stage_3':  '3 - Burza', 'stage_4':  '4- Cyklon'})

df.to_csv("summary.csv")
