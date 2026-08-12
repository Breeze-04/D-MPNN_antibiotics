import requests
import pandas as pd

url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"

params = {
    "standard_type": "MIC",
    "target_organism": "Escherichia coli",
    "limit": 1000
}

all_records = []

while url:
    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()

    all_records.extend(data["activities"])

    print("Downloaded:", len(all_records))

    next_url = data["page_meta"]["next"]

    if next_url:
        url = "https://www.ebi.ac.uk" + next_url
        params = None
    else:
        url = None

df = pd.DataFrame(all_records)

print("\nShape:")
print(df.shape)

# 保存原始数据
df.to_csv(
    "dataprocess/chembl_ecoli_MIC_raw.csv",
    index=False
)

print("\nSaved to:")
print("dataprocess/chembl_ecoli_MIC_raw.csv")