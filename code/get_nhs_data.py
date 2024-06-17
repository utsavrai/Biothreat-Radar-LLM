# %%
import pandas as pd

# %%

df = pd.read_csv(r"NHS Data/births_by_region.csv")

conditions = (df["Dimension"] == "TotalDeliveries") & (
    df["Org_Level"] == "NHS England (Region)"
)
df = df.loc[conditions].reset_index(drop=True)

rnm = {"Org_Name": "Region", "Value": "Number of deliveries"}

df = df.rename(columns=rnm)
df["Region"] = (
    df["Region"]
    .str.title()
    .str.replace("Of", "of")
    .str.replace("And", "and")
    .str.replace("Commissioning Region", "")
    .str.strip()
)

cols = ["Region", "Number of deliveries"]
df = df[cols]
df.to_csv(r"NHS Data/births_by_region.csv", index=False)
