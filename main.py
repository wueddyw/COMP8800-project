import pandas as pd

cols = [f"f{i}" for i in range(41)] + ["label", "difficulty"]

df = pd.read_csv("data/KDDTrain+.TXT", names=cols)

print(df.head())
print(df["label"].value_counts())
print(df.describe())