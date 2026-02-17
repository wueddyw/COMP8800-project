import pandas as pd
import json

COLS = [f"f{i}" for i in range(41)] + ["label", "difficulty"]
FEATURES = [f"f{i}" for i in range(41)]

df = pd.read_csv("data/KDDTest+.TXT", names=COLS)

# Pick one normal and one attack from the test set
normal_row = df[df["label"] == "normal"].iloc[0]
attack_row = df[df["label"] != "normal"].iloc[0]  # first attack (real labeled attack type)

with open("demo_normal.json", "w") as f:
    json.dump({"data": normal_row[FEATURES].to_dict()}, f, indent=2)

with open("demo_attack.json", "w") as f:
    json.dump({"data": attack_row[FEATURES].to_dict()}, f, indent=2)

print("Saved demo_normal.json and demo_attack.json")
print("Normal label was:", normal_row["label"])
print("Attack label was:", attack_row["label"])
