import os
import pandas as pd
from lulc_validation.lulc_val import StratVal

csvs = os.listdir(os.getcwd())

csvs = [f for f in csvs if f.endswith(".csv")]

dfs = []

for f_df in csvs:
    df = pd.read_csv(f_df)
    dfs.append(df)

dfs = pd.concat(dfs, ignore_index=True)

dfs = dfs.loc[:, ["class", "strata"]]

dfs["map_class"] = dfs["strata"]

dfs.columns = ["ref_class", "map_class", "strata"]

n_strata = [
    1779768,
    3549325,
    541204,
    687659,
    14279258,
    15115599,
    4972515,
    116131948
]

strat_val = StratVal(
    strata_list=[1, 2, 3, 4, 5, 6, 7, 8],
    class_list=[1, 2, 3, 4, 5, 6, 7, 8],
    n_strata=n_strata,
    samples_df=dfs,
    strata_col="strata",
    ref_class="ref_class",
    map_class="map_class"
)

print(f"accuracy: {strat_val.accuracy()}")
print("")
print(f"user's accuracy: {strat_val.users_accuracy()}")
print("")
print(f"producer's accuracy: {strat_val.producers_accuracy()}")
print("")
print(f"accuracy se: {strat_val.accuracy_se()}")
print("")
print(f"user's accuracy se: {strat_val.users_accuracy_se()}")
print("")
print(f"producer's accuracy se: {strat_val.producers_accuracy_se()}")

