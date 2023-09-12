import lulc_validation
import pandas as pd
from lulc_validation.lulc_val import StratVal

df = pd.read_csv("samples.csv")

strat_val = StratVal(
    strata_list=[1, 2, 3, 4],
    class_list=[1, 2, 3, 4],
    n_strata=[40000, 30000, 20000, 10000],
    samples_df=df,
    strata_col="strata",
    ref_class="ref_class",
    map_class="map_class"
)

print(f"accuracy: {strat_val.accuracy()}")
print(f"user's accuracy: {strat_val.users_accuracy()}")
print(f"producer's accuracy: {strat_val.producers_accuracy()}")
print(f"accuracy se: {strat_val.accuracy_se()}")
print(f"user's accuracy se: {strat_val.users_accuracy_se()}")
print(f"producers's accuracy se: {strat_val.producers_accuracy_se()}")