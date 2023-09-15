# lulc-validation

A package to estimate overall accuracy, user's accuracy, and producer's accuracy for a land use / land cover (LULC) map when i) the reference data is generated via a stratified sampling approach, and ii) the strata do not match the map classes. 

This package is based on [Stehman, S.V. (2014)](https://www.tandfonline.com/doi/abs/10.1080/01431161.2014.930207), Estimating area and map accuracy for stratified random sampling when the strata are different from the map classes. *International Journal of Remote Sensing*, 35, 13.

## Install

Clone this repo, then pip install locally. 

```
pip install .
```

or from PyPi.

```
pip install lulc-validation
```

## Use

A sample dataset is provided in this repo at `data/samples.csv`. This is based on Table. 2 in [Stehman (2014)](https://www.tandfonline.com/doi/abs/10.1080/01431161.2014.930207).

Initialise a `StratVal` object:

```
import pandas as pd
import os
from lulc_validation.lulc_val import StratVal

df = pd.read_csv(os.path.join("data", "samples.csv"))

strat_val = StratVal(
    strata_list=[1, 2, 3, 4], # List of labels for strata.
    class_list=[1, 2, 3, 4], # List of labels for LULC map classes.
    n_strata=[40000, 30000, 20000, 10000], # List of the total number of pixels in each strata.
    samples_df=df, # pandas DataFrame of reference data
    strata_col="stratum", # Column label for strata in `samples_df`
    ref_class="ref_class", # Column label for reference classes in `samples_df`
    map_class="map_class" # Column label for map classes in `samples_df`
)
```

A `StratVal` object has the following methods:

* `accuracy()`: returns a float type number representing the overall accuracy of the LULC map accounting for the stratified sampling design of the reference data. This implementation is based on the worked example of *3. Numerical examples* in [Stehman (2014)](https://www.tandfonline.com/doi/abs/10.1080/01431161.2014.930207).
* `users_accuracy()`: returns a dict object where keys indicate the map class and values represent user's accuracy for the corresponding class accounting for the stratified sampling design of the reference data. This implementation is based on the worked example of *3. Numerical examples* in [Stehman (2014)](https://www.tandfonline.com/doi/abs/10.1080/01431161.2014.930207).
* `producers_accuracy()`: returns a dict object where keys indicate the map class and values represent producer's accuracy for the corresponding class accounting for the stratified sampling design of the reference data. This implementation is based on the worked example of *3. Numerical examples* in [Stehman (2014)](https://www.tandfonline.com/doi/abs/10.1080/01431161.2014.930207).
* `accuracy_se()`: returns the standard error of the estimate of the overall accuracy of the LULC map accounting for the stratified sampling design of the reference data. This implementation is based on the worked example of *3. Numerical examples* in [Stehman (2014)](https://www.tandfonline.com/doi/abs/10.1080/01431161.2014.930207).
* `users_accuracy_se()`: returns a dict object where keys indicate the map class and values represent standard errors of the estimates of the user's accuracy for the corresponding class accounting for the stratified sampling design of the reference data. This implementation is based on the worked example of *3. Numerical examples* in [Stehman (2014)](https://www.tandfonline.com/doi/abs/10.1080/01431161.2014.930207).
* `producers_accuracy_se()`: returns a dict object where keys indicate the map class and values represent standard errors of the estimates of the producer's accuracy for the corresponding class accounting for the stratified sampling design of the reference data. This implementation is based on the worked example of *3. Numerical examples* in [Stehman (2014)](https://www.tandfonline.com/doi/abs/10.1080/01431161.2014.930207).

```
print(f"accuracy: {strat_val.accuracy()}")
print(f"user's accuracy: {strat_val.users_accuracy()}")
print(f"producer's accuracy: {strat_val.producers_accuracy()}")
print(f"accuracy se: {strat_val.accuracy_se()}")
print(f"user's accuracy se: {strat_val.users_accuracy_se()}")
print(f"producers's accuracy se: {strat_val.producers_accuracy_se()}")
```

## Development

pytest is used for testing and tests are based on replicating calculations in the worked example of *3. Numerical examples* in [Stehman (2014)](https://www.tandfonline.com/doi/abs/10.1080/01431161.2014.930207).

To run the tests:

```
pytest
``` 