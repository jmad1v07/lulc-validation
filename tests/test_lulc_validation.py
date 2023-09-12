import pytest
import pandas as pd

from lulc_validation.lulc_val import StratVal

def test_accuracy(sample_reference_data):

    strat_val = StratVal(
        strata_list=[1, 2, 3, 4],
        class_list=[1, 2, 3, 4],
        n_strata=[40000, 30000, 20000, 10000],
        samples_df=sample_reference_data,
        strata_col="strata",
        ref_class="ref_class",
        map_class="map_class"
    )

    accuracy = strat_val.accuracy()

    assert (round(accuracy, 2) == 0.63)

def test_users_accuracy(sample_reference_data):

    strat_val = StratVal(
        strata_list=[1, 2, 3, 4],
        class_list=[1, 2, 3, 4],
        n_strata=[40000, 30000, 20000, 10000],
        samples_df=sample_reference_data,
        strata_col="strata",
        ref_class="ref_class",
        map_class="map_class"
    )

    users_accuracy = strat_val.users_accuracy()

    assert (round(users_accuracy["2"], 3) == 0.574)

def test_producers_accuracy(sample_reference_data):

    strat_val = StratVal(
        strata_list=[1, 2, 3, 4],
        class_list=[1, 2, 3, 4],
        n_strata=[40000, 30000, 20000, 10000],
        samples_df=sample_reference_data,
        strata_col="strata",
        ref_class="ref_class",
        map_class="map_class"
    )

    producers_accuracy = strat_val.producers_accuracy()

    assert (round(producers_accuracy["2"], 3) == 0.794)

def test_accuracy_se(sample_reference_data):
    strat_val = StratVal(
        strata_list=[1, 2, 3, 4],
        class_list=[1, 2, 3, 4],
        n_strata=[40000, 30000, 20000, 10000],
        samples_df=sample_reference_data,
        strata_col="strata",
        ref_class="ref_class",
        map_class="map_class"
    )

    accuracy_se = strat_val.accuracy_se()

    assert(round(accuracy_se, 3) == 0.085)

def test_users_accuracy_se(sample_reference_data):

    strat_val = StratVal(
        strata_list=[1, 2, 3, 4],
        class_list=[1, 2, 3, 4],
        n_strata=[40000, 30000, 20000, 10000],
        samples_df=sample_reference_data,
        strata_col="strata",
        ref_class="ref_class",
        map_class="map_class"
    )

    users_accuracy_se = strat_val.users_accuracy_se()

    assert (round(users_accuracy_se["2"], 3) == 0.125)

def test_producers_accuracy_se(sample_reference_data):
    """
    The worked example of Stehman (2014) returns a standard error
    for the Producer's accuracy of class B / 2 as 0.114. Here,
    we get 0.117. This seems to be because s_2_xh has been assigned a 
    value of 0.00 for strata 4 in Stehman's (2014) worked example but 
    the actual value of s_2_xh for strata 4 is 0.10 (see producer's 
    accuracy column in Table 3).
    """

    strat_val = StratVal(
        strata_list=[1, 2, 3, 4],
        class_list=[1, 2, 3, 4],
        n_strata=[40000, 30000, 20000, 10000],
        samples_df=sample_reference_data,
        strata_col="strata",
        ref_class="ref_class",
        map_class="map_class"
    )

    producers_accuracy_se = strat_val.producers_accuracy_se()

    assert (round(producers_accuracy_se["2"], 3) == 0.117)