import pytest
import pandas as pd
import os

@pytest.fixture
def sample_reference_data():
    df = pd.read_csv(os.path.join(os.getcwd(), "data", "samples.csv"))

    return df
