import numpy as np
import requests
import pytest
from tests import SHAPE

URL = "https://audio-service-685944380771.europe-west6.run.app/predict"
N_SAMPLES = 10


@pytest.fixture(scope="module")
def test_data():
    rng = np.random.default_rng(42)
    num_samples = N_SAMPLES
    spectrograms = rng.normal(size=(num_samples, *SHAPE)).astype(float)
    labels = rng.integers(low=0, high=10, size=(num_samples,))
    return spectrograms, labels


@pytest.mark.parametrize("idx", range(N_SAMPLES))
def test_predict_response_no_error(test_data, idx):
    batch_samples, _ = test_data
    sample = batch_samples[idx]
    response = requests.post(URL, json={"audio_specs": sample.tolist()})
    assert response.status_code == 200
    result = response.json()
    if isinstance(result, list):
        result = result[0]
    assert "prediction" in result
    assert result["prediction"] != "error"
