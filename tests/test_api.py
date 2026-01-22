import torch
import numpy as np
import requests
import pytest
from src.proj.data import MyDataset, preprocess

URL = "https://audio-service-685944380771.europe-west6.run.app/predict"
N_SAMPLES = 10

@pytest.fixture(scope="module")
def test_data():
    ds = MyDataset("data/raw")
    ds.preprocess("data/processed/test")
    data = torch.load("data/processed/test/test.pt", weights_only=False)
    spectrograms = data["spectrograms"].numpy()
    labels = data["labels"].numpy()
    num_samples = len(spectrograms)
    batch_indices = np.random.choice(num_samples, N_SAMPLES, replace=False)
    batch_samples = spectrograms[batch_indices]
    batch_labels = labels[batch_indices]
    return batch_samples, batch_labels

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