import torch
import numpy as np
import requests

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

data = torch.load("data/processed/test/test.pt", weights_only=False)
spectrograms = data["spectrograms"].numpy()
labels = data["labels"].numpy()
num_samples = len(spectrograms)

url = "http://127.0.0.1:3000/predict"
total = 100

batch_indices = np.random.choice(num_samples, total, replace=False)
batch_samples = spectrograms[batch_indices]
batch_labels = labels[batch_indices]

correct = 0

for i, sample in enumerate(batch_samples):
    response = requests.post(url, json={"audio_specs": sample.tolist()})
    result = response.json()
    if isinstance(result, list):
        result = result[0]

    pred = result.get("prediction", "error")

    if pred == batch_labels[i]:
        color = GREEN
        correct = correct + 1
    else: color = RED

    print(f"Sample {i}: Predicted={color}{pred}{RESET}, True={batch_labels[i]}")

print(f"Accuracy: {correct}/{total}")