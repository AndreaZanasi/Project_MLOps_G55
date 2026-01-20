import torch
import requests
import numpy as np

data = torch.load("data/processed/test/test.pt", weights_only=False)
url = "http://127.0.0.1:3000/predict"

audio_sample = data["spectrograms"][0].numpy()
label = data["labels"][0].item()

resp = requests.post(
    url,
    json={"audio_specs": audio_sample.tolist()},
)
print("Single sample output:", resp.json())
print("Ground truth label:", label)

print("\n" + "="*50 + "\n")

batch_samples = data["spectrograms"][:5].numpy()
batch_labels = data["labels"][:5].numpy()

batch_results = []
for i, sample in enumerate(batch_samples):
    resp = requests.post(
        url,
        json={"audio_specs": sample.tolist()},
    )
    
    result = resp.json()
    
    if isinstance(result, list):
        result = result[0]
        
    batch_results.append(result)
    
    print(f"Sample {i}: Predicted={result.get('prediction', 'error')}, True={batch_labels[i]}")

print("\nBatch labels:", batch_labels.tolist())