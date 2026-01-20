import bentoml
from proj.api.export_onnx import inference
import onnxruntime as ort
import librosa
import numpy as np
import numpy.typing as npt

@bentoml.service(workers=4)
class AudioService:
    """Model service for classification"""

    def __init__(self) -> None:
        self.model_session = ort.InferenceSession("models/model.onnx")

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=128,
        max_latency_ms=1000,
    )
    def predict(self, audio_specs: np.ndarray) -> list[dict]:
        audio_specs = np.asarray(audio_specs, dtype=np.float32)
        audio_specs = audio_specs[:, np.newaxis, :, :]

        predictions = inference(self.model_session, audio_specs)

        results = []
        for i in range(predictions.shape[0]):
            pred_index = int(np.argmax(predictions[i]))
            results.append({"prediction": pred_index})

        return results