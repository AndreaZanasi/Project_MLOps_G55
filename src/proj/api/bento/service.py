import bentoml
import onnxruntime as ort
import numpy as np


@bentoml.service(workers=4)
class AudioService:
    """Model service for classification"""

    def __init__(self) -> None:
        self.model_session = ort.InferenceSession("models/model.onnx")

    def inference(self, ort_session: ort.InferenceSession, audio):
        input_names = [i.name for i in ort_session.get_inputs()]
        output_names = [o.name for o in ort_session.get_outputs()]
        batch = {input_names[0]: audio.astype(np.float32)}
        output = ort_session.run(output_names, batch)
        return output[0]

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=128,
        max_latency_ms=1000,
    )
    def predict(self, audio_specs: np.ndarray) -> list[dict]:
        audio_specs = np.asarray(audio_specs, dtype=np.float32)
        audio_specs = audio_specs[:, np.newaxis, :, :]

        predictions = self.inference(self.model_session, audio_specs)

        results = []
        for i in range(predictions.shape[0]):
            pred_index = int(np.argmax(predictions[i]))
            results.append({"prediction": pred_index})

        return results
