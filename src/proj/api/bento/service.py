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

    @bentoml.api
    def preprocess_audio(self, input_data):
        target_sr = 32000
        n_mels = 64
        n_fft = 1024
        hop_length = 320
        target_length = 1168

        audio_array = np.array(input_data["audio_array"], dtype=np.float32)
        sr = input_data.get("sr", 32000)

        if sr != target_sr:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)

        mel_spec = librosa.feature.melspectrogram(
            y=audio_array,
            sr=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=50,
            fmax=14000,
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
        if mel_spec_db.shape[1] < target_length:
            pad_width = target_length - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=-80.0)
        else:
            mel_spec_db = mel_spec_db[:, :target_length]

        return mel_spec_db[np.newaxis, :, :].astype(np.float32)

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=128,
        max_latency_ms=1000,
    )
    def predict(self, audio_specs: np.ndarray) -> list[dict]:
        if audio_specs.ndim == 3 and audio_specs.shape[0] == 1:
            audio_specs = np.expand_dims(audio_specs, axis=1)
        
        predictions = inference(self.model_session, audio_specs)

        results = []
        for i in range(predictions.shape[0]):
            pred_index = np.argmax(predictions[i])
            results.append({"prediction": int(pred_index)})  
        return results