import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import asr_service.config as config


class Pipeline:
    def __init__(self, model_params: config.ModelParams = config.model_params) -> None:
        torch_dtype = torch.float16 if model_params.DEVICE == "cuda:0" else torch.float32

        match model_params.MODEL_SIZE:
            case "large":
                if config.paths.MODEL_LARGE.exists():
                    model_path = config.paths.MODEL_LARGE
                else:
                    "openai/whisper-large-v3"
                processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
            case "medium":
                if config.paths.MODEL_MEDIUM.exists():
                    model_path = config.paths.MODEL_MEDIUM
                else:
                    "openai/whisper-medium"
                processor = AutoProcessor.from_pretrained("openai/whisper-medium")
            case "small":
                if config.paths.MODEL_SMALL.exists():
                    model_path = config.paths.MODEL_SMALL
                else:
                    model_path = "openai/whisper-small"
                processor = AutoProcessor.from_pretrained("openai/whisper-small")
            
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(model_params.DEVICE)


        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=model_params.DEVICE,
        )

    def __call__(self, audio_file_path: str) -> str:
        audio, sr = librosa.load(audio_file_path, sr=16_000)
        pipeline_input = {
            "raw": audio,
            "sampling_rate": sr,
        }
        return self.pipe(pipeline_input)["text"]
