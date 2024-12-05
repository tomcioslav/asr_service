import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import asr_service.config as config


class Pipeline:
    def __init__(self, model_params: config.ModelParams = config.model_params) -> None:
        torch_dtype = torch.float16 if model_params.DEVICE == "cuda:0" else torch.float32

        match model_params.MODEL_SIZE:
            case "small":
                model_id = "openai/whisper-small"
                model_path = config.paths.MODEL_SMALL
                
                # Download and save the model and processor to our configured path
                if not model_path.exists():
                    processor = AutoProcessor.from_pretrained(model_id)
                    processor.save_pretrained(model_path)
                    
                    model = AutoModelForSpeechSeq2Seq.from_pretrained(
                        model_id,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                        use_safetensors=True,
                    )
                    model.save_pretrained(model_path)
                else:
                    processor = AutoProcessor.from_pretrained(model_path)
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
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=model_params.DEVICE,
            generate_kwargs={
                "language": "en", 
                "task": "transcribe",
                "max_new_tokens": 128
            }
        )

    def __call__(self, audio_path: str) -> dict:
        # Load audio and ensure it's mono
        speech, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Ensure the audio is properly shaped (should be 1D array)
        if len(speech.shape) > 1:
            speech = speech.squeeze()

        return self.pipe(
            inputs=speech
        )
