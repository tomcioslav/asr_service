from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

from asr_service.utils import convert_to_wav
from asr_service.pipeline import Pipeline
import asr_service.config as config

pipeline = Pipeline()


app = FastAPI(
    title="ASR API",
    description=(
        "This is an API for automatic speech recognition (ASR). "
        f"It uses {config.model_params.MODEL_SIZE.value} model."
    ),
    version="0.1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/wav_to_text")
async def wav_to_text(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file_path = temp_file.name
        content = await file.read()
        temp_file.write(content)

    # Call the pipeline to process the audio file
    result = pipeline(temp_file_path)

    # Clean up: Delete the temporary file
    os.remove(temp_file_path)

    # Return the text
    return {"text": result["text"]}

@app.post("/webm_to_text")
async def wemb_to_text(file: UploadFile = File(...)):
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file_path = temp_file.name
        content = await file.read()
        temp_file.write(content)
        
    wav_file_path = convert_to_wav(temp_file.name, output_path=temp_file_path)
        
    result = pipeline(temp_file_path)

    # Clean up: Delete the temporary file
    os.remove(temp_file_path)

    # Return the text
    return {"text": result["text"]}