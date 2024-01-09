from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os

from asr_service.pipeline import Pipeline

pipeline = Pipeline()


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/to_text")
async def to_text(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file_path = temp_file.name
        content = await file.read()
        temp_file.write(content)

    # Call the pipeline to process the audio file
    text = pipeline(temp_file_path)

    # Clean up: Delete the temporary file
    os.remove(temp_file_path)

    # Return the text
    return {"text": text}
