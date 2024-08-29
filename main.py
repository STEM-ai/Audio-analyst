import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import openai

# Load environment variables
load_dotenv()

# Initialize OpenAI API client
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

# Load the Whisper model and processor from Hugging Face
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Initialize FastAPI application
app = FastAPI(
    title="Whisper to GPT-4 Summarizer",
    version="1.0",
    description="An API server that transcribes audio using Whisper and summarizes the text using GPT-4",
)

# Add a default GET route to the root URL
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Whisper to GPT-4 Summarizer API"}

@app.post("/process_audio")
async def process_audio(audio_file: UploadFile = File(...)):
    print("Received request at /process_audio endpoint")

    # Check if the audio file is provided
    if not audio_file:
        print("No audio file provided")
        raise HTTPException(status_code=400, detail="No audio file provided")

    try:
        print("Reading audio file...")
        # Step 1: Read the audio file
        audio_bytes = await audio_file.read()

        print("Processing audio file with Whisper processor...")
        # Convert bytes to audio array as needed for Whisper
        input_features = processor(audio_bytes, return_tensors="pt", sampling_rate=16000)["input_features"]

        print("Transcribing audio to text using Whisper model...")
        # Transcribe audio to text using Whisper
        predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(f"Transcribed Text: {transcription}")

    except Exception as e:
        print(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail="Failed to transcribe audio")

    try:
        print("Summarizing transcribed text using GPT-4...")
        # Step 2: Summarize the transcribed text using GPT-4
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a summarizer."},
                {"role": "user", "content": transcription}
            ]
        )
        summary = gpt_response["choices"][0]["message"]["content"]
        print(f"Summary: {summary}")

        return {"summary": summary}

    except Exception as e:
        print(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail="Failed to summarize text")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)