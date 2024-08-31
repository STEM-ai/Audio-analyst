from fastapi import FastAPI, Form, HTTPException, Request
import uvicorn
from transformers import pipeline
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import time


# curl -X POST "http://0.0.0.0:3000/process_audio" -F "audio_file=@sample1.flac"

# Load environment variables
load_dotenv()

# Initialize OpenAI API client
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

TWILIO_ACCOUNT_SID = os.environ['TWILIO_ACCOUNT_SID']
TWILIO_AUTH_TOKEN = os.environ['TWILIO_AUTH_TOKEN']

# Initialize the Whisper pipeline
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")

# Initialize FastAPI application
app = FastAPI(
    title="Whisper to GPT-4 Summarizer",
    version="1.0",
    description=
    "An API server that transcribes audio using Whisper and summarizes the text using GPT-4",
)


# Add a default GET route to the root URL
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Whisper to GPT-4 Summarizer API"}


@app.post("/process_audio")

#    audio_file: UploadFile = File(...), language: str = None):
#if not audio_file:
#    raise HTTPException(status_code=400, detail="No audio file provided")

#print("Audio file received:", audio_file.filename)

try:
    # Read the audio file
    audio_bytes = await audio_file.read()
async def process_audio(request: Request):
    try:
        # Read the form data once
        form_data = await request.form()
        print("Received form data:", form_data)

        # Extract RecordingUrl from form data
        RecordingUrl = form_data.get('RecordingUrl')
        if not RecordingUrl:
            print("RecordingUrl not found in form data")
            raise HTTPException(status_code=400, detail="RecordingUrl is missing")

        print(f"Received RecordingUrl: {RecordingUrl}")

        # Add a delay to ensure Twilio has processed the recording
        time.sleep(5)    

        # Download the audio file from Twilio's Recording URL
        audio_response = requests.get(RecordingUrl, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        audio_response.raise_for_status()  # Ensure we handle HTTP errors
        audio_bytes = audio_response.content
        print("Audio file downloaded from Twilio")

        # Convert audio to a format that can be processed
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)

        # Use the pipeline to transcribe the audio file
        transcription = pipe("temp_audio.wav")
        transcribed_text = transcription['text']
        print(f"Transcribed Text: {transcribed_text}")

        # Summarize with GPT-4
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Résume le message de la boîte vocal. Ton résumé doit être en français, et détaillé."},
                {"role": "user", "content": transcribed_text}
            ]
        )
        summary = completion.choices[0].message.content
        print(f"Summary: {summary}")

        return {"summary": summary}

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Failed to process audio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)