from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from transformers import pipeline
import os
from dotenv import load_dotenv
from openai import OpenAI


# curl -X POST "http://0.0.0.0:3000/process_audio" -F "audio_file=@sample1.flac"

# Load environment variables
load_dotenv()

# Initialize OpenAI API client
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

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
async def process_audio(
        audio_file: UploadFile = File(...), language: str = None):
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided")

    print("Audio file received:", audio_file.filename)

    try:
        # Read the audio file
        audio_bytes = await audio_file.read()
        print("Reading audio file...")

        # Use the pipeline to transcribe the audio file, specifying the language if provided
        if language:
            transcription = pipe(audio_bytes, language=language)
        else:
            transcription = pipe(audio_bytes)

        transcribed_text = transcription['text']
        print(f"Transcribed Text: {transcribed_text}")

    except Exception as e:
        print(f"Error during transcription: {e}")
        raise HTTPException(status_code=500,
                            detail="Failed to transcribe audio")

    try:
        print("Summarizing transcribed text using GPT-4...")
        # Step 2: Summarize the transcribed text using GPT-4
        completion = client.chat.completions.create(model="gpt-4",
                                                    messages=[{
                                                        "role":
                                                        "system",
                                                        "content":
                                                        "You are a summarizer."
                                                    }, {
                                                        "role":
                                                        "user",
                                                        "content":
                                                        transcribed_text
                                                    }])
        summary = completion.choices[0].message.content
        print(f"Summary: {summary}")
        return {"summary": summary}

    except Exception as e:
        print(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail="Failed to summarize text")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
