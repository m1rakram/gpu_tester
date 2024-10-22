from fastapi import FastAPI
import time
import asyncio
import uvicorn
import random
import string
from whisper import load_audio


def id_generator(size=50, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

app = FastAPI()

from transformers import pipeline
#pipe = pipeline("text-to-audio", model="BHOSAI/tts_news_speaker_female")
pipe = pipeline("text-to-audio", model="BHOSAI/tts_news_speaker_female_mixed",  device = "cuda")
pipe2 = pipeline("automatic-speech-recognition", model="openai/whisper-small", device = "cuda")

def transcription():
    
    waveform = load_audio("test4.wav")
    transcription = pipe2(waveform)
    return transcription


async def synthesis():
    loop = asyncio.get_event_loop()
    speech = await loop.run_in_executor(None, pipe, id_generator())
    return "done"

async def asr():
    loop = asyncio.get_event_loop()
    speech = await loop.run_in_executor(None, transcription)
    return speech


@app.get("/home")
async def root():
    """
    My home route
    """
    start = time.time()
    a = await synthesis()
    end = time.time()
    print('It took {} seconds to finish execution.'.format(end-start))

    return {
        'a': "selem",
    }
    
    
    
@app.get("/asr")
async def root_asr():
    """
    My home route
    """
    start = time.time()
    a = await asr()
    end = time.time()
    print('It took {} seconds to finish execution.'.format(end-start))

    return {
        'a': a,
    }
        
    
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=3579, reload=True)