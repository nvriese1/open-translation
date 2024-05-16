# Base modules / libraries
import os
import uvicorn
import json 
import numpy as np
import logging
import asyncio
import itertools
import warnings
import time
from natsort import natsorted
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import *

from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException, Depends, APIRouter
from fastapi.responses import JSONResponse

# Start logging
logging.basicConfig(level=logging.INFO, force=True) 

# Custom modules / libraries
from utils.default import *

# Load environment
ENV = os.environ.get('ENV', ".env") 
load_dotenv(ENV, override=True)
logging.debug(f'app.py: reading environment variables from ENV: {ENV}')

# Deployment configuration
LOCAL_DEPLOYMENT = os.getenv('LOCAL_DEPLOYMENT', 'false').lower() == 'true'  
PORT = os.environ.get('PORT')
CONTAINER_PATH = os.getcwd()
MODEL_SAVE_PATH = os.path.join(CONTAINER_PATH, 'models')
LANG_DET_PATH = os.path.join(MODEL_SAVE_PATH, 'lang_det')
LANG_TR_PATH = os.path.join(MODEL_SAVE_PATH, 'lang_tr')

logging.info(f'LOCAL_DEPLOYMENT: {LOCAL_DEPLOYMENT}')
logging.info(f'CONTAINER_PATH: {CONTAINER_PATH}')
logging.info(f'MODEL_SAVE_PATH: {MODEL_SAVE_PATH}')
logging.info(f'PORT: {PORT}')

model_dict: dict = {
    'lang_det': LANG_DET_PATH,
    'lang_tr': LANG_TR_PATH,
}

def load_model_sync(model_type: str, model_path: str) -> object:
    
    DEVICE = select_device()
    logging.info(f"Running model: {model_type} on DEVICE: {DEVICE}")
    
    if model_type == 'lang_det':
        return LanguageDetector(
            model=AutoModelForSequenceClassification.from_pretrained(
                model_path,
            ),
            tokenizer=AutoTokenizer.from_pretrained(
                model_path,
            ),
            device=DEVICE,
        )
        
    elif model_type == 'lang_tr':
        return LanguageTranslator(
            model=MBartForConditionalGeneration.from_pretrained(
                model_path,
            ),
            tokenizer=MBart50TokenizerFast.from_pretrained(
                model_path,
            ),
            device=DEVICE,
        )

async def load_model_async(model_type: str, model_path: str) -> object:
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(
            pool, load_model_sync, model_type, model_path
        )

# Main model loader
async def load_models(model_dict: dict) -> Dict[str, object]:
    if not isinstance(model_dict, dict):
        raise ValueError(
            f'load_models() requires a model_dict of type "dict", {type(model_dict)} passed.'
        )
    
    tasks = []
    for model_type, model_path in model_dict.items():
        model_type_lower = model_type.lower()
        task = asyncio.create_task(load_model_async(model_type_lower, model_path))
        tasks.append(task)

    loaded_models = await asyncio.gather(*tasks)
    return dict(zip(model_dict.keys(), loaded_models))

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # Startup logic
    logging.info(f"Loading models...")
    app.state.loaded_models = await load_models(model_dict)
    yield
    # Shutdown logic
    del app.state.loaded_models

app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def example_auth_middleware(request: Request, call_next: Callable):
    
    authorization: str = request.headers.get("Authorization", None)
    token = ""
    
    if authorization is not None:
        scheme, token = authorization.split()
    
    request.state.token = token
    response = await call_next(request)
    return response


@app.get('/')
async def health_check():
    return {"message": "No issues to report."}

   
@app.post('/detect_translate')
async def detect_translate(raw_request: Request):
    
    try:
        request = await raw_request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    logging.info(f"Request recieved: {request}")
    
    # Get prompt args
    text: str = request.get('text', '')
    translate_to: Union[Literal['auto'], str] = request.get('translate_to', 'auto')
    reconstruct_formatting: bool = request.get('reconstruct_formatting', False)
    
    # Load models
    language_detector = raw_request.app.state.loaded_models.get('lang_det')
    language_translator = raw_request.app.state.loaded_models.get('lang_tr')
    
    # Type checking and task routing
    if (isinstance(translate_to, str)):
                
        if translate_to.lower() == 'auto':
            # Used on queries: detect incoming language and translate to english if needed
            
            # Detect language
            detection_scores: dict = await language_detector.detect(text=text)
            detected_language: str = language_detector.det_lang
            logging.info(f"Detected language / scores: {detection_scores}")
            
            if not detected_language == 'en':
                
                # Translate if needed
                logging.info(f"Translating from '{detected_language}' to 'en'...")
                translation_needed = True
                translation_text = await language_translator.translate(
                    text=text, 
                    source_lang=detected_language, 
                    target_lang='en',
                    reconstruct_formatting=reconstruct_formatting,
                )
                logging.info(f"Translated text: {translation_text}")
            
            else:
                translation_needed = False
                translation_text = text
            
            translation_metadata = {
                'source_language': detected_language,
                'target_language': 'en',
                'translate_to': 'auto',
                'translation_needed': translation_needed,
            }
            
        else:
            # Used on translations back to original language on response
            
            # Translate 
            translation_text = await language_translator.translate(
                text=text, 
                source_lang='en', 
                target_lang=translate_to.lower(),
                reconstruct_formatting=reconstruct_formatting,
            )
            logging.info(f"Translated text: {translation_text}")
            translation_metadata = {
                'source_language': 'en',
                'target_language': translate_to.lower(),
                'translate_to': translate_to.lower(),
            }
        
    else:
        raise ValueError(
            f'Unexpected type for argument "translate_to", expected type(<str>),\
                received type {type(translate_to)} and value {str(translate_to)}.'
            )
   
    # Return response 
    return {'text': translation_text, 'metadata': translation_metadata}
    


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=PORT)