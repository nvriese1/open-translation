import os
import re
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

from requests.auth import HTTPBasicAuth
import torch
import torch.nn.functional as F
import aiohttp

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaForSequenceClassification 
from transformers.models.xlm_roberta.tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast 
from transformers.models.mbart.modeling_mbart import MBartForConditionalGeneration 
from transformers.models.mbart50.tokenization_mbart50_fast import MBart50TokenizerFast 

logger = logging.getLogger('translation')

ENV = load_dotenv()
if ENV:    
    # Deployment configuration
    LOCAL_DEPLOYMENT = os.getenv('LOCAL_DEPLOYMENT', 'false').lower() == 'true' 
    PORT = os.environ.get('PORT')
    
def select_device() -> Literal['mps', 'cuda', 'cpu']:
    """Selects the most appropriate device for ops based on hardware."""
    # Check for mps (apple silicon GPUs)
    if torch.backends.mps.is_available():
        return 'mps'
    # Check for cuda (nvidia GPUs)
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

DEVICE = select_device()

def batch_operation(
    items: List[Any],
    operation: Callable, 
    **kwargs,
) -> List[Any]:
    """Performs a single operation on a batch of items with optional keyword arguments."""
    return [operation(item, **kwargs) for item in items]

async def fastmap(
    iterable: Iterable, 
    operation: Callable, 
    batch_size: int = 5, 
    max_workers: int = 100, 
    **kwargs,
) -> List:
    """
    Maps an operation over an iterable with 'max_workers' workers and optional kwargs, maintaining input order.
    
    Parameters:
    - iterable: An Iterable of any type.
    - operation: A Callable that applies to a batch of items from iterable.
    - batch_size: Number of items in each batch.
    - max_workers: The maximum number of worker threads to use.
    - kwargs: Additional keyword arguments to pass to the operation.
    """
    if not iterable:
        return iterable
    
    batch_size = batch_size if batch_size <= len(iterable) else len(iterable)
    batches = [iterable[i:i + batch_size] for i in range(0, len(iterable), batch_size)]
    optimal_workers = min(len(batches), max_workers)
    loop = asyncio.get_running_loop()
    
    with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
        futures = [loop.run_in_executor(executor, lambda b=batch: batch_operation(b, operation, **kwargs)) for batch in batches]
        results = await asyncio.gather(*futures)
        return [item for sublist in results for item in sublist]

class LanguageDetector:
    def __init__(
        self, 
        model: Union[None, str, XLMRobertaForSequenceClassification] = None, 
        tokenizer: Union[None, str, XLMRobertaTokenizerFast] = None,
        device: Union[None, str] = None,
    ) -> None:
        """
        Initializes the LanguageDetector with a specified model and tokenizer for detecting the language of a given text.
        Automatically loads a default pretrained XLM-Roberta model and tokenizer if none are specified, or users can provide
        custom model and tokenizer paths or instances for language detection.

        :param model: The language detection model or the path to the pretrained model. Defaults to a predefined XLM-Roberta model for language detection.
        :type model: Union[None, str, XLMRobertaForSequenceClassification], optional
        :param tokenizer: The tokenizer or the path to the tokenizer associated with the language detection model. Defaults to the tokenizer corresponding to the predefined XLM-Roberta model.
        :type tokenizer: Union[None, str, XLMRobertaTokenizerFast], optional
        :param device: The device to run the model on (CPU/GPU). Automatically determined based on availability and 'device_map' configuration of the model.
        :type device: Union[None, str], optional

        :raises ValueError: If the provided model or tokenizer types are not supported.

        Note:
            The class sets the model to evaluation mode for inference.

        Example::

            >>> language_detector = LanguageDetector()
            >>> result = language_detector.detect("This is an example text.")
            >>> print(result)
        """    
        
        self.__init_models__(model, tokenizer, device)
        return
    
    def __init_models__(self, model, tokenizer, device) -> None:
        """
        Initializes or loads the language detection model and tokenizer.

        :param model: The language detection model or the path to the pretrained model. If None, a default model is loaded.
        :type model: Union[str, XLMRobertaForSequenceClassification], optional
        :param tokenizer: The tokenizer or the path to the tokenizer. If None, a default tokenizer is loaded.
        :type tokenizer: Union[str, XLMRobertaTokenizerFast], optional
        :param device: The device to run the model on. Automatically determined if None.
        :type device: str, optional

        :raises ValueError: If the provided model or tokenizer types are not supported.
        """
        
        # Verify and load language detection model if not provided
        if model is None:
            # No model or model path passed
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "models/lang_det",
                device_map='auto'
            )
        elif isinstance(model, str):
            # Model path passed
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model,
                device_map='auto'
            )
        elif isinstance(model, XLMRobertaForSequenceClassification):
            # Model passed
            self.model = model
        else:
            raise ValueError(
                f'Unsupported model for language detection, type: "{type(model)}", \
                    expected type: "transformers.models.xlm_roberta.modeling_xlm_roberta.XLMRobertaForSequenceClassification"'
            )
        
        # Verify and load language detection tokenizer if not provided
        if tokenizer is None:
            # No tokenizer or tokenizer path passed
            self.tokenizer = AutoTokenizer.from_pretrained(
                "models/lang_det"
            )
        elif isinstance(tokenizer, str):
            # Tokenizer path passed
            self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(
                tokenizer,
            )
        elif isinstance(tokenizer, XLMRobertaTokenizerFast):
            # Tokenizer passed
            self.tokenizer = tokenizer
        else:
            raise ValueError(
                f'Unsupported tokenizer for language detection, type: "{type(tokenizer)}", \
                    expected type: "transformers.models.xlm_roberta.tokenization_xlm_roberta_fast.XLMRobertaTokenizerFast"'
            )
        
        if device is not None:
            self.device = device
        elif DEVICE is not None:
            self.device = DEVICE
        else:
            self.device = 'cpu'
        
        logging.info(f'Running Language Detection on device: {self.device}')
        self.model.to(self.device)
        self.model.eval()
        return
        
    async def detect(self, text: str) -> Dict:
        """
        Detects the language of the provided text using the loaded model and tokenizer.

        :param text: The text for which to detect the language.
        :type text: str

        :return: A dictionary where keys are language codes and values are confidence scores for the respective languages.
        :rtype: Dict[str, float]

        Example::

            >>> result = language_detector.detect("Este es un texto de ejemplo.")
            >>> print(result)
            # Output might be something like: {'es': 0.99, 'en': 0.01}
        """
        
        if not text.strip():
            self.det_lang = 'en'
            return {'en': 0.99}
        
        inputs: BatchEncoding = self.tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits: BatchEncoding = self.model(**inputs).logits
        
        # Predict and map raw predictions to languages
        preds: torch.Tensor = torch.softmax(logits.float(), dim=-1)
        vals, idxs = torch.max(preds, dim=1)

        scores: Dict = {self.model.config.id2label[k.item()]: v.item() for k, v in zip(idxs, vals)}
        self.det_lang = next(iter(scores))
        
        # Handle edge case for low confidence predictions on short queries
        if (self.det_lang == 'sw') and (len(str(text)) < 10):
            self.det_lang = 'en'
            scores = {'en': 0.99}
        
        return scores


class LanguageTranslator:
    def __init__(self, model=None, tokenizer=None, device=None) -> None:
        """
        Initializes the LanguageTranslator with a specified model and tokenizer for translating text between languages.
        Automatically loads a default pretrained mBART model and tokenizer if none are specified, or users can provide
        custom model and tokenizer paths or instances for language translation.

        :param model: The language translation model or the path to the pretrained model. Defaults to a predefined mBART model.
        :type model: Union[None, str, MBartForConditionalGeneration], optional
        :param tokenizer: The tokenizer or the path to the tokenizer associated with the language translation model. Defaults to the tokenizer corresponding to the predefined mBART model.
        :type tokenizer: Union[None, str, MBart50TokenizerFast], optional
        :param device: The device to run the model on (CPU/GPU). Automatically determined based on availability.
        :type device: Union[None, str], optional

        :raises ValueError: If the provided model or tokenizer types are not supported.

        Note:
            - This class is designed to work with asynchronous functions for translation to handle potentially long-running translation tasks efficiently.
            - Assumes DEVICE is a global variable specifying the execution device. If DEVICE is not defined, defaults to CPU.

        Example::

            >>> language_translator = LanguageTranslator()
            >>> translated_text = await language_translator.translate("This is an example text.", source_lang='en', target_lang='fr')
            >>> print(translated_text)
        """    
        
        self.__init_models__(model, tokenizer, device)
        self.__init_langs__()
        # Regex characters to preserve response formatting along
        self.regex_chars = ['^\s*[\*\-\+]', '\*\*', '`{3}', '\n\n+', '\n', '\t', '   ']
        # Markdown characters to preserve response formatting along (requires same ordering as self.split_chars)
        self.markdown_chars = ['*', '**', '```', '\n\n', '\n', '\t', '   ']
        return
    
    def __init_models__(self, model, tokenizer, device) -> None:
        """
        Initializes the translation model and tokenizer. Verifies input arguments and loads the appropriate model and tokenizer.

        :param model: The translation model or the path to the pretrained model. If None, a default model is loaded.
        :type model: Union[None, str, MBartForConditionalGeneration], optional
        :param tokenizer: The tokenizer or the path to the tokenizer. If None, a default tokenizer is loaded.
        :type tokenizer: Union[None, str, MBart50TokenizerFast], optional
        :param device: The device to run the model on. If None, defaults to CPU or DEVICE global variable.
        :type device: str, optional

        :raises ValueError: If the model or tokenizer provided are of unsupported types.
        """
        
        # Verify and load language detection model if not provided
        if model is None:
            self.model = MBartForConditionalGeneration.from_pretrained(
                "models/lang_tr",
                device_map='auto'
            )
        elif isinstance(model, MBartForConditionalGeneration):
            self.model = model
        else:
            raise ValueError(
                f'Unsupported model for language translation, type: "{type(model)}", \
                    expected type: "transformers.models.mbart.modeling_mbart.MBartForConditionalGeneration"'
            )
        
        # Verify and load language detection tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = MBart50TokenizerFast.from_pretrained(
                "models/lang_tr"
            )
        elif isinstance(tokenizer, MBart50TokenizerFast):
            self.tokenizer = tokenizer
        else:
            raise ValueError(
                f'Unsupported tokenizer for language translation, type: "{type(tokenizer)}", \
                    expected type: "transformers.models.mbart50.tokenization_mbart50_fast. MBart50TokenizerFast"'
            )
        
        if device is not None:
            self.device = device
        elif DEVICE is not None:
            self.device = DEVICE
        else:
            self.device = 'cpu'
        
        logging.info(f'Running Language Translation on device: {self.device}')
        self.model.to(self.device)
        self.model.eval()
        # N tokens translation model can output in a single shot
        self.max_sequence_len: int = 200 
        return
    
    def __init_langs__(self) -> None:
        """
        Private method to initialize mappings between language detection model (xlm-roberta-base-language-detection), 
        and language translation model (mbart-large-50-many-to-many-mmt).
        """
        
        self.lang_dict = {
            'ar': 'ar_AR', 'cs': 'cs_CZ', 'de': 'de_DE',
            'en': 'en_XX', 'es': 'es_XX', 'et': 'et_EE',
            'fi': 'fi_FI', 'fr': 'fr_XX', 'gu': 'gu_IN',
            'hi': 'hi_IN', 'it': 'it_IT', 'ja': 'ja_XX',
            'kk': 'kk_KZ', 'ko': 'ko_KR', 'lt': 'lt_LT',
            'lv': 'lv_LV', 'my': 'my_MM', 'ne': 'ne_NP',
            'nl': 'nl_XX', 'ro': 'ro_RO', 'ru': 'ru_RU',
            'si': 'si_LK', 'tr': 'tr_TR', 'vi': 'vi_VN',
            'zh': 'zh_CN', 'af': 'af_ZA', 'az': 'az_AZ',
            'bn': 'bn_IN', 'fa': 'fa_IR', 'he': 'he_IL',
            'hr': 'hr_HR', 'id': 'id_ID', 'ka': 'ka_GE',
            'km': 'km_KH', 'mk': 'mk_MK', 'ml': 'ml_IN',
            'mn': 'mn_MN', 'mr': 'mr_IN', 'pl': 'pl_PL',
            'ps': 'ps_AF', 'pt': 'pt_XX', 'sv': 'sv_SE',
            'sw': 'sw_KE', 'ta': 'ta_IN', 'te': 'te_IN',
            'th': 'th_TH', 'tl': 'tl_XX', 'uk': 'uk_UA',
            'ur': 'ur_PK', 'xh': 'xh_ZA', 'gl': 'gl_ES',
            'sl': 'sl_SI'
        }
        
    async def __batch_sentences(self, text: Union[str, List[str]], max_tokens: int, format_map: Optional[List[Optional[int]]] = None) -> Tuple[List[List[str]], Optional[List[Optional[int]]]]:
        """
        Splits the input text into batches that do not exceed the maximum token limit of the model.
        Necessary for processing long texts that might exceed the model's maximum sequence length.

        :param text: The input text or list of texts to be split into batches.
        :type text: Union[str, List[str]]
        :param max_tokens: The maximum number of tokens allowed in a single batch.
        :type max_tokens: int
        :param format_map: The formatting map for the text, with integers representing formatting markers and None for regular text segments.
        :type format_map: Optional[List[Optional[int]]], optional

        :return: A tuple containing a list of batches and the potentially updated format map.
        :rtype: Tuple[List[List[str]], Optional[List[Optional[int]]]]

        Example::
            >>> batches, format_map = await self.__batch_sentences(text, max_tokens=200, format_map=None)
        """
        if isinstance(text, str):        
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        elif isinstance(text, list):
            paragraphs = text

        batches = []
        format_map_index = 0  

        for paragraph in paragraphs:
            # Check if the entire paragraph is within the token limit
            paragraph_tokens = len(self.tokenizer.tokenize(paragraph))
            if paragraph_tokens <= max_tokens:
                batches.append([paragraph])
                format_map_index += 1
            else:
                # If the paragraph exceeds the token limit, split it into sentences
                sentences = [s.strip() + '. ' for s in paragraph.split('. ') if s.strip()]
                current_batch = []
                current_tokens = 0

                for sentence in sentences:
                    sentence_tokens = len(self.tokenizer.tokenize(sentence))
                    if current_tokens + sentence_tokens > max_tokens:
                        if current_batch:
                            batches.append(current_batch)
                            current_batch = [sentence]
                            current_tokens = sentence_tokens
                            # Insert None in the format_map at the current index for each new sentence created by splitting
                            if format_map is not None:
                                format_map.insert(format_map_index, None)
                        else:
                            current_batch = [sentence]
                            current_tokens = sentence_tokens
                    else:
                        current_batch.append(sentence)
                        current_tokens += sentence_tokens

                if current_batch:
                    batches.append(current_batch)
                    format_map_index += 1

        return (batches, format_map)

    async def __translate_batch(self, batch: List[str], source_lang: str, target_lang: str) -> str:
        """
        Translates a batch of sentences from the source language to the target language using the loaded model and tokenizer.

        :param batch: A batch of sentences to be translated.
        :type batch: List[str]
        :param source_lang: The source language code.
        :type source_lang: str
        :param target_lang: The target language code.
        :type target_lang: str

        :return: The translated text for the given batch.
        :rtype: str

        Example::
            >>> translation = await self.__translate_batch(batch, source_lang='en', target_lang='fr')
        """
        batch_text = ' '.join(batch)
        self.tokenizer.src_lang = source_lang
        encoded = self.tokenizer(batch_text, return_tensors="pt").to(self.device)
        
        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang]
        )

        translation = self.tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )
        return translation[0]
    
    async def __extract_formatting(self, text: str) -> Tuple[List[Union[int, None]], List[str]]:
        """
        Splits the text based on formatting characters or sequences and returns a list of text segments and formatting delimiters.
        Each formatting delimiter is replaced with its index in the formatting_chars list.

        :param text: The original text with formatting.
        :type text: str

        :return: A tuple containing a list of text segments and indices for formatting delimiters, excluding any empty strings.
        :rtype: Tuple[List[Union[int, None]], List[str]]

        Example::
            >>> format_map, texts = await self.__extract_formatting(text)
        """
        # Initialize the list to store the results
        split_fmt = []

        # Process each line separately
        for line in text.splitlines(keepends=True):
            # Split line while keeping the delimiters, and filter out empty strings
            split_text = [segment for segment in re.split(f'({"|".join(self.regex_chars)})', line, flags=re.MULTILINE) if segment != '']

            for segment in split_text:
                # Try to match the segment against each pattern
                matched_index = None
                for i, fmt in enumerate(self.regex_chars):
                    if re.fullmatch(fmt, segment, flags=re.MULTILINE):
                        matched_index = i
                        break
                if matched_index is not None:
                    # If a match was found, append the index
                    split_fmt.append(matched_index)
                else:
                    # If no match was found, append the original segment
                    split_fmt.append(segment)
        
        format_map = [item if isinstance(item, int) else None for item in split_fmt]
        texts = [item for item in split_fmt if isinstance(item, str)]
        
        return (format_map, texts)

    async def __reapply_formatting(self, translated_text: str, format_map: List[Union[int, str]]) -> list:
        """
        Reapplies formatting characters to the translated text based on their original positions.

        :param translated_text: The translated text without formatting.
        :type translated_text: str
        :param format_map: A list of formatting characters with their positions in the original text.
        :type format_map: List[Union[int, str]]

        :return: The translated text with re-applied formatting.
        :rtype: list

        Example::
            >>> reconst = await self.__reapply_formatting(translated_text, format_map)
        """
        
        translation_iter = iter(translated_text)  # Create an iterator for translations
        for i, value in enumerate(format_map):
            if value is None:
                format_map[i] = next(translation_iter)  # Replace None with the next translation
            
        index_to_markdown = {i: char for i, char in enumerate(self.markdown_chars)}
        reconst = [index_to_markdown[item] if isinstance(item, int) else item for item in format_map]
        return reconst

    async def translate(
        self, 
        text: str, 
        source_lang: str = '', 
        target_lang: str = '', 
        reconstruct_formatting: bool = False
    ) -> str:
        """
        Translates input text from the source language to the target language, preserving and reconstructing text formatting if specified.
        Handles long texts by splitting into manageable batches that don't exceed the model's max sequence length.

        :param text: The input text to be translated.
        :type text: str
        :param source_lang: The ISO 639-1 code of the source language. Uses model's default if not specified.
        :type source_lang: str, optional
        :param target_lang: The ISO 639-1 code of the target language. Uses model's default if not specified.
        :type target_lang: str, optional
        :param reconstruct_formatting: If True, preserves original text formatting in the translated text.
        :type reconstruct_formatting: bool, optional

        :return: The translated text, with formatting reconstructed if specified.
        :rtype: str

        :raises ValueError: If source or target language codes are not supported.

        Example::

            >>> language_translator = LanguageTranslator()
            >>> translated_text = await language_translator.translate("This is an example text.", source_lang='en', target_lang='fr', reconstruct_formatting=True)
            >>> print(translated_text)

        Note:
            - Automatically manages text splitting for long inputs.
            - When reconstruct_formatting is True, identifies and preserves formatting using markdown syntax.
        """
        if source_lang == target_lang:
            return text  # No translation needed
        
        format_map = None
        source_lang_code = self.lang_dict.get(source_lang, source_lang)
        target_lang_code = self.lang_dict.get(target_lang, target_lang)

        if reconstruct_formatting:
            format_map, text = await self.__extract_formatting(text)
            
        batches, format_map = await self.__batch_sentences(text, max_tokens=self.max_sequence_len, format_map=format_map)        
        tasks = [self.__translate_batch(batch, source_lang_code, target_lang_code) for batch in batches]
        translations = await asyncio.gather(*tasks)

        if reconstruct_formatting:
            translations = await self.__reapply_formatting(
                translated_text=translations, 
                format_map=format_map
            )
        return ' '.join(translations)
