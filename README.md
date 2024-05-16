# Translation

Services:
- Translation

Models:
- LangDetection: (cpu): hosted on cloud run
- LangTranslation: (gpu): hosted on Huggingface Dedicated Endpoints

Orchestration:
GCP Cloud Run (Translation)

Serves RapidAPI endpoints:
    * /Detect: Detect language (cpu)
    * /Translate: calls /Detect with 'auto' and HF dedicated endpoint (gpu)