"""앱 전역 설정 (Document AI, OpenAI, Gemini).

OpenAI 키는 환경 변수 OPENAI_API_KEY, Gemini 키는 GEMINI_API_KEY 가 있으면 우선한다.
"""
from __future__ import annotations

import os

# Google Document AI (ocr_util)
PROJECT_ID = os.getenv("DETAIL_PAGE_PROJECT_ID", "detail-page-471204")
PROCESSOR_ID = os.getenv("DETAIL_PAGE_PROCESSOR_ID", "33bf5ca98f334262")
LOCATION = os.getenv("DETAIL_PAGE_LOCATION", "us")
OCR_MAX_DIM = int(os.getenv("OCR_MAX_DIM", "10000"))
OCR_PARALLELISM = max(1, int(os.getenv("OCR_PARALLELISM", "6")))

# OpenAI
OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    "",
)
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "gpt-5")
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2")
OPENAI_IMAGE_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "medium")
# 이미지 위 문구 좌표 추출 (멀티모달)
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5.5")

# Gemini (텍스트 → JSON 단계에서 사용)
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    "",
)
#GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3-flash-preview")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview")
