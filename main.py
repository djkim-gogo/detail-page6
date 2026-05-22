"""
제품 사진 → OCR → GPT JSON → **첫 `image2_prompt_en`으로 images.generate**
→ 비전 모델로 **문구 좌표 JSON** 추출 → 화면 표시.

이미지 편집: POST /api/image-edit (기존).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import tempfile
import time
import uuid
from contextvars import ContextVar
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFilter

from text_mask_refiner import inpaint_with_mask, refine_mask_to_text_pixels

from config import (
    GEMINI_API_KEY,
    GEMINI_IMAGE_MODEL,
    GEMINI_TEXT_MODEL,
    LOCATION,
    OCR_MAX_DIM,
    OPENAI_API_KEY,
    OPENAI_IMAGE_MODEL,
    OPENAI_IMAGE_QUALITY,
    OPENAI_TEXT_MODEL,
    OPENAI_VISION_MODEL,
    PROCESSOR_ID,
    PROJECT_ID,
)
from ocr_util import ocr_image_file

app = FastAPI(title="제품 상세 프롬프트 생성", version="1.0.0")

_BASE_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _BASE_DIR / "static"
_TEMPLATES_DIR = _BASE_DIR / "templates"
_INDEX_HTML_PATH = _TEMPLATES_DIR / "index.html"

if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

_ASSETS = _BASE_DIR / "assets"
_PROMPT_HEAD = _ASSETS / "prompt_head.txt"
_PROMPT_TAIL = _ASSETS / "prompt_tail.txt"


def _load_prompt_parts() -> tuple[str, str]:
    if not _PROMPT_HEAD.is_file() or not _PROMPT_TAIL.is_file():
        raise RuntimeError(
            f"프롬프트 자산이 없습니다: {_PROMPT_HEAD}, {_PROMPT_TAIL}"
        )
    head = _PROMPT_HEAD.read_text(encoding="utf-8").rstrip() + "\n"
    tail = _PROMPT_TAIL.read_text(encoding="utf-8")
    return head, tail


def build_user_message(ocr_text: str, ui_prompt: str) -> str:
    """[입력] 헤더 아래에 OCR + (선택) UI 프롬프트를 붙여 전체 사용자 메시지를 만든다."""
    head, tail = _load_prompt_parts()
    body = "\n## 제품 사진에서 추출한 텍스트 (OCR)\n\n"
    body += (ocr_text.strip() if ocr_text.strip() else "(OCR 결과가 비어 있습니다.)") + "\n"
    if (ui_prompt or "").strip():
        body += "\n## 사용자 추가 프롬프트 (UI 입력)\n\n" + ui_prompt.strip() + "\n"
    body += "\n"
    return head + body + tail


def _openai_api_key() -> str:
    return (os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY or "").strip()


def _parse_gpt_json(raw: str) -> tuple[str, dict[str, Any] | None]:
    """모델 응답에서 JSON 객체 1개를 파싱. 실패 시 (raw, None).

    Gemini/OpenAI 가 본문 앞뒤에 코드펜스, 설명문, 또는 우발적인 추가 토큰
    (예: 빈 줄 + 단독 `}`) 을 붙여 보내도 첫 번째 완결된 객체를 추출한다.
    """
    text = (raw or "").strip()
    if not text:
        return text, None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return text, parsed
    except json.JSONDecodeError:
        pass
    # ```json ... ``` 코드펜스 안쪽 우선
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        inner = fence.group(1).strip()
        parsed = _raw_decode_first_object(inner)
        if isinstance(parsed, dict):
            return text, parsed
    # 본문 어디서든 첫 '{' 부터 raw_decode 로 잘라 시도 (trailing garbage 허용)
    parsed = _raw_decode_first_object(text)
    if isinstance(parsed, dict):
        return text, parsed
    return text, None


def _raw_decode_first_object(text: str) -> Any:
    """text 의 첫 '{' 위치부터 JSONDecoder.raw_decode 로 한 객체만 파싱.

    뒤따르는 추가 문자(빈 줄, 잉여 `}` 등) 는 무시된다. 실패하면 None 반환.
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(text[start:])
        return obj
    except json.JSONDecodeError:
        return None


_GEMINI_CLIENT: genai.Client | None = None


def _gemini_client() -> genai.Client:
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        key = (os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY or "").strip()
        if not key:
            raise RuntimeError("GEMINI_API_KEY 가 config.py 또는 환경 변수에 없습니다.")
        _GEMINI_CLIENT = genai.Client(api_key=key)
    return _GEMINI_CLIENT


GEMINI_JSON_SYSTEM_INSTRUCTION = (
    "You output only one valid JSON object matching the user's schema. "
    "No markdown fences, no natural language before or after the JSON."
)


def _chat_sync(
    user_message: str,
    model: str,
    product_image: bytes | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Gemini 로 (선택적으로 제품 사진과 함께) 텍스트→JSON 응답 생성.

    product_image 가 주어지면 멀티모달 입력으로 함께 전달한다.
    """
    system_instruction = GEMINI_JSON_SYSTEM_INSTRUCTION
    client = _gemini_client()

    contents: list[Any] = [user_message]
    if product_image:
        try:
            img = Image.open(BytesIO(product_image)).convert("RGB")
            contents.append(img)
        except Exception:
            # PIL 로드 실패 시 inline_data Part 로 폴백
            contents.append(
                genai_types.Part.from_bytes(data=product_image, mime_type="image/png")
            )

    cfg_with_json = genai_types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
    )
    try:
        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=cfg_with_json,
        )
    except Exception:
        # 일부 모델/버전에서 response_mime_type 미지원 → 폴백
        cfg = genai_types.GenerateContentConfig(system_instruction=system_instruction)
        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=cfg,
        )
    raw = (getattr(resp, "text", None) or "").strip()
    return _parse_gpt_json(raw)


def choose_api_size(width: int, height: int) -> str:
    ratio = width / height
    if 0.9 <= ratio <= 1.1:
        return "1024x1024"
    if ratio < 1.0:
        return "1024x1536"
    return "1536x1024"


_FIRST_IMAGE_LAYOUT_KEYS: tuple[str, ...] = (
    "index",
    "section_type",
    "width_px",
    "height_px",
    "main_copy",
    "sub_copy",
    "bullet_points",
    "trust_or_proof",
    "layout_notes",
    "typography_notes",
    "image2_prompt_en",
)


def slice_first_block_for_layout(block: dict[str, Any]) -> dict[str, Any]:
    """section_type ~ image2_prompt_en 사이(양끝 포함)의 필드를 JSON 객체 순서대로 유지."""
    keys = list(block.keys())
    if "section_type" in keys and "image2_prompt_en" in keys:
        i0 = keys.index("section_type")
        i1 = keys.index("image2_prompt_en")
        if i0 <= i1:
            return {k: block[k] for k in keys[i0 : i1 + 1]}
    return {k: block[k] for k in _FIRST_IMAGE_LAYOUT_KEYS if k in block}

IMAGE_TEXT_CLEAN_PROMPT = (
"둘째 이미지는 제품 박스야. \n"
"첫째 이미지에서 둘째 이미지 제품 영역 외 글자를 지워.\n"
"첫째 이미지에서 절대 제품 박스 안에 있는 글자는 지우지 마.\n"
"아이콘, 로고 등 이미지는 절대 지우면 안 돼.\n"
"도형이나 이미지 안의 글자는 지우면 안 돼.\n"
"이미지에서 OCR 로 추출한 텍스트 영역만 지울지 검토할 영역이고 나머지 영역은 절대 수정하지 마.\n"
)

# 4단계 (image_gen2) 에서 선택 가능한 이미지 모델 + 품질 조합 (첫 항목이 기본값)
IMAGE_GEN2_OPTIONS: list[dict[str, str]] = [
    {"model": OPENAI_IMAGE_MODEL, "quality": "low",    "label": f"{OPENAI_IMAGE_MODEL} (low)"},
    {"model": OPENAI_IMAGE_MODEL, "quality": "medium", "label": f"{OPENAI_IMAGE_MODEL} (medium)"},
    {"model": OPENAI_IMAGE_MODEL, "quality": "high",   "label": f"{OPENAI_IMAGE_MODEL} (high)"},
    {"model": GEMINI_IMAGE_MODEL, "quality": "",       "label": GEMINI_IMAGE_MODEL},
]

# 6단계 (image_gen_text) 에서 선택 가능한 이미지 모델 + 품질 조합 (첫 항목이 기본값)
IMAGE_GEN_TEXT_OPTIONS: list[dict[str, str]] = [
    {"model": OPENAI_IMAGE_MODEL, "quality": "low",    "label": f"{OPENAI_IMAGE_MODEL} (low)"},
    {"model": OPENAI_IMAGE_MODEL, "quality": "medium", "label": f"{OPENAI_IMAGE_MODEL} (medium)"},
    {"model": OPENAI_IMAGE_MODEL, "quality": "high",   "label": f"{OPENAI_IMAGE_MODEL} (high)"},
    {"model": GEMINI_IMAGE_MODEL, "quality": "",       "label": GEMINI_IMAGE_MODEL},
]

# 7단계 (overlay_refine) 모델 선택지. gpt-5 의 reasoning_effort 단계별 옵션.
_OVERLAY_REFINE_MODEL = OPENAI_VISION_MODEL
OVERLAY_REFINE_OPTIONS: list[dict[str, str]] = [
    {"model": _OVERLAY_REFINE_MODEL, "reasoning_effort": "minimal",
     "label": f"{_OVERLAY_REFINE_MODEL} (reasoning: minimal — 가장 빠름)"},
    {"model": _OVERLAY_REFINE_MODEL, "reasoning_effort": "low",
     "label": f"{_OVERLAY_REFINE_MODEL} (reasoning: low)"},
    {"model": _OVERLAY_REFINE_MODEL, "reasoning_effort": "medium",
     "label": f"{_OVERLAY_REFINE_MODEL} (reasoning: medium — 기본)"},
    {"model": _OVERLAY_REFINE_MODEL, "reasoning_effort": "high",
     "label": f"{_OVERLAY_REFINE_MODEL} (reasoning: high — 정밀, 느림)"},
]
#OVERLAY_REFINE_DEFAULT_EFFORT = "medium"
OVERLAY_REFINE_DEFAULT_EFFORT = "high"


def build_image_text_prompt(first: dict[str, Any]) -> str:
    """3단계 결과 이미지 위에 텍스트를 합성하기 위한 프롬프트 (현재 미사용 — 참조용)."""
    main_copy = (first.get("main_copy") or "").strip()
    sub_copy = (first.get("sub_copy") or "").strip()
    bullets = first.get("bullet_points") or []
    if isinstance(bullets, list):
        bullets_text = "\n".join(f"- {bp}" for bp in bullets if str(bp).strip())
    else:
        bullets_text = str(bullets).strip()
    trust_or_proof = first.get("trust_or_proof") or ""
    if isinstance(trust_or_proof, list):
        trust_text = "\n".join(f"- {t}" for t in trust_or_proof if str(t).strip())
    else:
        trust_text = str(trust_or_proof).strip()
    typography_notes = (first.get("typography_notes") or "").strip()

    parts: list[str] = [
        "Take the attached image and render the following Korean marketing copy directly onto it as clean, well-styled typography.\n",
        "- Preserve the existing image aesthetic, color palette, and composition.\n",
        "- Place text in visually appropriate regions (do not cover key product imagery).\n",
        "- Use clear hierarchy: main copy largest, sub copy smaller, bullets even smaller.\n",
        "- Korean typography only, sharp and readable.\n",
    ]
    if main_copy:
        parts.append(f"\n[Main copy]\n{main_copy}\n")
    if sub_copy:
        parts.append(f"\n[Sub copy]\n{sub_copy}\n")
    if bullets_text:
        parts.append(f"\n[Bullet points]\n{bullets_text}\n")
    if trust_text:
        parts.append(f"\n[Trust or proof]\n{trust_text}\n")
    if typography_notes:
        parts.append(f"\n[Typography notes]\n{typography_notes}\n")
    return "".join(parts)


def _ocr_with_positions_sync(image_bytes: bytes) -> dict[str, Any]:
    """이미지를 Document AI 로 OCR 하고 라인 단위 텍스트 + 픽셀 좌표를 반환.

    좌표는 입력 image_bytes 원본 크기 기준으로 환산.
    """
    from google.cloud import documentai_v1
    from ocr_util import _encode_for_ocr, _get_client, _resize_for_ocr

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = img.size
    img_ocr = _resize_for_ocr(img, OCR_MAX_DIM)
    ocr_w, ocr_h = img_ocr.size
    enc_bytes, mime = _encode_for_ocr(img_ocr)

    client = _get_client()
    processor_name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)
    raw_document = documentai_v1.RawDocument(content=enc_bytes, mime_type=mime)
    request = documentai_v1.ProcessRequest(name=processor_name, raw_document=raw_document)
    result = client.process_document(request=request)
    doc = result.document
    full_text = doc.text or ""
    sx = orig_w / max(1, ocr_w)
    sy = orig_h / max(1, ocr_h)

    lines_out: list[dict[str, Any]] = []
    for page in doc.pages:
        page_w = float(getattr(page.dimension, "width", 0) or ocr_w)
        page_h = float(getattr(page.dimension, "height", 0) or ocr_h)
        for line in (page.lines or []):
            segs: list[str] = []
            for seg in line.layout.text_anchor.text_segments:
                s = int(seg.start_index) if getattr(seg, "start_index", None) else 0
                e = int(seg.end_index) if getattr(seg, "end_index", None) else 0
                if e > s:
                    segs.append(full_text[s:e])
            text = "".join(segs).strip()
            if not text:
                continue
            poly = line.layout.bounding_poly
            xs: list[float] = []
            ys: list[float] = []
            if poly.vertices:
                xs = [float(v.x) for v in poly.vertices]
                ys = [float(v.y) for v in poly.vertices]
            elif poly.normalized_vertices:
                xs = [float(v.x) * page_w for v in poly.normalized_vertices]
                ys = [float(v.y) * page_h for v in poly.normalized_vertices]
            if not xs or not ys:
                continue
            x0, y0 = min(xs) * sx, min(ys) * sy
            x1, y1 = max(xs) * sx, max(ys) * sy
            w_box = max(0.0, x1 - x0)
            h_box = max(0.0, y1 - y0)
            lines_out.append({
                "text": text,
                "x": int(round(x0)),
                "y": int(round(y0)),
                "width": int(round(w_box)),
                "height": int(round(h_box)),
                "font_size": int(round(h_box * 0.85)),
            })
    return {
        "lines": lines_out,
        "full_text": full_text.strip(),
        "image_width": orig_w,
        "image_height": orig_h,
    }


def _text_area_diff_sync(
    lines_a: list[dict[str, Any]],
    lines_b: list[dict[str, Any]],
    iou_threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """lines_a 의 각 라인이 lines_b 의 라인들과 충분히 겹치지 않으면 (IoU < threshold) 반환.

    "a 에는 있고 b 에는 없는" 영역을 찾는다. 좌표는 dict 의 x/y/width/height 키 사용.
    """
    def _xyxy(ln: dict[str, Any]) -> tuple[float, float, float, float, float]:
        x = float(ln.get("x") or 0)
        y = float(ln.get("y") or 0)
        w = float(ln.get("width") or 0)
        h = float(ln.get("height") or 0)
        return x, y, x + w, y + h, max(0.0, w * h)

    result: list[dict[str, Any]] = []
    for a in (lines_a or []):
        if not isinstance(a, dict):
            continue
        x1a, y1a, x2a, y2a, area_a = _xyxy(a)
        if area_a <= 0:
            continue
        max_iou = 0.0
        for b in (lines_b or []):
            if not isinstance(b, dict):
                continue
            x1b, y1b, x2b, y2b, area_b = _xyxy(b)
            if area_b <= 0:
                continue
            ix1 = max(x1a, x1b); iy1 = max(y1a, y1b)
            ix2 = min(x2a, x2b); iy2 = min(y2a, y2b)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = area_a + area_b - inter
            iou = inter / union if union > 0 else 0.0
            if iou > max_iou:
                max_iou = iou
        if max_iou < iou_threshold:
            result.append(a)
    return result


def _refine_diff_sync(b64: str, threshold: int = 60) -> dict[str, Any]:
    """diff 이미지에서 임계값 미만의 흐린 픽셀을 검정으로 만들어 진한 차이만 남김.

    threshold (0~255): 각 픽셀의 최대 채널 강도가 이 값 미만이면 0으로.
    """
    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    kept = -1
    total = img.width * img.height
    try:
        import numpy as np

        arr = np.array(img)
        max_intensity = arr.max(axis=-1)
        mask = max_intensity >= int(threshold)
        out_arr = np.zeros_like(arr)
        out_arr[mask] = arr[mask]
        out_img = Image.fromarray(out_arr)
        kept = int(mask.sum())
    except Exception:
        # numpy 미사용 폴백: PIL 픽셀 루프 (느릴 수 있음)
        out_img = img.copy()
        px = out_img.load()
        kept = 0
        for y in range(out_img.height):
            for x in range(out_img.width):
                r, g, b = px[x, y]
                if max(r, g, b) < int(threshold):
                    px[x, y] = (0, 0, 0)
                else:
                    kept += 1
    bbox = out_img.getbbox()
    bio = BytesIO()
    out_img.save(bio, format="PNG")
    return {
        "b64_json": base64.b64encode(bio.getvalue()).decode("ascii"),
        "output_width": out_img.width,
        "output_height": out_img.height,
        "threshold": int(threshold),
        "kept_pixels": kept,
        "total_pixels": total,
        "bbox": list(bbox) if bbox else None,
    }


def _diff_images_sync(b64_a: str, b64_b: str) -> dict[str, Any]:
    """두 이미지를 픽셀 단위로 비교해 차이 이미지를 PNG b64 로 반환."""
    from PIL import ImageChops

    img_a = Image.open(BytesIO(base64.b64decode(b64_a))).convert("RGB")
    img_b = Image.open(BytesIO(base64.b64decode(b64_b))).convert("RGB")
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size)
    diff = ImageChops.difference(img_a, img_b)
    bbox = diff.getbbox()
    diff_pixels = 0
    try:
        import numpy as np  # 선택 의존성

        arr = np.array(diff)
        diff_pixels = int((arr.sum(axis=-1) > 0).sum())
    except Exception:
        diff_pixels = -1
    bio = BytesIO()
    diff.save(bio, format="PNG")
    return {
        "b64_json": base64.b64encode(bio.getvalue()).decode("ascii"),
        "output_width": diff.width,
        "output_height": diff.height,
        "diff_pixels": diff_pixels,
        "bbox": list(bbox) if bbox else None,
    }


def _gemini_image_edit_sync(
    prompt: str,
    input_image: bytes,
    model: str,
    extra_images: list[tuple[bytes, str]] | None = None,
) -> dict[str, Any]:
    """Gemini 멀티모달 모델로 입력 이미지 + 프롬프트 → 새 이미지 생성.

    detail-page4 와 동일한 패턴(client.models.generate_content)을 사용하되,
    응답에서 inline_data 이미지 파트를 추출. 이미지 출력 가능한 모델
    (예: gemini-2.5-flash-image-preview, gemini-2.5-flash-image)을 사용해야 함.
    extra_images 가 있으면 함께 contents 에 동봉.
    """
    client = _gemini_client()
    img = Image.open(BytesIO(input_image)).convert("RGB")
    extra_pils: list[Image.Image] = []
    for (b, _n) in (extra_images or []):
        if not b:
            continue
        try:
            extra_pils.append(Image.open(BytesIO(b)).convert("RGB"))
        except Exception:
            pass

    # 1) detail-page4 와 같은 단순 호출
    last_err: Exception | None = None
    resp = None
    contents_list: list[Any] = [prompt, img] + extra_pils
    try:
        resp = client.models.generate_content(model=model, contents=contents_list)
    except Exception as e:
        last_err = e

    # 2) IMAGE 모달리티를 명시한 호출 폴백
    if resp is None:
        try:
            cfg = genai_types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])
            resp = client.models.generate_content(
                model=model, contents=contents_list, config=cfg,
            )
        except Exception as e:
            raise RuntimeError(
                f"Gemini generate_content 호출 실패 (model={model}): {last_err or e}"
            ) from e

    candidates = getattr(resp, "candidates", None) or []
    text_summary = ""
    mime_seen: list[str] = []
    for cand in candidates:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in (getattr(content, "parts", None) or []):
            ptext = getattr(part, "text", None)
            if ptext:
                text_summary += str(ptext)
            inline = getattr(part, "inline_data", None)
            if not inline:
                # 신버전 SDK 에서 file_data / inlineData 등으로 노출될 수도 있음
                inline = getattr(part, "inlineData", None)
            if not inline:
                continue
            data = getattr(inline, "data", None)
            mime = getattr(inline, "mime_type", None) or getattr(inline, "mimeType", None) or "?"
            mime_seen.append(str(mime))
            if not data:
                continue
            if isinstance(data, str):
                try:
                    data = base64.b64decode(data)
                except Exception:
                    continue
            try:
                with Image.open(BytesIO(data)) as gen:
                    gw, gh = gen.size
            except Exception:
                continue
            return {
                "b64_json": base64.b64encode(data).decode("ascii"),
                "output_width": gw,
                "output_height": gh,
            }

    # 직접 resp.text 가 있으면 미리 보여줌
    if not text_summary:
        text_summary = (getattr(resp, "text", None) or "")
    snippet = (text_summary or "")[:240]
    raise RuntimeError(
        "Gemini 응답에 이미지 파트가 없습니다. "
        f"model={model}, candidates={len(candidates)}, parts_mime={mime_seen or '없음'}, "
        f"text='{snippet}'. 이미지 출력을 지원하는 모델을 사용하세요 "
        "(예: gemini-2.5-flash-image-preview, gemini-2.5-flash-image)."
    )


def _build_text_mask_png(
    width: int, height: int, rects: list[dict[str, Any]], pad: int = 4
) -> bytes:
    """텍스트 영역만 투명(alpha=0), 나머지는 불투명(alpha=255) 인 RGBA PNG 마스크.

    OpenAI images.edit 의 mask 규약: 투명한 영역만 모델이 재생성/편집한다.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"마스크 크기가 잘못됨: {width}x{height}")
    mask = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    draw = ImageDraw.Draw(mask)
    for r in rects or []:
        if not isinstance(r, dict):
            continue
        try:
            x = int(r.get("x") or 0)
            y = int(r.get("y") or 0)
            w = int(r.get("width") or 0)
            h = int(r.get("height") or 0)
        except (ValueError, TypeError):
            continue
        if w <= 0 or h <= 0:
            continue
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(width, x + w + pad)
        y1 = min(height, y + h + pad)
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 0))
    bio = BytesIO()
    mask.save(bio, format="PNG")
    return bio.getvalue()


def _composite_mask_regions(
    base_png: bytes, edited_png: bytes,
    rects: list[dict[str, Any]], pad: int = 4,
    mask_png_override: bytes | None = None,
) -> bytes:
    """edited_png 의 마스크 영역만 base_png 위에 덮어쓴 PNG 반환.

    mask_png_override 가 주어지면 그 마스크의 alpha=0 영역을 합성 영역으로 쓴다
    (API 에 실제 보낸 마스크와 합성을 일치). 없으면 rects+pad 로 박스를 만든다.
    """
    base = Image.open(BytesIO(base_png)).convert("RGBA")
    edited = Image.open(BytesIO(edited_png)).convert("RGBA")
    if edited.size != base.size:
        edited = edited.resize(base.size, Image.LANCZOS)
    if mask_png_override:
        mp = Image.open(BytesIO(mask_png_override)).convert("RGBA")
        if mp.size != base.size:
            mp = mp.resize(base.size, Image.NEAREST)
        # mask 의 alpha=0 (재생성 영역) → composite mask=255 (edited 사용)
        import numpy as _np
        alpha = _np.array(mp.split()[-1])
        comp_arr = (255 - alpha).astype("uint8")
        region_mask = Image.fromarray(comp_arr, mode="L")
    else:
        region_mask = Image.new("L", base.size, 0)
        draw = ImageDraw.Draw(region_mask)
        bw, bh = base.size
        for r in rects or []:
            if not isinstance(r, dict):
                continue
            try:
                x = int(r.get("x") or 0)
                y = int(r.get("y") or 0)
                w = int(r.get("width") or 0)
                h = int(r.get("height") or 0)
            except (ValueError, TypeError):
                continue
            if w <= 0 or h <= 0:
                continue
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(bw, x + w + pad)
            y1 = min(bh, y + h + pad)
            if x1 <= x0 or y1 <= y0:
                continue
            draw.rectangle([x0, y0, x1, y1], fill=255)
    out = base.copy()
    out.paste(edited, (0, 0), region_mask)
    bio = BytesIO()
    out.convert("RGB").save(bio, format="PNG")
    return bio.getvalue()


def _expand_rects_with_margin(
    rects: list[dict[str, Any]], margin: int, img_width: int | None = None, img_height: int | None = None
) -> list[dict[str, Any]]:
    """각 rect 를 margin 픽셀씩 확장."""
    expanded = []
    for r in rects:
        if not isinstance(r, dict):
            expanded.append(r)
            continue
        try:
            x = int(r.get("x", 0))
            y = int(r.get("y", 0))
            w = int(r.get("width", 0))
            h = int(r.get("height", 0))
        except (ValueError, TypeError):
            expanded.append(r)
            continue

        x_new = max(0, x - margin)
        y_new = max(0, y - margin)
        x_end = x + w + margin
        y_end = y + h + margin

        # 이미지 크기 제한 (선택적)
        if img_width is not None:
            x_end = min(img_width, x_end)
        if img_height is not None:
            y_end = min(img_height, y_end)

        w_new = x_end - x_new
        h_new = y_end - y_new

        # 확장 후 크기가 유효한 경우만 포함
        if w_new > 0 and h_new > 0:
            expanded_r = {**r, "x": x_new, "y": y_new, "width": w_new, "height": h_new}
            expanded.append(expanded_r)
    return expanded


def _mask_alpha_bbox(mask_png: bytes) -> tuple[int, int, int, int] | None:
    """RGBA 마스크의 alpha<255 영역(재생성 대상) 의 (x, y, w, h) bbox 반환.

    투명 영역이 없으면 None.
    """
    img = Image.open(BytesIO(mask_png)).convert("RGBA")
    alpha = img.split()[-1]
    # alpha<255 → 255, 그 외 → 0 (재생성 영역을 흰색으로 표현)
    inv = alpha.point(lambda v: 0 if v == 255 else 255)
    bbox = inv.getbbox()
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def _crop_png(png: bytes, bbox: tuple[int, int, int, int]) -> bytes:
    x, y, w, h = bbox
    img = Image.open(BytesIO(png))
    crop = img.crop((x, y, x + w, y + h))
    bio = BytesIO()
    crop.save(bio, format="PNG")
    return bio.getvalue()


def _paste_resized(
    base_png: bytes, patch_png: bytes, bbox: tuple[int, int, int, int]
) -> bytes:
    """patch_png 를 bbox 크기로 리사이즈해 base_png 의 (bbox.x, bbox.y) 에 붙여 PNG 반환."""
    x, y, w, h = bbox
    base = Image.open(BytesIO(base_png)).convert("RGB")
    patch = Image.open(BytesIO(patch_png)).convert("RGB")
    if patch.size != (w, h):
        patch = patch.resize((w, h), Image.LANCZOS)
    base.paste(patch, (x, y))
    bio = BytesIO()
    base.save(bio, format="PNG")
    return bio.getvalue()


def _apply_alpha_outside_mask(rgb_png: bytes, mask_png: bytes, margin: int = 20) -> bytes:
    """rgb_png 에 마스크의 각 영역(alpha=0) 을 margin 픽셀씩 확장한 영역만 원본 유지, 그 외는 흰색.

    margin: 각 마스크 영역을 margin 픽셀씩 dilate (기본 20px).
    """
    rgb = Image.open(BytesIO(rgb_png)).convert("RGBA")
    mask = Image.open(BytesIO(mask_png)).convert("RGBA")
    if mask.size != rgb.size:
        mask = mask.resize(rgb.size, Image.NEAREST)

    mask_alpha = mask.split()[-1]
    # mask alpha=0 (마스크 영역) 을 margin 픽셀씩 dilate
    # MinFilter: 각 픽셀을 주변 최소값으로 대체 → alpha=0 영역(=마스크 영역) 이 확장됨
    if margin > 0:
        size = margin * 2 + 1
        mask_alpha = mask_alpha.filter(ImageFilter.MinFilter(size))
    # 확장된 마스크 영역 → inside=255 (원본 유지), 그 외 → inside=0 (흰색 덮어씀)
    inside_mask = mask_alpha.point(lambda v: 255 if v == 0 else 0)

    # 마스크 영역 외 (inside_arr == 0) 를 흰색으로 덮어씀
    rgba_arr = np.array(rgb)
    inside_arr = np.asarray(inside_mask)
    rgba_arr[inside_arr == 0, 0] = 255
    rgba_arr[inside_arr == 0, 1] = 255
    rgba_arr[inside_arr == 0, 2] = 255
    rgba_arr[inside_arr == 0, 3] = 255

    rgba = Image.fromarray(rgba_arr, mode="RGBA")
    # RGB 로 저장하여 alpha 채널 제거
    rgb_final = rgba.convert("RGB")
    bio = BytesIO()
    rgb_final.save(bio, format="PNG")
    return bio.getvalue()


def _crop_paste_api_call(
    api_key: str, prompt: str,
    api_input_bytes: bytes, mask_png_bytes: bytes,
    model: str, quality: str,
) -> dict[str, Any]:
    """mask 영역 외 부분은 투명하게 만들어 API 호출 → 응답의 mask 영역만 원본에 붙여 반환.

    반환 dict 키:
      - bbox: (x, y, w, h) | None  (None 이면 마스크에 투명 영역 없음 → 폴백)
      - cropped_input_bytes: bytes | None  (마스크 외 투명 처리된 입력 — API 에 보낸 이미지)
      - cropped_mask_bytes: bytes | None   (= mask_png_bytes)
      - api_response_bytes: bytes | None   (API 가 반환한 원시 이미지)
      - final_bytes: bytes                 (최종 합성 이미지 — 항상 존재)
      - final_width / final_height: int
    """
    bbox = _mask_alpha_bbox(mask_png_bytes) if mask_png_bytes else None
    if bbox is None:
        # 폴백: 전체 이미지 + 마스크 없음으로 호출
        out = _images_generate_sync(
            api_key, prompt, choose_api_size(*Image.open(BytesIO(api_input_bytes)).size),
            api_input_bytes, "step3_inpainted.png",
            model, quality, None, mask_png_bytes,
        )
        raw = base64.b64decode(out["b64_json"])
        with Image.open(BytesIO(raw)) as g:
            gw, gh = g.size
        return {
            "bbox": None,
            "cropped_input_bytes": None,
            "cropped_mask_bytes": None,
            "api_response_bytes": raw,
            "final_bytes": raw,
            "final_width": gw, "final_height": gh,
        }

    in_w, in_h = Image.open(BytesIO(api_input_bytes)).size
    # mask 를 api_input 과 같은 크기로 crop/paste (resize 하지 않음)
    mask_img = Image.open(BytesIO(mask_png_bytes)).convert("RGBA")
    if mask_img.size != (in_w, in_h):
        canvas = Image.new("RGBA", (in_w, in_h), (0, 0, 0, 0))
        canvas.paste(mask_img, (0, 0))
        _bio = BytesIO()
        canvas.save(_bio, format="PNG")
        mask_full = _bio.getvalue()
    else:
        mask_full = mask_png_bytes

    # API 호출: 1번 원본 이미지 + 2번 마스크를 전달
    size = choose_api_size(in_w, in_h)
    out = _images_generate_sync(
        api_key, prompt, size,
        api_input_bytes, "step3_inpainted.png",
        model, quality, None, mask_full,
    )
    raw = base64.b64decode(out["b64_json"])

    # mask 영역만 원본 api_input_bytes 에 합성 (픽셀 단위)
    final_bytes = _composite_mask_regions(
        api_input_bytes, raw, [], 4, mask_full,
    )
    with Image.open(BytesIO(final_bytes)) as g:
        fw, fh = g.size
    return {
        "bbox": bbox,
        "cropped_input_bytes": api_input_bytes,
        "cropped_mask_bytes": mask_full,
        "api_response_bytes": raw,
        "final_bytes": final_bytes,
        "final_width": fw, "final_height": fh,
    }


def _inpaint_and_save(
    base_image_bytes: bytes, refined_mask_png: bytes,
    label: str, page: int | None = None,
) -> tuple[bytes, str | None]:
    """refined mask 의 alpha=0 영역을 주변 색으로 메꾼 PNG 를 만들고 세션 디렉토리에 저장.

    반환: (inpainted_png_bytes, relative_file_path)
    실패 시 (base_image_bytes, None) 으로 폴백.
    """
    try:
        inpainted = inpaint_with_mask(base_image_bytes, refined_mask_png)
    except Exception as ex:
        print(f"[{label}] 인페인트 실패, 원본 4단계 이미지 그대로 사용: {ex}")
        return base_image_bytes, None
    try:
        sdir = _session_dir_var.get()
        if sdir is None:
            sdir = _LOG_ROOT / "ad_hoc"
            sdir.mkdir(parents=True, exist_ok=True)
            page_dir = sdir
        else:
            page_label = "common" if page is None else f"page_{page}"
            page_dir = sdir / page_label
            page_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%H%M%S") + "_" + uuid.uuid4().hex[:4]
        fname = f"{_safe_name(label)}__inpainted_{stamp}.png"
        out_path = page_dir / fname
        out_path.write_bytes(inpainted)
        try:
            rel = str(out_path.relative_to(_LOG_ROOT))
        except ValueError:
            rel = str(out_path)
        return inpainted, rel
    except Exception as ex:
        print(f"[{label}] 인페인트 결과 저장 실패: {ex}")
        return inpainted, None


def _refine_mask_with_logging(
    coarse_png: bytes, base_image_bytes: bytes,
    label: str, page: int | None = None,
) -> tuple[bytes, str | None, dict[str, Any]]:
    """거친 마스크를 글자 픽셀만 남긴 세부 마스크로 정교화하고 파일로 저장.

    반환: (refined_png_bytes, refined_file_relpath, info_dict)
    실패 시 (coarse_png, None, {"error": ...}) 로 폴백.
    """
    try:
        refined_png, info = refine_mask_to_text_pixels(base_image_bytes, coarse_png)
    except Exception as ex:
        print(f"[{label}] 마스크 정교화 실패, 거친 마스크 그대로 사용: {ex}")
        return coarse_png, None, {"error": str(ex)}
    refined_path = _save_mask_to_session(
        refined_png, label=f"{label}_refined", page=page,
    )
    return refined_png, refined_path, info


def _save_mask_to_session(
    mask_bytes: bytes, label: str = "image_gen_text",
    page: int | None = None,
) -> str | None:
    """현재 요청의 세션 로그 디렉토리에 마스크 PNG 를 저장하고 상대 경로 반환.

    세션 디렉토리가 없으면 logs/ad_hoc/ 에 timestamp 파일명으로 저장.
    저장 실패 시 None 반환.
    """
    if not mask_bytes:
        return None
    try:
        sdir = _session_dir_var.get()
        if sdir is None:
            sdir = _LOG_ROOT / "ad_hoc"
            sdir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
            page_dir = sdir
            fname = f"{_safe_name(label)}__mask__{stamp}.png"
        else:
            page_label = "common" if page is None else f"page_{page}"
            page_dir = sdir / page_label
            page_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%H%M%S") + "_" + uuid.uuid4().hex[:4]
            fname = f"{_safe_name(label)}__mask_{stamp}.png"
        out_path = page_dir / fname
        out_path.write_bytes(mask_bytes)
        try:
            return str(out_path.relative_to(_LOG_ROOT))
        except ValueError:
            return str(out_path)
    except Exception as ex:
        print(f"[mask save] 실패: {ex}")
        return None


def _images_generate_sync(
    api_key: str,
    prompt: str,
    size: str,
    product_image: bytes | None = None,
    product_filename: str | None = None,
    model: str | None = None,
    quality: str | None = None,
    extra_images: list[tuple[bytes, str]] | None = None,
    mask: bytes | None = None,
) -> dict[str, Any]:
    """첫 image2_prompt_en 으로 이미지 생성.

    product_image 가 주어지면 업로드한 제품 사진을 입력으로 사용해
    images.edit (이미지 + 프롬프트) 로 생성한다. 없으면 텍스트→이미지로 폴백.
    extra_images: [(bytes, filename), ...] — 추가 입력 이미지 (예: 원본 제품 사진).
    model 미지정 시 config 의 OPENAI_IMAGE_MODEL 사용.
    "gemini" 로 시작하는 모델명이면 Gemini API 로 라우팅 (quality 무시).
    """
    use_model = (model or OPENAI_IMAGE_MODEL).strip() or OPENAI_IMAGE_MODEL
    use_quality = (quality or OPENAI_IMAGE_QUALITY).strip() or OPENAI_IMAGE_QUALITY
    if use_model.lower().startswith("gemini"):
        if not product_image:
            raise RuntimeError("Gemini 이미지 생성에는 입력 이미지가 필요합니다.")
        return _gemini_image_edit_sync(prompt, product_image, use_model, extra_images)
    client = OpenAI(api_key=api_key)
    out_bytes: bytes
    b64_data: str | None = None

    def _bio_for(b: bytes, name: str) -> BytesIO:
        n = name or "image.png"
        ext = Path(n).suffix.lower()
        if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
            n = "image.png"
        bio_ = BytesIO(b)
        bio_.name = n
        return bio_

    if product_image:
        primary_bio = _bio_for(product_image, product_filename or "product.png")
        if extra_images:
            image_param: Any = [primary_bio] + [
                _bio_for(b, n) for (b, n) in extra_images if b
            ]
        else:
            image_param = primary_bio
        edit_kwargs: dict[str, Any] = {
            "model": use_model,
            "image": image_param,
            "prompt": prompt,
            "size": size,
            "quality": use_quality,
        }
        if mask:
            mask_bio = BytesIO(mask)
            mask_bio.name = "mask.png"
            edit_kwargs["mask"] = mask_bio
        try:
            result = client.images.edit(**edit_kwargs, output_format="png")
        except TypeError:
            result = client.images.edit(**edit_kwargs)
        if not result.data:
            raise RuntimeError("images.edit: 데이터 없음")
        d0 = result.data[0]
        b64_data = getattr(d0, "b64_json", None) or None
        if b64_data:
            out_bytes = base64.b64decode(b64_data)
        elif getattr(d0, "url", None):
            import urllib.request

            with urllib.request.urlopen(d0.url) as resp:
                out_bytes = resp.read()
            b64_data = base64.b64encode(out_bytes).decode("ascii")
        else:
            raise RuntimeError("images.edit: base64/url 없음")
    else:
        kwargs: dict[str, Any] = {
            "model": use_model,
            "prompt": prompt,
            "size": size,
            "quality": use_quality,
        }
        try:
            result = client.images.generate(**kwargs, output_format="png")
        except TypeError:
            result = client.images.generate(**kwargs)
        if not result.data:
            raise RuntimeError("images.generate: 데이터 없음")
        d0 = result.data[0]
        b64_data = getattr(d0, "b64_json", None) or None
        if b64_data:
            out_bytes = base64.b64decode(b64_data)
        elif getattr(d0, "url", None):
            import urllib.request

            with urllib.request.urlopen(d0.url) as resp:
                out_bytes = resp.read()
            b64_data = base64.b64encode(out_bytes).decode("ascii")
        else:
            raise RuntimeError("images.generate: base64/url 없음")

    with Image.open(BytesIO(out_bytes)) as gen:
        gw, gh = gen.size
    return {"b64_json": b64_data, "output_width": gw, "output_height": gh}


def build_vision_layout_prompt(meta_block: dict[str, Any]) -> str:
    """비전 모델로 보낼 문구 좌표 추출 프롬프트(텍스트)를 생성한다."""
    meta_json = json.dumps(meta_block, ensure_ascii=False, indent=2)
    return f"""첨부 PNG는 상세페이지용으로 방금 생성된 이미지입니다.

아래 JSON은 **첫 번째** image_prompt 항목에서, section_type 부터 image2_prompt_en 까지(그 사이에 존재하는 필드 전부)를 담은 것입니다.

{meta_json}

작업:
1) 위 내용 중에 이미지에 문구를 넣을 좌표와 문구 리스트를 뽑아
2) 응답은 **유효한 JSON 한 개만** 출력합니다. 마크다운·설명 문·코드펜스 금지.

출력 JSON 스키마(키 이름 유지):
[
  {{
    "text": "매일 챙겨 먹는데,\\n왜 몸은 피곤할까?",
    "x": 80,
    "y": 115,
    "width": 700,
    "height": 140,
    "align": "left",
    "font_style": "고딕체",
    "font_size": 50,
    "font_color": "#333333",
    "style": "bold"
  }}
]
"""


VISION_LAYOUT_SYSTEM_INSTRUCTION = (
    "You output only one valid JSON object. "
    "No markdown fences, no natural language before or after the JSON."
)


def _vision_layout_sync(
    api_key: str,
    vision_model: str,
    image_b64_png: str,
    user_text: str,
) -> tuple[str, dict[str, Any] | None]:
    """생성 이미지 + 미리 만든 프롬프트로 문구 좌표 JSON 추출."""
    data_uri = f"data:image/png;base64,{image_b64_png}"
    client = OpenAI(api_key=api_key)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": VISION_LAYOUT_SYSTEM_INSTRUCTION,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                },
            ],
        },
    ]
    try:
        resp = client.chat.completions.create(
            model=vision_model,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(
            model=vision_model,
            messages=messages,
        )
    raw = (resp.choices[0].message.content or "").strip()
    return _parse_gpt_json(raw)


STEP3_TEXT_LAYOUT_PROMPT = (
    "아래 내용으로 상세페이지용 이미지 만들거야. "
    "이미지에 들어갈 문구의 내용, 위치, 크기, 색 등 문구에 대한 정보를 json 으로 만들어"
)

STEP3_5_IMAGE_GEN_PROMPT = (
    "상세페이지 이미지 만들어\n"
    "문구는 아래 문구만 그대로 사용해\n"
    "문구를 수정하면 안 돼"
)

"""
STEP6_TEXT_REMOVE_PROMPT = (
    "첨부한 마스크 영역을 지운 이미지 생성해.\n"
    "영역내 글자만 지워.\n"
    "지운 부분은 배경을 자연스럽게 복원해.\n"
    "상자, 테두리, 선, 아이콘 등 디자인 요소를 추가하지 마.\n"
    "영역 이외는 절대 수정하지 마.\n"
    "영역내에서도 도형, 로고, 이미지는 지우면 안 돼\n"
)
"""

STEP6_TEXT_REMOVE_PROMPT = (
    "이미지의 마스크 영역은 원래 글자 또는 얼룩이 있던 영역입니다.\n"
    "마스크 영역 안에서 글자, 얼룩, 잔상만 제거하고,\n"
    "그 아래에 원래 있었던 배경/디자인 표면을 자연스럽게 복원해 주세요.\n"
    "마스크 주변 약 20px~50px 범위의 색상, 질감, 그라데이션, 그림자, 하이라이트, 패턴, 도형 흐름을 참고해서\n"
    "마스크 영역 안이 주변과 끊김 없이 이어지도록 채워 주세요.\n"
    "중요:\n"
    "- 마스크 영역 밖은 절대 수정하지 마세요.\n"
    "- 마스크 영역 안에 새 글자를 만들지 마세요.\n"
    "- 새 상자, 테두리, 선, 아이콘, 장식 요소를 추가하지 마세요.\n"
    "- 원래 없던 디자인을 새로 만들지 마세요.\n"
    "- 단순히 흰색 배경으로 덮지 마세요.\n"
    "- 마스크 주변에 있는 기존 디자인 요소가 마스크 안으로 이어져야 한다면, 그 형태와 색감을 자연스럽게 연장해 주세요.\n"
    "- 캡슐형 바, 카드 박스, 그라데이션 배경, 사진 배경, 아이콘 영역, 장식 패턴 등 어떤 디자인이든 주변 정보를 기준으로 자연스럽게 복원해 주세요.\n"
)

"""
STEP6_TEXT_REMOVE_PROMPT = (
    "Remove the masked area and fill with matching surrounding background texture.\n"
    "Edit only the transparent pixels of the provided mask.\n"
    "Do not remove or alter any text outside the transparent mask area.\n"
    "Do not redesign, simplify, clean up, crop, move, or reinterpret the image.\n"
    "Do not add borders, boxes, guides, icons, or new design elements.\n"
)
"""

OVERLAY_REFINE_PROMPT_HEAD = (
    "이 이미지에 아래 문구를 넣으려고 해. 위치와 색을 확인해 보고 "
    "더 적합하도록 아래 정보를 수정해 봐.\n"
)
OVERLAY_REFINE_OUTPUT_GUIDE = (
    "\n\n"
    "출력은 JSON 객체 하나로만 해. 형식: {\"lines\": [...]}.\n"
    "lines 배열의 각 항목은 입력과 동일한 키를 유지해 "
    "(text, x, y, width, height, font_size, font_color, style, align). "
    "x, y, width, height 는 정수 픽셀, font_color 는 '#RRGGBB', "
    "style 은 'bold' 또는 'normal', align 은 'left' | 'center' | 'right' 중 하나. "
    "라인 개수와 순서는 입력과 동일하게 유지해."
)
OVERLAY_REFINE_SYSTEM_INSTRUCTION = (
    "You output only one valid JSON object. "
    "No markdown fences, no natural language before or after the JSON."
)


def _refine_overlay_layout_sync(
    api_key: str,
    vision_model: str,
    image_b64_png: str,
    current_lines: list[dict[str, Any]],
    image_width: int | None = None,
    image_height: int | None = None,
    reasoning_effort: str | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    """글자 제거 이미지 + 현재 합성 정보를 GPT 비전에 보내 다듬어진 lines 반환.

    image_width / image_height 를 받아 프롬프트에 좌표계 grounding 을 명시한다.

    Returns: (refined_lines, vision_prompt, raw_response)
    실패 시 원본 lines 그대로 반환 (raw 는 빈 문자열).
    """
    grounding = ""
    if image_width and image_height:
        grounding = f"""
이미지 크기는 {image_width}×{image_height} px 이야.
좌표는 이미지 좌상단을 (0, 0) 으로 두고,
x는 오른쪽으로, y는 아래쪽으로 증가하는 정수 픽셀 좌표계를 사용해.
각 라인의 x, y는 라인 박스의 좌상단 모서리야.
width/height는 박스의 가로·세로 길이야.
라인의 우하단 모서리는 (x+width, y+height)야.
중요 규칙:
- 출력은 JSON 객체 하나만 반환해.
- 형식은 반드시 {{"lines": [...]}} 로 유지해.
- lines 배열의 라인 개수와 순서는 입력과 동일하게 유지해.
- x, y, width, height는 모두 정수 픽셀로 반환해.
- 모든 라인은 이미지 안에 있어야 해.
- 기존 위치가 크게 틀리지 않으면 ±50px 이내에서만 보정해.
- 특별한 이유가 없으면 모든 문구의 align은 "left"로 해.
- font_color는 배경 대비가 좋게 조정하되, 제품 톤과 어울리는 색을 우선 사용해.
수정 전 JSON:
        """
        
    user_text = (
        OVERLAY_REFINE_PROMPT_HEAD
        + grounding
        + json.dumps(current_lines, ensure_ascii=False, indent=2)
        + OVERLAY_REFINE_OUTPUT_GUIDE
    )
    data_uri = f"data:image/png;base64,{image_b64_png}"
    client = OpenAI(api_key=api_key)
    input_messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": user_text},
                {"type": "input_image", "image_url": data_uri},
            ],
        },
    ]
    base_kwargs: dict[str, Any] = {
        "model": vision_model,
        "input": input_messages,
        "instructions": OVERLAY_REFINE_SYSTEM_INSTRUCTION,
    }
    if reasoning_effort:
        base_kwargs["reasoning"] = {"effort": reasoning_effort}
    try:
        resp = client.responses.create(
            **base_kwargs,
            text={"format": {"type": "json_object"}},
        )
    except Exception:
        # reasoning 미지원 또는 text.format 미지원 모델 대비 단계별 fallback
        fallback_kwargs = {k: v for k, v in base_kwargs.items() if k != "reasoning"}
        try:
            resp = client.responses.create(
                **fallback_kwargs,
                text={"format": {"type": "json_object"}},
            )
        except Exception:
            resp = client.responses.create(**fallback_kwargs)
    raw = (getattr(resp, "output_text", None) or "").strip()
    _, parsed = _parse_gpt_json(raw)
    refined: list[dict[str, Any]] = []
    if isinstance(parsed, dict):
        candidate = parsed.get("lines")
        if isinstance(candidate, list):
            for ln in candidate:
                if isinstance(ln, dict):
                    refined.append(ln)
    if not refined:
        return (current_lines, user_text, raw)
    # 원본과 같은 길이가 아니면 안전을 위해 원본 사용
    if len(refined) != len(current_lines):
        return (current_lines, user_text, raw)
    return (refined, user_text, raw)


_LABEL_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "text": ("text", "content", "copy", "label", "string", "value",
             "내용", "문구", "글자", "본문"),
    "x": ("x", "left", "pos_x", "posX", "x_position"),
    "y": ("y", "top", "pos_y", "posY", "y_position"),
    "width": ("width", "w", "너비"),
    "height": ("height", "h", "높이"),
    "font_size": ("font_size", "fontSize", "size", "font-size", "글자크기", "폰트크기"),
    "font_color": ("font_color", "color", "fontColor", "font-color",
                   "글자색", "색", "색상"),
    "style": ("style", "weight", "fontWeight", "font-weight"),
    "align": ("align", "textAlign", "text-align", "alignment"),
}


def _normalize_label(d: dict[str, Any]) -> dict[str, Any]:
    """라벨 dict 의 키를 표준 이름으로 매핑하고 nested position/size/font 를 평탄화."""
    out: dict[str, Any] = dict(d)
    pos = d.get("position") or d.get("pos") or d.get("좌표")
    if isinstance(pos, dict):
        out.setdefault("x", pos.get("x") or pos.get("left"))
        out.setdefault("y", pos.get("y") or pos.get("top"))
    size = d.get("size") or d.get("크기")
    if isinstance(size, dict):
        out.setdefault("width", size.get("width") or size.get("w"))
        out.setdefault("height", size.get("height") or size.get("h"))
    font = d.get("font") or d.get("폰트")
    if isinstance(font, dict):
        out.setdefault("font_size", font.get("size") or font.get("fontSize"))
        out.setdefault("font_color", font.get("color") or font.get("fontColor"))
        out.setdefault("style", font.get("weight") or font.get("style"))
    for canon, aliases in _LABEL_KEY_ALIASES.items():
        if out.get(canon) not in (None, ""):
            continue
        for a in aliases:
            if a == canon:
                continue
            v = out.get(a)
            if v not in (None, ""):
                out[canon] = v
                break
    return out


def _looks_like_label(d: Any) -> bool:
    if not isinstance(d, dict):
        return False
    norm = _normalize_label(d)
    txt = norm.get("text")
    return isinstance(txt, str) and bool(txt.strip())


def _extract_layout_labels(layout: Any) -> list[dict[str, Any]]:
    """3단계 JSON 에서 label 배열(문구 + 좌표 + 폰트 등)을 재귀적으로 추출."""
    if _looks_like_label(layout):
        return [_normalize_label(layout)]
    if isinstance(layout, list):
        direct = [_normalize_label(x) for x in layout if _looks_like_label(x)]
        if direct:
            return direct
        nested: list[dict[str, Any]] = []
        for item in layout:
            nested.extend(_extract_layout_labels(item))
        return nested
    if isinstance(layout, dict):
        best: list[dict[str, Any]] = []
        for v in layout.values():
            cands = _extract_layout_labels(v)
            if len(cands) > len(best):
                best = cands
        return best
    return []


_DECORATION_CHARSET = (
    r"\s.,·•‣▪▶○●◦◯⁃・"
    r"♥♡☆★✦✧"
    r"\-–—\*\+"
    r"'\"‘’“”"
    r"!?！？"
)
_LEADING_DECORATION_RE = re.compile(rf"^[{_DECORATION_CHARSET}]+")
_TRAILING_DECORATION_RE = re.compile(rf"[{_DECORATION_CHARSET}]+$")
# 앞쪽 일련번호 (예: "(3). ", "1. ", "[3] ", "1) ") 제거용
_LEADING_NUMBER_LABEL_RE = re.compile(r"^\s*[\(\[]?\d{1,3}[\)\]]?[.:]?\s+")


_INTERNAL_QUOTE_RE = re.compile(r"['\"‘’“”]")


def _strip_leading_decoration(s: str) -> str:
    """선행/후행 불릿·따옴표·문장부호·장식 문자 제거 (• · ♥ - * ' " . , ! ? 등).

    추가로 OCR 이 잡곤 하는 일련번호 라벨 ('(3). ', '1. ', '[3] ', '1) ' 등) 도
    선행 부분에서 제거한다 — 그래야 layout 텍스트(번호 없는 본문) 와 매칭된다.
    """
    s = (s or "").strip()
    s = _LEADING_NUMBER_LABEL_RE.sub("", s)
    s = _LEADING_DECORATION_RE.sub("", s)
    s = _TRAILING_DECORATION_RE.sub("", s)
    return s


def _normalize_match_token(t: str) -> str:
    """매칭용 토큰 정규화: 양끝 장식 제거 + 토큰 내부의 따옴표 제거."""
    return _INTERNAL_QUOTE_RE.sub("", _strip_leading_decoration(t))


def _tokenize_for_match(s: str) -> list[str]:
    """문자열을 매칭용 토큰 리스트로 변환 (양끝 장식 제거 + 토큰별 내부 따옴표 제거)."""
    cleaned = _strip_leading_decoration(s or "")
    return [t for t in (_normalize_match_token(tok) for tok in cleaned.split()) if t]


def _normalize_for_similarity(s: str) -> str:
    """글자수 유사도 비교용 정규화: 양끝 장식 제거 + 내부 따옴표 제거 + 모든 공백 제거 + 소문자."""
    cleaned = _strip_leading_decoration(s or "")
    cleaned = _INTERNAL_QUOTE_RE.sub("", cleaned)
    cleaned = "".join(cleaned.split())
    return cleaned.lower()


def _char_similarity(a: str, b: str) -> float:
    """글자수 기준 유사도 (difflib.SequenceMatcher ratio)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(a=a, b=b).ratio()


def _layout_text_lines(layout_texts: list[str]) -> set[str]:
    """layout_texts 의 각 줄을 (양쪽 공백 제거된) 원문 + 선행 장식 제거 버전 양쪽 모두 포함."""
    out: set[str] = set()
    for t in layout_texts or []:
        if not isinstance(t, str):
            continue
        for line in t.splitlines():
            s = line.strip()
            if s:
                out.add(s)
                stripped = _strip_leading_decoration(s)
                if stripped:
                    out.add(stripped)
    return out


def _sample_text_color_and_weight(
    image_bytes: bytes, box: tuple[int, int, int, int]
) -> tuple[str, str]:
    """OCR 영역 픽셀에서 글자색(평균 RGB) + bold 여부를 추정.

    - 다크/라이트 후보를 둘 다 뽑고, 면적이 적은 쪽을 글자로 선택한다.
      (글자는 보통 stroke 라서 배경보다 면적이 작다 — 회색 배경 + 흰 글자
       처럼 평균이 중간으로 가는 케이스에서도 잘 동작.)
    - bold: 글자 픽셀 밀도 (>= 0.18) 이면 'bold', 아니면 'normal'.
    """
    try:
        x, y, w, h = box
        if w <= 0 or h <= 0:
            return ("#111111", "bold")
        with Image.open(BytesIO(image_bytes)) as im:
            img = im.convert("RGB")
            iw, ih = img.size
            cx0 = max(0, min(iw - 1, x))
            cy0 = max(0, min(ih - 1, y))
            cx1 = max(cx0 + 1, min(iw, x + w))
            cy1 = max(cy0 + 1, min(ih, y + h))
            crop = img.crop((cx0, cy0, cx1, cy1))
        pixels = list(crop.getdata())
        if not pixels:
            return ("#111111", "bold")
        n = len(pixels)
        lums = [0.299 * r + 0.587 * g + 0.114 * b for (r, g, b) in pixels]
        # 양쪽 분위수에서 다크/라이트 평균 명도 추정
        sorted_lums = sorted(lums)
        q = max(1, n // 5)
        dark_lum = sum(sorted_lums[:q]) / q
        light_lum = sum(sorted_lums[-q:]) / q
        contrast = light_lum - dark_lum
        if contrast < 25:
            # 거의 균일 → 평균색을 그대로 반환
            ar = sum(p[0] for p in pixels) / n
            ag = sum(p[1] for p in pixels) / n
            ab = sum(p[2] for p in pixels) / n
            return (f"#{int(ar):02X}{int(ag):02X}{int(ab):02X}", "normal")
        # 중간 명도를 경계로 다크/라이트 픽셀을 분리
        mid = (dark_lum + light_lum) / 2
        margin = max(15.0, contrast * 0.15)
        dark_pixels = [pixels[i] for i, lm in enumerate(lums) if lm < mid - margin]
        light_pixels = [pixels[i] for i, lm in enumerate(lums) if lm > mid + margin]
        # 둘 중 면적이 적은 쪽을 글자로 (글자는 보통 stroke 라서 적음)
        if not dark_pixels and not light_pixels:
            ar = sum(p[0] for p in pixels) / n
            ag = sum(p[1] for p in pixels) / n
            ab = sum(p[2] for p in pixels) / n
            return (f"#{int(ar):02X}{int(ag):02X}{int(ab):02X}", "normal")
        if not light_pixels:
            text_pixels = dark_pixels
        elif not dark_pixels:
            text_pixels = light_pixels
        elif len(dark_pixels) <= len(light_pixels):
            text_pixels = dark_pixels
        else:
            text_pixels = light_pixels
        tn = len(text_pixels)
        ar = sum(p[0] for p in text_pixels) / tn
        ag = sum(p[1] for p in text_pixels) / tn
        ab = sum(p[2] for p in text_pixels) / tn
        color = f"#{int(ar):02X}{int(ag):02X}{int(ab):02X}"
        density = tn / n
        weight = "bold" if density >= 0.18 else "normal"
        return (color, weight)
    except Exception:
        return ("#111111", "bold")


_LEADING_BULLET_RE = re.compile(
    r"^[\s•●·・‧∙◦▪▫■‣⁃]+"
)


def _strip_leading_bullet(text: Any) -> Any:
    """맨 앞의 일련번호 라벨 + 불릿(•, ●, ·, ・ 등) + 인접 공백을 제거.
    비문자열은 그대로 반환. (예: "(3). 자고..." → "자고...", "• 효과..." → "효과...")
    """
    if not isinstance(text, str):
        return text
    out = _LEADING_NUMBER_LABEL_RE.sub("", text)
    out = _LEADING_BULLET_RE.sub("", out)
    # 일련번호 + 불릿이 결합된 경우 (예: "(3). • 자고...") 한 번 더 시도
    out = _LEADING_NUMBER_LABEL_RE.sub("", out)
    return out


def _apply_pixel_styles_to_lines(
    image_bytes: bytes, lines: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """각 라벨의 (x,y,w,h) 영역을 image_bytes 에서 샘플링해 font_color, style 을 재계산."""
    out: list[dict[str, Any]] = []
    for ln in lines or []:
        if not isinstance(ln, dict):
            continue
        ln2 = dict(ln)
        try:
            x = int(ln.get("x", 0) or 0)
            y = int(ln.get("y", 0) or 0)
            w = int(ln.get("width", 0) or 0)
            h = int(ln.get("height", 0) or 0)
            color, weight = _sample_text_color_and_weight(image_bytes, (x, y, w, h))
            ln2["font_color"] = color
            ln2["style"] = weight
        except Exception:
            pass
        out.append(ln2)
    return out


def _expand_matched_to_ocr_lines(matched: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """매칭 결과를 개별 OCR 라인으로 평탄화.

    그룹 매칭 (`_members` 포함) 은 각 멤버 OCR 라인을, 단일 라인 매칭은 그대로 반환.
    """
    out: list[dict[str, Any]] = []
    for entry in matched or []:
        if not isinstance(entry, dict):
            continue
        members = entry.get("_members")
        if isinstance(members, list) and members:
            for m in members:
                if isinstance(m, dict):
                    out.append({k: v for k, v in m.items() if k != "_members"})
        else:
            out.append({k: v for k, v in entry.items() if k != "_members"})
    return out


def _merge_ocr_with_layout_styles(
    ocr_lines: list[dict[str, Any]], layout_labels: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """OCR 라인의 x/y/w/h 좌표 + step3 라벨의 font_size/color/style/align 결합.

    매칭: OCR text(strip) 가 layout label.text 의 한 줄(strip) 과 정확히 일치할 때만 스타일 적용.
    """
    out: list[dict[str, Any]] = []
    for ocr in ocr_lines:
        if not isinstance(ocr, dict):
            continue
        ocr_text = (ocr.get("text") or "").strip()
        if not ocr_text:
            continue
        ocr_stripped = _strip_leading_decoration(ocr_text)
        best: dict[str, Any] | None = None
        for lab in layout_labels:
            if not isinstance(lab, dict):
                continue
            lab_text = lab.get("text") or ""
            for line in lab_text.splitlines():
                ls = line.strip()
                if not ls:
                    continue
                if ls == ocr_text or _strip_leading_decoration(ls) == ocr_stripped:
                    best = lab
                    break
            if best:
                break
        merged: dict[str, Any] = {
            "text": ocr_text,
            "x": ocr.get("x", 0),
            "y": ocr.get("y", 0),
            "width": ocr.get("width", 0),
            "height": ocr.get("height", 0),
        }
        if best:
            for k in ("font_size", "font_color", "style"):
                v = best.get(k)
                if v not in (None, ""):
                    merged[k] = v
        merged["align"] = "left"
        out.append(merged)
    return out


def _join_ocr_group_text(lines: list[dict[str, Any]]) -> str:
    parts = []
    for ln in lines:
        t = (ln.get("text") or "").strip() if isinstance(ln, dict) else ""
        if t:
            parts.append(t)
    return " ".join(parts)


def _merge_ocr_group_box(lines: list[dict[str, Any]]) -> dict[str, Any]:
    """여러 OCR 라인을 하나의 라벨로 합쳐 bounding box + 합쳐진 text + _members 반환.

    _members 에는 원본 OCR 라인들이 그대로 보관되어 step 6 가 라인별 좌표를 꺼내 쓸 수 있다.
    """
    xs = [int(ln.get("x", 0) or 0) for ln in lines]
    ys = [int(ln.get("y", 0) or 0) for ln in lines]
    rights = [int(ln.get("x", 0) or 0) + int(ln.get("width", 0) or 0) for ln in lines]
    bottoms = [int(ln.get("y", 0) or 0) + int(ln.get("height", 0) or 0) for ln in lines]
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(rights), max(bottoms)
    first = lines[0]
    return {
        "text": _join_ocr_group_text(lines),
        "x": x0,
        "y": y0,
        "width": max(0, x1 - x0),
        "height": max(0, y1 - y0),
        "font_size": int(first.get("font_size") or 0),
        "_members": [dict(ln) for ln in lines],
    }


def _bbox_area_of_lines(lines: list[dict[str, Any]]) -> int:
    if not lines:
        return 0
    xs = [int(ln.get("x", 0) or 0) for ln in lines]
    ys = [int(ln.get("y", 0) or 0) for ln in lines]
    rs = [int(ln.get("x", 0) or 0) + int(ln.get("width", 0) or 0) for ln in lines]
    bs = [int(ln.get("y", 0) or 0) + int(ln.get("height", 0) or 0) for ln in lines]
    return max(0, max(rs) - min(xs)) * max(0, max(bs) - min(ys))


def _match_ocr_to_layout_texts(
    ocr_lines: list[dict[str, Any]], layout_texts: list[str]
) -> list[dict[str, Any]]:
    """4단계 입력 문구의 각 줄(선행 장식 제거 후 토큰화) 과 정확히 같은 토큰 시퀀스를 형성하는
    OCR 라인 부분집합(OCR 등장 순서 유지, 인접 여부는 무관) 을 찾아 매칭한다.

    단일 라인 매칭은 OCR 라인 그대로, 다중 라인 매칭은 bounding box 와 _members 를 묶은
    합본 항목으로 반환. 동일 토큰 시퀀스를 만드는 부분집합이 여러 개일 경우 bounding box
    면적이 가장 작은 (가장 공간적으로 응집된) 조합을 선택한다.
    """
    # 1) 4단계 입력을 줄 단위 + 토큰 단위로 정리. 긴 줄부터 우선 매칭.
    raw_lines: list[str] = []
    for t in layout_texts or []:
        if isinstance(t, str):
            for ln in t.splitlines():
                s = ln.strip()
                if s:
                    raw_lines.append(s)
    seen: set[tuple[str, ...]] = set()
    layout_entries: list[tuple[tuple[str, ...], str]] = []
    for line in raw_lines:
        toks = tuple(_tokenize_for_match(line))
        if not toks or toks in seen:
            continue
        seen.add(toks)
        layout_entries.append((toks, line))
    layout_entries.sort(key=lambda x: -len(x[0]))

    # 2) OCR 라인 토큰화 (양끝 장식 제거 + 토큰별 내부 따옴표 제거)
    n = len(ocr_lines)
    ocr_tokens: list[list[str]] = []
    for ln in ocr_lines:
        if isinstance(ln, dict):
            t = (ln.get("text") or "").strip()
            ocr_tokens.append(_tokenize_for_match(t))
        else:
            ocr_tokens.append([])

    used = [False] * n
    matched: list[dict[str, Any]] = []

    def _best_subset(target: list[str]) -> list[int] | None:
        """target 토큰열을 정확히 형성하는 (in OCR-order) 부분집합을 찾되,
        점수 (멤버 수, bounding-box 면적) 가 최소인 조합을 선택."""
        target_len = len(target)
        best: tuple[tuple[int, int], list[int]] | None = None

        def search(start_i: int, pos: int, chosen: list[int]) -> None:
            nonlocal best
            if best is not None and len(chosen) >= best[0][0]:
                # 이미 best 보다 멤버 수가 많거나 같음 → 더 좋아질 수 없음 (멤버 수 추가)
                # 같은 멤버 수에서는 bbox 가 작아야 갱신되는데, 추가로 멤버를 더 넣어야 하므로 같지 않음
                return
            if pos == target_len:
                if not chosen:
                    return
                mc = len(chosen)
                area = _bbox_area_of_lines([ocr_lines[i] for i in chosen])
                score = (mc, area)
                if best is None or score < best[0]:
                    best = (score, list(chosen))
                return
            remain = target_len - pos
            for i in range(start_i, n):
                if used[i]:
                    continue
                toks = ocr_tokens[i]
                k = len(toks)
                if not k or k > remain:
                    continue
                if target[pos : pos + k] != toks:
                    continue
                chosen.append(i)
                search(i + 1, pos + k, chosen)
                chosen.pop()

        search(0, 0, [])
        return best[1] if best else None

    # 4) 5단계 OCR 의 정규화 텍스트(공백 제거 + 따옴표 제거 + lowercase) 사전 계산
    ocr_norms: list[str] = [
        _normalize_for_similarity((ln.get("text") or "")) if isinstance(ln, dict) else ""
        for ln in ocr_lines
    ]

    def _best_fuzzy_subset(layout_line: str, threshold: float = 0.85) -> list[int] | None:
        """exact token 매칭 실패 시, 글자수 기준 유사도(>= threshold) 로 매칭.

        단일 OCR 라인과 1~4 줄 연속 그룹 후보 중 유사도 최고, 동일 시 멤버 수 최소,
        다음 bbox 최소 순으로 선택.
        """
        layout_norm = _normalize_for_similarity(layout_line)
        if not layout_norm:
            return None
        cands: list[tuple[float, int, int, list[int]]] = []
        # 단일 라인
        for i in range(n):
            if used[i] or not ocr_norms[i]:
                continue
            sim = _char_similarity(layout_norm, ocr_norms[i])
            if sim >= threshold:
                area = _bbox_area_of_lines([ocr_lines[i]])
                cands.append((sim, 1, area, [i]))
        # 2~4 줄 OCR 연속 그룹 (in OCR-order, 인접 인덱스)
        for size in (2, 3, 4):
            if size > n:
                break
            for start in range(n - size + 1):
                indices = list(range(start, start + size))
                if any(used[k] for k in indices):
                    continue
                joined_norm = "".join(ocr_norms[k] for k in indices)
                if not joined_norm:
                    continue
                sim = _char_similarity(layout_norm, joined_norm)
                if sim >= threshold:
                    area = _bbox_area_of_lines([ocr_lines[k] for k in indices])
                    cands.append((sim, size, area, indices))
        if not cands:
            return None
        # 유사도 ↓, 멤버 수 ↑, bbox ↑ 순으로 정렬 → 첫 번째 가 최선
        cands.sort(key=lambda c: (-c[0], c[1], c[2]))
        return cands[0][3]

    for tokens_tuple, raw_line in layout_entries:
        target = list(tokens_tuple)
        indices = _best_subset(target)
        if not indices:
            # exact 실패 → 글자수 85% 이상 유사 매칭 fallback
            indices = _best_fuzzy_subset(raw_line)
        if not indices:
            continue
        for i in indices:
            used[i] = True
        if len(indices) == 1:
            matched.append(dict(ocr_lines[indices[0]]))
        else:
            matched.append(_merge_ocr_group_box([ocr_lines[i] for i in indices]))

    return matched


def _extract_texts_from_layout(node: Any) -> list[str]:
    """3단계 JSON 에서 'text' 키의 문자열 값을 재귀적으로 수집."""
    out: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "text" and isinstance(v, str) and v.strip():
                out.append(v.strip())
            else:
                out.extend(_extract_texts_from_layout(v))
    elif isinstance(node, list):
        for item in node:
            out.extend(_extract_texts_from_layout(item))
    return out

STEP3_JSON_SYSTEM_INSTRUCTION = (
    "You output only one valid JSON object with this exact schema (no markdown, "
    "no prose): {\"labels\": [ {\"text\": \"<string, may contain \\n>\", "
    "\"x\": <int px>, \"y\": <int px>, \"width\": <int px>, \"height\": <int px>, "
    "\"font_size\": <int px>, \"font_color\": \"<#RRGGBB>\", "
    "\"style\": \"bold\" | \"normal\", \"align\": \"left\" | \"center\" | \"right\"} ] }. "
    "Coordinates are pixels from the image top-left; the image canvas size is the "
    "width_px x height_px given in the user input. Every label must include every field."
)


def _openai_chat_json_sync(
    api_key: str,
    model: str,
    user_text: str,
) -> tuple[str, dict[str, Any] | None]:
    """OpenAI 챗(텍스트 전용)으로 JSON 응답 생성."""
    client = OpenAI(api_key=api_key)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": STEP3_JSON_SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_text},
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(model=model, messages=messages)
    raw = (resp.choices[0].message.content or "").strip()
    return _parse_gpt_json(raw)


def _openai_image_edit_sync(
    api_key: str,
    raw: bytes,
    edit_prompt: str,
    api_size: str,
) -> dict[str, Any]:
    client = OpenAI(api_key=api_key)
    result = client.images.edit(
        model=OPENAI_IMAGE_MODEL,
        image=BytesIO(raw),
        prompt=edit_prompt,
        size=api_size,
        quality=OPENAI_IMAGE_QUALITY,
        output_format="png",
    )
    if not result.data:
        raise RuntimeError("API가 이미지 데이터를 반환하지 않았습니다.")
    b64_data = result.data[0].b64_json
    if not b64_data:
        raise RuntimeError("base64 이미지가 비어 있습니다.")
    out_bytes = base64.b64decode(b64_data)
    with Image.open(BytesIO(out_bytes)) as gen:
        gw, gh = gen.size
    return {"b64_json": b64_data, "output_width": gw, "output_height": gh}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    if not _INDEX_HTML_PATH.is_file():
        raise HTTPException(
            status_code=500, detail=f"템플릿이 없습니다: {_INDEX_HTML_PATH}"
        )
    return HTMLResponse(_INDEX_HTML_PATH.read_text(encoding="utf-8"))


# ───────────────────────── 단계별 입출력 로깅 ─────────────────────────
# 각 요청마다 logs/{timestamp}_{rand}/ 디렉토리가 생성되고, 모든 step_start /
# step_done / step_error / step_skip / complete 이벤트가 페이지별 JSON 파일로
# 저장된다. base64 이미지 필드는 .png 로 디코딩되어 별도 파일로 분리되고
# JSON 안에는 파일 경로만 남는다.
_LOG_ROOT = Path(__file__).resolve().parent / "logs"
_CACHE_DIR = Path(__file__).resolve().parent / "cache" / "pre_step6"


def _save_pre_step6_cache(
    product_image_bytes: bytes,
    product_filename: str,
    step4_image_b64: str,
    first_image_width: int,
    first_image_height: int,
    step3_ocr_lines: list[dict[str, Any]],
    step3_ocr_text: str,
    matched_ocr_lines: list[dict[str, Any]],
    matched_lines_compact: list[dict[str, Any]],
    layout_texts: list[str],
    all_image_prompts: list[dict[str, Any]] | None,
    image_prompt: dict[str, Any] | None,
    ui_prompt: str = "",
    ocr_text: str = "",
    gpt_json: dict[str, Any] | None = None,
) -> None:
    """제품 사진 + 1~5단계 결과를 cache 디렉토리에 저장."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (_CACHE_DIR / "product.png").write_bytes(product_image_bytes)
        (_CACHE_DIR / "step4_image.png").write_bytes(base64.b64decode(step4_image_b64))
        meta = {
            "product_filename": product_filename,
            "first_image_width": first_image_width,
            "first_image_height": first_image_height,
            "step3_ocr_lines": step3_ocr_lines,
            "step3_ocr_text": step3_ocr_text,
            "matched_ocr_lines": matched_ocr_lines,
            "matched_lines_compact": matched_lines_compact,
            "layout_texts": layout_texts,
            "all_image_prompts": all_image_prompts or [],
            "image_prompt": image_prompt,
            "ui_prompt": ui_prompt,
            "ocr_text": ocr_text,
            "gpt_json": gpt_json,
        }
        (_CACHE_DIR / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[cache] 5단계까지 결과 저장됨: {_CACHE_DIR}")
    except Exception as ex:
        print(f"[cache] 저장 실패: {ex}")


def _load_pre_step6_cache() -> dict[str, Any] | None:
    """저장된 5단계까지 결과 로드."""
    meta_path = _CACHE_DIR / "meta.json"
    product_path = _CACHE_DIR / "product.png"
    step4_path = _CACHE_DIR / "step4_image.png"
    if not (meta_path.exists() and product_path.exists() and step4_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["product_bytes"] = product_path.read_bytes()
        meta["step4_image_b64"] = base64.b64encode(step4_path.read_bytes()).decode("ascii")
        return meta
    except Exception as ex:
        print(f"[cache] 로드 실패: {ex}")
        return None

_session_dir_var: ContextVar[Path | None] = ContextVar(
    "_session_dir_var", default=None
)
_step_counter_var: ContextVar[dict[str, int] | None] = ContextVar(
    "_step_counter_var", default=None
)

_B64_IMAGE_KEYS = {
    "image_b64", "base_image_b64", "first_image_b64", "second_image_b64",
    "b64_json", "labeled_image_b64", "image_a_b64", "image_b_b64",
    "diff_image_b64", "overlay_image_b64",
}
_LOGGED_EVENTS = {"step_start", "step_done", "step_error", "step_skip", "complete"}


def _new_session_dir() -> Path:
    name = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    d = _LOG_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80] or "item"


def _decode_b64_to(path: Path, b64: str) -> bool:
    try:
        path.write_bytes(base64.b64decode(b64))
        return True
    except Exception:
        return False


def _sanitize_for_log(node: Any, out_dir: Path, prefix: str,
                      counter: list[int]) -> Any:
    """큰 base64 / 텍스트를 별도 파일로 빼고 JSON에는 경로/요약만 남긴다."""
    if isinstance(node, dict):
        result = {}
        for k, v in node.items():
            if (k in _B64_IMAGE_KEYS and isinstance(v, str) and len(v) > 200):
                counter[0] += 1
                img_name = f"{prefix}__{_safe_name(k)}_{counter[0]}.png"
                if _decode_b64_to(out_dir / img_name, v):
                    result[k] = {"_file": img_name, "_b64_len": len(v)}
                else:
                    txt_name = img_name + ".b64.txt"
                    (out_dir / txt_name).write_text(v, encoding="utf-8")
                    result[k] = {"_file": txt_name, "_b64_len": len(v)}
            elif isinstance(v, str) and len(v) > 8000:
                counter[0] += 1
                txt_name = f"{prefix}__{_safe_name(k)}_{counter[0]}.txt"
                (out_dir / txt_name).write_text(v, encoding="utf-8")
                result[k] = {"_file": txt_name, "_len": len(v)}
            else:
                result[k] = _sanitize_for_log(v, out_dir, prefix, counter)
        return result
    if isinstance(node, list):
        return [_sanitize_for_log(x, out_dir, prefix, counter) for x in node]
    return node


def _log_event(obj: dict[str, Any]) -> None:
    sdir = _session_dir_var.get()
    if sdir is None:
        return
    event = obj.get("event")
    if event not in _LOGGED_EVENTS:
        return
    step = str(obj.get("step") or event)
    page = obj.get("page")
    if page is None:
        # data 안에 page 가 들어있는 케이스 (일부 경로)
        data = obj.get("data") if isinstance(obj.get("data"), dict) else None
        if data is not None:
            page = data.get("page")
    page_label = "common" if page is None else f"page_{page}"
    page_dir = sdir / page_label
    page_dir.mkdir(parents=True, exist_ok=True)

    counters = _step_counter_var.get()
    if counters is None:
        counters = {}
        _step_counter_var.set(counters)
    key = f"{page_label}/{step}/{event}"
    counters[key] = counters.get(key, 0) + 1
    seq = counters[key]
    order = sum(counters.values())  # 전체 순번 (디렉토리 정렬용)

    base = f"{order:03d}_{_safe_name(step)}_{_safe_name(event)}"
    if seq > 1:
        base = f"{base}_{seq}"
    prefix = base
    sanitized = _sanitize_for_log(obj, page_dir, prefix, [0])
    try:
        (page_dir / f"{base}.json").write_text(
            json.dumps(sanitized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        # 로그 실패가 메인 스트림을 막지 않도록 swallow
        pass


def _ndjson(obj: dict[str, Any]) -> bytes:
    try:
        _log_event(obj)
    except Exception:
        pass
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


async def _generate_stream(
    raw: bytes,
    filename: str,
    suffix: str,
    prompt: str,
    use_cache: bool = False,
) -> AsyncIterator[bytes]:
    t_start = time.perf_counter()

    def elapsed_total() -> float:
        return round(time.perf_counter() - t_start, 2)

    # 캐시 모드: 1~5단계 결과를 cache 에서 로드
    cache: dict[str, Any] | None = None
    if use_cache:
        cache = _load_pre_step6_cache()
        if cache is None:
            yield _ndjson({
                "event": "step_error", "step": "cache",
                "elapsed": 0.0,
                "message": "저장된 5단계 결과가 없습니다. 처음부터 실행해주세요.",
            })
            yield _ndjson({"event": "complete", "elapsed": 0.0, "status": "error"})
            return
        # 캐시의 제품 사진과 파일명 사용
        raw = cache.get("product_bytes") or raw
        filename = cache.get("product_filename") or filename

    tmp_path: Path | None = None
    try:
        fd_tmp, tmp_name = tempfile.mkstemp(suffix=suffix, prefix="product_")
        os.close(fd_tmp)
        tmp_path = Path(tmp_name)
        tmp_path.write_bytes(raw)

        # 1) OCR
        yield _ndjson({"event": "step_start", "step": "ocr", "label": "OCR (Document AI)", "elapsed": elapsed_total()})
        t0 = time.perf_counter()
        if cache:
            ocr_text = cache.get("ocr_text", "") or ""
            ocr_err = None
        else:
            ocr_text, ocr_err = await asyncio.to_thread(ocr_image_file, tmp_path)
        step_elapsed = round(time.perf_counter() - t0, 2)
        if ocr_err:
            yield _ndjson({
                "event": "step_error", "step": "ocr",
                "elapsed": elapsed_total(), "step_elapsed": step_elapsed,
                "message": f"OCR 실패: {ocr_err}",
            })
            yield _ndjson({"event": "complete", "elapsed": elapsed_total(), "status": "error"})
            return
        yield _ndjson({
            "event": "step_done", "step": "ocr",
            "elapsed": elapsed_total(), "step_elapsed": step_elapsed,
            "data": {"ocr_text": ocr_text, "ocr_chars": len(ocr_text or "")},
        })

        # 2) GPT 텍스트(JSON)
        try:
            user_message = build_user_message(ocr_text, prompt)
        except RuntimeError as e:
            yield _ndjson({
                "event": "step_error", "step": "prompt_build",
                "elapsed": elapsed_total(), "message": str(e),
            })
            yield _ndjson({"event": "complete", "elapsed": elapsed_total(), "status": "error"})
            return

        api_key = _openai_api_key()
        if not api_key:
            yield _ndjson({
                "event": "step_error", "step": "config",
                "elapsed": elapsed_total(),
                "message": "OPENAI_API_KEY 가 config.py 또는 환경 변수에 없습니다.",
            })
            yield _ndjson({"event": "complete", "elapsed": elapsed_total(), "status": "error"})
            return

        yield _ndjson({
            "event": "step_start", "step": "gpt",
            "label": f"상세페이지 생성용 프롬프트 만들기",
            "elapsed": elapsed_total(),
            "data": {
                "model": GEMINI_TEXT_MODEL,
                "system_instruction": GEMINI_JSON_SYSTEM_INSTRUCTION,
                "user_message": user_message,
                "user_message_chars": len(user_message),
                "ui_prompt": prompt,
                "has_product_image": True,
            },
        })
        t0 = time.perf_counter()
        try:
            if cache:
                gpt_raw = ""
                gpt_json = cache.get("gpt_json")
            else:
                gpt_raw, gpt_json = await asyncio.to_thread(
                    _chat_sync, user_message, GEMINI_TEXT_MODEL, raw
                )
        except Exception as e:
            yield _ndjson({
                "event": "step_error", "step": "gpt",
                "elapsed": elapsed_total(), "step_elapsed": round(time.perf_counter() - t0, 2),
                "message": f"Gemini API 오류: {e}",
            })
            yield _ndjson({"event": "complete", "elapsed": elapsed_total(), "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)
        yield _ndjson({
            "event": "step_done", "step": "gpt",
            "elapsed": elapsed_total(), "step_elapsed": step_elapsed,
            "data": {"gpt_json": gpt_json, "gpt_raw": gpt_raw, "parsed": isinstance(gpt_json, dict)},
        })

        # 3) 첫 image2_prompt_en → 문구 정보 JSON 생성 (OpenAI 챗)
        first: dict[str, Any] | None = None
        prompts_all: list[Any] = []
        if isinstance(gpt_json, dict):
            prompts = gpt_json.get("image_prompts")
            if isinstance(prompts, list) and prompts and isinstance(prompts[0], dict):
                first = prompts[0]
                prompts_all = prompts

        if not first or not prompts_all:
            if not isinstance(gpt_json, dict):
                reason = "2단계 응답이 JSON 객체로 파싱되지 않아 3단계를 건너뜁니다."
            elif not isinstance(gpt_json.get("image_prompts"), list):
                reason = "2단계 JSON 에 image_prompts 배열이 없어 3단계를 건너뜁니다."
            else:
                reason = "image_prompts 가 비어 있어 3단계를 건너뜁니다."
            yield _ndjson({
                "event": "step_skip", "step": "image_gen",
                "elapsed": elapsed_total(),
                "message": reason,
            })
            yield _ndjson({"event": "complete", "elapsed": elapsed_total(), "status": "ok"})
            return

        image_prompt_full = json.dumps(first, ensure_ascii=False, indent=2)
        step3_user_text = f"{STEP3_TEXT_LAYOUT_PROMPT}\n\n{image_prompt_full}"
        yield _ndjson({
            "event": "step_start", "step": "image_gen",
            "label": f"문구 정보 JSON 생성 (OpenAI {OPENAI_TEXT_MODEL})",
            "elapsed": elapsed_total(),
            "data": {
                "model": OPENAI_TEXT_MODEL,
                "system_instruction": STEP3_JSON_SYSTEM_INSTRUCTION,
                "user_message": step3_user_text,
                "user_message_chars": len(step3_user_text),
                "image_prompt": image_prompt_full,
            },
        })
        t0 = time.perf_counter()
        try:
            if cache:
                step3_raw = ""
                step3_json = {"labels": []}
            else:
                step3_raw, step3_json = await asyncio.to_thread(
                    _openai_chat_json_sync, api_key, OPENAI_TEXT_MODEL, step3_user_text
                )
        except Exception as ex:
            yield _ndjson({
                "event": "step_error", "step": "image_gen",
                "elapsed": elapsed_total(), "step_elapsed": round(time.perf_counter() - t0, 2),
                "message": f"OpenAI API 오류: {ex}",
            })
            yield _ndjson({"event": "complete", "elapsed": elapsed_total(), "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)
        yield _ndjson({
            "event": "step_done", "step": "image_gen",
            "elapsed": elapsed_total(), "step_elapsed": step_elapsed,
            "data": {
                "layout_json": step3_json,
                "layout_raw": step3_raw,
                "parsed": isinstance(step3_json, dict),
            },
        })

        # 3.4) 3단계 결과 JSON 의 문구 + 사용자 제품 사진 → 상세페이지 이미지 생성
        if cache:
            layout_texts = cache.get("layout_texts", []) or []
        else:
            layout_labels = _extract_layout_labels(step3_json)
            layout_texts = [
                lab.get("text", "").strip() for lab in layout_labels
                if isinstance(lab, dict) and (lab.get("text") or "").strip()
            ]
            if not layout_texts and step3_raw:
                layout_texts = [step3_raw]
        texts_block = "\n".join(f"- {t}" for t in layout_texts)
        image2_prompt_en = (first.get("image2_prompt_en") or "").strip()
        image_gen2_prompt = f"{STEP3_5_IMAGE_GEN_PROMPT}\n{texts_block}".rstrip()
        if image2_prompt_en:
            image_gen2_prompt += f"\n\n{image2_prompt_en}"
        wpx = int(first.get("width_px") or 860)
        hpx = int(first.get("height_px") or 2000)
        gen_size = choose_api_size(wpx, hpx)
        image_gen2_quality = "low"
        yield _ndjson({
            "event": "step_start", "step": "image_gen2",
            "label": f"상세페이지 이미지 생성 ({OPENAI_IMAGE_MODEL}, {gen_size}, {image_gen2_quality})",
            "elapsed": elapsed_total(),
            "data": {
                "model": OPENAI_IMAGE_MODEL,
                "size": gen_size,
                "quality": image_gen2_quality,
                "available_options": IMAGE_GEN2_OPTIONS,
                "image_prompt": image_gen2_prompt,
                "layout_texts": layout_texts,
                "width_px": wpx,
                "height_px": hpx,
            },
        })
        t0 = time.perf_counter()
        first_image_b64: str | None = None
        first_image_width: int | None = None
        first_image_height: int | None = None
        try:
            if cache:
                first_image_b64 = cache["step4_image_b64"]
                first_image_width = int(cache["first_image_width"])
                first_image_height = int(cache["first_image_height"])
            else:
                img_out = await asyncio.to_thread(
                    _images_generate_sync, api_key, image_gen2_prompt, gen_size,
                    raw, filename, OPENAI_IMAGE_MODEL, image_gen2_quality,
                )
                first_image_b64 = img_out["b64_json"]
                first_image_width = int(img_out["output_width"])
                first_image_height = int(img_out["output_height"])
        except Exception as ex:
            yield _ndjson({
                "event": "step_error", "step": "image_gen2",
                "elapsed": elapsed_total(), "step_elapsed": round(time.perf_counter() - t0, 2),
                "message": str(ex),
            })
            yield _ndjson({"event": "complete", "elapsed": elapsed_total(), "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)
        yield _ndjson({
            "event": "step_done", "step": "image_gen2",
            "elapsed": elapsed_total(), "step_elapsed": step_elapsed,
            "data": {
                "first_image_b64": first_image_b64,
                "first_image_width": first_image_width,
                "first_image_height": first_image_height,
            },
        })

        # 3.5) 새 이미지 생성 결과 OCR (Document AI, 4단계 프롬프트 입력 보조)
        yield _ndjson({
            "event": "step_start", "step": "image_gen_ocr",
            "label": "상세페이지 이미지 OCR (Document AI)",
            "elapsed": elapsed_total(),
            "data": {"input_image_ref": "image_gen2"},
        })
        t0 = time.perf_counter()
        step3_ocr_text = ""
        step3_ocr_lines: list[dict[str, Any]] = []
        try:
            if cache:
                step3_ocr_text = cache.get("step3_ocr_text", "") or ""
                step3_ocr_lines = cache.get("step3_ocr_lines", []) or []
            else:
                ocr_step3 = await asyncio.to_thread(
                    _ocr_with_positions_sync, base64.b64decode(first_image_b64)
                )
                step3_ocr_text = ocr_step3["full_text"]
                step3_ocr_lines = ocr_step3["lines"]
        except Exception as ex:
            yield _ndjson({
                "event": "step_error", "step": "image_gen_ocr",
                "elapsed": elapsed_total(), "step_elapsed": round(time.perf_counter() - t0, 2),
                "message": str(ex),
            })
            yield _ndjson({"event": "complete", "elapsed": elapsed_total(), "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)
        yield _ndjson({
            "event": "step_done", "step": "image_gen_ocr",
            "elapsed": elapsed_total(), "step_elapsed": step_elapsed,
            "data": {
                "ocr_text": step3_ocr_text,
                "ocr_chars": len(step3_ocr_text or ""),
                "lines": step3_ocr_lines,
                "line_count": len(step3_ocr_lines),
            },
        })

        # 6) 5단계 OCR 결과 중 4단계 입력 문구(layout_texts) 와 일치하는 라인 좌표만 지우기
        if cache:
            matched_ocr_lines = cache.get("matched_ocr_lines", []) or []
            matched_lines_compact = cache.get("matched_lines_compact", []) or []
        else:
            matched_lines = _match_ocr_to_layout_texts(step3_ocr_lines, layout_texts)
            # 그룹 매칭(2~4 줄)도 지울 영역은 각 OCR 라인의 좌표를 개별 항목으로 전개
            matched_ocr_lines = _expand_matched_to_ocr_lines(matched_lines)
            matched_lines_compact = [
                {
                    "text": ln.get("text", ""),
                    "x": ln.get("x", 0),
                    "y": ln.get("y", 0),
                    "width": ln.get("width", 0),
                    "height": ln.get("height", 0),
                }
                for ln in matched_ocr_lines
                if isinstance(ln, dict)
            ]

        # 5단계까지 결과를 cache 디렉토리에 저장 (정상 모드에서만)
        if not cache:
            try:
                _save_pre_step6_cache(
                    product_image_bytes=raw,
                    product_filename=filename,
                    step4_image_b64=first_image_b64,
                    first_image_width=first_image_width,
                    first_image_height=first_image_height,
                    step3_ocr_lines=step3_ocr_lines,
                    step3_ocr_text=step3_ocr_text,
                    matched_ocr_lines=matched_ocr_lines,
                    matched_lines_compact=matched_lines_compact,
                    layout_texts=layout_texts,
                    all_image_prompts=prompts_all,
                    image_prompt=first,
                    ui_prompt=prompt,
                    ocr_text=ocr_text,
                    gpt_json=gpt_json,
                )
            except Exception as _cache_ex:
                print(f"[cache] 저장 호출 실패: {_cache_ex}")

        text_prompt = STEP6_TEXT_REMOVE_PROMPT
        # 지울 영역은 프롬프트가 아니라 마스크 PNG 로 API 에 전달
        coarse_png_bytes: bytes | None = None
        coarse_file_path: str | None = None
        refined_png_bytes: bytes | None = None
        refined_file_path: str | None = None
        refine_info: dict[str, Any] = {}
        if matched_lines_compact:
            try:
                _expanded_rects = _expand_rects_with_margin(
                    matched_lines_compact, 5, first_image_width, first_image_height
                )
                coarse_png_bytes = _build_text_mask_png(
                    first_image_width, first_image_height, _expanded_rects
                )
                coarse_file_path = _save_mask_to_session(
                    coarse_png_bytes, label="image_gen_text_coarse"
                )
            except Exception as ex:
                print(f"[image_gen_text] 거친 마스크 빌드 실패: {ex}")
                coarse_png_bytes = None
            if coarse_png_bytes:
                refined_png_bytes, refined_file_path, refine_info = _refine_mask_with_logging(
                    coarse_png_bytes, base64.b64decode(first_image_b64),
                    label="image_gen_text",
                )
        # API 에 전달할 마스크는 coarse (사각) 마스크 — 모델이 약간의 여유 공간을 갖고 배경을 다시 그릴 수 있게
        mask_png_bytes = coarse_png_bytes
        # API 입력 이미지: 4단계 이미지의 refined mask 영역을 주변 색으로 미리 메꾼 것
        step3_bytes_main = base64.b64decode(first_image_b64)
        api_input_bytes_main = step3_bytes_main
        inpainted_file_path: str | None = None
        if refined_png_bytes:
            api_input_bytes_main, inpainted_file_path = await asyncio.to_thread(
                _inpaint_and_save, step3_bytes_main, refined_png_bytes,
                "image_gen_text",
            )
        step4_quality = "medium"
        yield _ndjson({
            "event": "step_start", "step": "image_gen_text",
            "label": f"글자를 제거한 이미지 생성 ({OPENAI_IMAGE_MODEL}, {gen_size}, {step4_quality})",
            "elapsed": elapsed_total(),
            "data": {
                "model": OPENAI_IMAGE_MODEL,
                "quality": step4_quality,
                "available_options": IMAGE_GEN_TEXT_OPTIONS,
                "size": gen_size,
                "text_prompt": text_prompt,
                "mask_rect_count": len(matched_lines_compact),
                "mask_rects": matched_lines_compact,
                "mask_file": coarse_file_path,
                "refined_mask_file": refined_file_path,
                "refine_info": refine_info,
                "inpainted_image_file": inpainted_file_path,
                "inpainted_image_b64": base64.b64encode(api_input_bytes_main).decode("ascii"),
                "base_image_b64": first_image_b64,
                "base_image_width": first_image_width,
                "base_image_height": first_image_height,
            },
        })
        t0 = time.perf_counter()
        second_image_b64: str | None = None
        second_image_width: int | None = None
        second_image_height: int | None = None
        stage_images: dict[str, Any] = {}
        try:
            # 마스크 영역만 잘라 API 호출 후 응답을 원본에 다시 붙임
            cp = await asyncio.to_thread(
                _crop_paste_api_call,
                api_key, text_prompt,
                api_input_bytes_main, mask_png_bytes,
                OPENAI_IMAGE_MODEL, OPENAI_IMAGE_QUALITY,
            )
            final_bytes = cp["final_bytes"]
            second_image_b64 = base64.b64encode(final_bytes).decode("ascii")
            second_image_width = int(cp["final_width"])
            second_image_height = int(cp["final_height"])
            stage_images = {
                "api_input_b64": base64.b64encode(api_input_bytes_main).decode("ascii"),
                "mask_b64": base64.b64encode(cp["cropped_mask_bytes"]).decode("ascii") if cp.get("cropped_mask_bytes") else (base64.b64encode(mask_png_bytes).decode("ascii") if mask_png_bytes else None),
                "cropped_sent_b64": base64.b64encode(cp["cropped_input_bytes"]).decode("ascii") if cp["cropped_input_bytes"] else None,
                "api_response_b64": base64.b64encode(cp["api_response_bytes"]).decode("ascii") if cp["api_response_bytes"] else None,
                "final_b64": second_image_b64,
                "bbox": list(cp["bbox"]) if cp["bbox"] else None,
            }
            # [DEBUG] 5개 이미지 파일로 저장
            try:
                _dbg_dir = _session_dir_var.get() or (_LOG_ROOT / "ad_hoc")
                _dbg_dir.mkdir(parents=True, exist_ok=True)
                _dbg_stamp = time.strftime("%H%M%S") + "_" + uuid.uuid4().hex[:4]
                def _save(name: str, data: bytes | None) -> None:
                    if not data:
                        return
                    p = _dbg_dir / f"image_gen_text__{name}__{_dbg_stamp}.png"
                    p.write_bytes(data)
                    print(f"[image_gen_text][DEBUG] {name}: {p}")
                _save("1_api_input", api_input_bytes_main)
                _save("2_mask", cp.get("cropped_mask_bytes") or mask_png_bytes)
                _save("3_cropped_sent", cp["cropped_input_bytes"])
                _save("4_api_response", cp["api_response_bytes"])
                _save("5_final", final_bytes)
                print(f"[image_gen_text][DEBUG] bbox={cp['bbox']}")
            except Exception as _dbg_ex:
                print(f"[image_gen_text][DEBUG] 디버그 저장 실패: {_dbg_ex}")
        except Exception as ex:
            yield _ndjson({
                "event": "step_error", "step": "image_gen_text",
                "elapsed": elapsed_total(), "step_elapsed": round(time.perf_counter() - t0, 2),
                "message": str(ex),
            })
            yield _ndjson({"event": "complete", "elapsed": elapsed_total(), "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)

        # 5단계 OCR 의 text/좌표/font_size + 4단계 이미지 픽셀에서 샘플링한
        # font_color/style 로 1차 합성 정보 (overlay_labels) 생성
        # (step_done 직전에 계산하여 6단계 결과에 "다듬기 전 정보 preview" 를 포함)
        sampled_lines = await asyncio.to_thread(
            _apply_pixel_styles_to_lines,
            base64.b64decode(first_image_b64),
            matched_ocr_lines,
        )
        overlay_labels = []
        for ln in sampled_lines:
            if not isinstance(ln, dict):
                continue
            cleaned = _strip_leading_bullet(ln.get("text"))
            if not (isinstance(cleaned, str) and cleaned.strip()):
                continue
            overlay_labels.append({**ln, "align": "center", "text": cleaned})

        yield _ndjson({
            "event": "step_done", "step": "image_gen_text",
            "elapsed": elapsed_total(), "step_elapsed": step_elapsed,
            "data": {
                "image_b64": second_image_b64,
                "image_width": second_image_width,
                "image_height": second_image_height,
                "input_lines": overlay_labels,
                "base_image_b64": second_image_b64,
                "base_image_width": second_image_width,
                "base_image_height": second_image_height,
                "stage_images": stage_images,
            },
        })

        # 7단계: GPT 비전에 글자 제거 이미지 + 1차 합성 정보를 보내
        #        위치/색을 다듬어진 정보로 받기
        refined_labels = overlay_labels
        refine_raw = ""
        refine_prompt = ""
        yield _ndjson({
            "event": "step_start", "step": "overlay_refine",
            "label": f"합성 정보 다듬기 ({OPENAI_VISION_MODEL}) — 글자 제거 이미지 + 현재 라인",
            "elapsed": elapsed_total(),
            "data": {
                "model": OPENAI_VISION_MODEL,
                "reasoning_effort": OVERLAY_REFINE_DEFAULT_EFFORT,
                "available_options": OVERLAY_REFINE_OPTIONS,
                "input_line_count": len(overlay_labels),
                "input_lines": overlay_labels,
            },
        })
        t0 = time.perf_counter()
        try:
            refined_labels, refine_prompt, refine_raw = await asyncio.to_thread(
                _refine_overlay_layout_sync,
                api_key, OPENAI_VISION_MODEL, second_image_b64, overlay_labels,
                second_image_width, second_image_height,
                OVERLAY_REFINE_DEFAULT_EFFORT,
            )
        except Exception as ex:
            yield _ndjson({
                "event": "step_error", "step": "overlay_refine",
                "elapsed": elapsed_total(),
                "step_elapsed": round(time.perf_counter() - t0, 2),
                "message": str(ex),
            })
            # 다듬기 실패해도 합성 단계는 1차 정보로 진행
            refined_labels = overlay_labels
        step_elapsed = round(time.perf_counter() - t0, 2)
        yield _ndjson({
            "event": "step_done", "step": "overlay_refine",
            "elapsed": elapsed_total(), "step_elapsed": step_elapsed,
            "data": {
                "lines": refined_labels,
                "line_count": len(refined_labels),
                "input_lines": overlay_labels,
                "base_image_b64": second_image_b64,
                "base_image_width": second_image_width,
                "base_image_height": second_image_height,
                "refine_prompt": refine_prompt,
                "refine_raw": refine_raw,
            },
        })

        # 8단계: 다듬어진 합성 정보로 글자 제거 이미지에 글자 합성 (canvas)
        yield _ndjson({
            "event": "step_start", "step": "removed_text_overlay",
            "label": "글자 제거 이미지에 글자 합성 (canvas) — 7단계에서 다듬어진 정보 사용",
            "elapsed": elapsed_total(),
            "data": {
                "base_image_ref": "image_gen_text",
                "lines_ref": "overlay_refine",
                "line_count": len(refined_labels),
            },
        })
        yield _ndjson({
            "event": "step_done", "step": "removed_text_overlay",
            "elapsed": elapsed_total(), "step_elapsed": 0.0,
            "data": {
                "base_image_b64": second_image_b64,
                "base_image_width": second_image_width,
                "base_image_height": second_image_height,
                "lines": refined_labels,
                "line_count": len(refined_labels),
            },
        })

        yield _ndjson({"event": "complete", "elapsed": elapsed_total(), "status": "ok"})
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


@app.post("/api/generate")
async def api_generate(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    use_cache: str = Form(""),
) -> StreamingResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    try:
        with Image.open(BytesIO(raw)) as im:
            im.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지를 읽을 수 없습니다: {e}") from e

    suffix = Path(file.filename or "upload.png").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
        suffix = ".png"

    use_cache_bool = str(use_cache).strip().lower() in ("1", "true", "yes", "on")
    return _stream_response(
        _generate_stream(raw, file.filename or "upload", suffix, prompt, use_cache_bool)
    )


def _stream_response(gen: AsyncIterator[bytes]) -> StreamingResponse:
    sdir = _new_session_dir()

    async def _wrapped() -> AsyncIterator[bytes]:
        token_dir = _session_dir_var.set(sdir)
        token_cnt = _step_counter_var.set({})
        try:
            async for chunk in gen:
                yield chunk
        finally:
            try:
                _session_dir_var.reset(token_dir)
            except Exception:
                pass
            try:
                _step_counter_var.reset(token_cnt)
            except Exception:
                pass

    return StreamingResponse(
        _wrapped(),
        media_type="application/x-ndjson",
        headers={
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
            "X-Session-Dir": sdir.name,
        },
    )


_BIG_FORM_PART_SIZE = 50 * 1024 * 1024  # 50 MB — base64 이미지 단일 필드 허용


async def _parse_big_form(request: Request) -> Any:
    """base64 이미지를 form field 로 보낼 때 Starlette 기본 1MB 제한을 회피."""
    try:
        return await request.form(max_part_size=_BIG_FORM_PART_SIZE)
    except TypeError:
        # 구버전 Starlette 호환
        return await request.form()


@app.post("/api/step/ocr")
async def api_step_ocr(file: UploadFile = File(...)) -> StreamingResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    suffix = Path(file.filename or "upload.png").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
        suffix = ".png"

    async def gen() -> AsyncIterator[bytes]:
        t_start = time.perf_counter()
        tmp_path: Path | None = None
        try:
            fd_tmp, tmp_name = tempfile.mkstemp(suffix=suffix, prefix="product_")
            os.close(fd_tmp)
            tmp_path = Path(tmp_name)
            tmp_path.write_bytes(raw)
            yield _ndjson({"event": "step_start", "step": "ocr",
                           "label": "OCR (Document AI)", "elapsed": 0.0})
            t0 = time.perf_counter()
            ocr_text, ocr_err = await asyncio.to_thread(ocr_image_file, tmp_path)
            step_elapsed = round(time.perf_counter() - t0, 2)
            total = round(time.perf_counter() - t_start, 2)
            if ocr_err:
                yield _ndjson({"event": "step_error", "step": "ocr",
                               "elapsed": total, "step_elapsed": step_elapsed,
                               "message": f"OCR 실패: {ocr_err}"})
                yield _ndjson({"event": "complete", "elapsed": total, "status": "error"})
                return
            yield _ndjson({"event": "step_done", "step": "ocr",
                           "elapsed": total, "step_elapsed": step_elapsed,
                           "data": {"ocr_text": ocr_text, "ocr_chars": len(ocr_text or "")}})
            yield _ndjson({"event": "complete", "elapsed": total, "status": "ok"})
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    return _stream_response(gen())


@app.post("/api/step/gpt")
async def api_step_gpt(
    file: UploadFile = File(...),
    ocr_text: str = Form(""),
    prompt: str = Form(""),
) -> StreamingResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    async def gen() -> AsyncIterator[bytes]:
        t_start = time.perf_counter()
        try:
            user_message = build_user_message(ocr_text, prompt)
        except RuntimeError as e:
            yield _ndjson({"event": "step_error", "step": "gpt",
                           "elapsed": 0.0, "message": str(e)})
            yield _ndjson({"event": "complete", "elapsed": 0.0, "status": "error"})
            return
        yield _ndjson({
            "event": "step_start", "step": "gpt",
            "label": f"상세페이지 생성용 프롬프트 만들기 : Gemini ({GEMINI_TEXT_MODEL}) → JSON",
            "elapsed": 0.0,
            "data": {
                "model": GEMINI_TEXT_MODEL,
                "system_instruction": GEMINI_JSON_SYSTEM_INSTRUCTION,
                "user_message": user_message,
                "user_message_chars": len(user_message),
                "ui_prompt": prompt,
                "has_product_image": True,
            },
        })
        t0 = time.perf_counter()
        try:
            gpt_raw, gpt_json = await asyncio.to_thread(
                _chat_sync, user_message, GEMINI_TEXT_MODEL, raw
            )
        except Exception as e:
            total = round(time.perf_counter() - t_start, 2)
            yield _ndjson({"event": "step_error", "step": "gpt",
                           "elapsed": total, "step_elapsed": round(time.perf_counter() - t0, 2),
                           "message": f"Gemini API 오류: {e}"})
            yield _ndjson({"event": "complete", "elapsed": total, "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)
        total = round(time.perf_counter() - t_start, 2)
        yield _ndjson({"event": "step_done", "step": "gpt",
                       "elapsed": total, "step_elapsed": step_elapsed,
                       "data": {"gpt_json": gpt_json, "gpt_raw": gpt_raw,
                                "parsed": isinstance(gpt_json, dict)}})
        yield _ndjson({"event": "complete", "elapsed": total, "status": "ok"})

    return _stream_response(gen())


@app.post("/api/step/image_gen")
async def api_step_image_gen(
    image_prompt: str = Form(...),
) -> StreamingResponse:
    async def gen() -> AsyncIterator[bytes]:
        t_start = time.perf_counter()
        api_key = _openai_api_key()
        if not api_key:
            yield _ndjson({"event": "step_error", "step": "image_gen",
                           "elapsed": 0.0,
                           "message": "OPENAI_API_KEY 가 config.py 또는 환경 변수에 없습니다."})
            yield _ndjson({"event": "complete", "elapsed": 0.0, "status": "error"})
            return
        step3_user_text = f"{STEP3_TEXT_LAYOUT_PROMPT}\n\n{image_prompt}"
        yield _ndjson({
            "event": "step_start", "step": "image_gen",
            "label": f"문구 정보 JSON 생성 (OpenAI {OPENAI_TEXT_MODEL})",
            "elapsed": 0.0,
            "data": {
                "model": OPENAI_TEXT_MODEL,
                "system_instruction": STEP3_JSON_SYSTEM_INSTRUCTION,
                "user_message": step3_user_text,
                "user_message_chars": len(step3_user_text),
                "image_prompt": image_prompt,
            },
        })
        t0 = time.perf_counter()
        try:
            step3_raw, step3_json = await asyncio.to_thread(
                _openai_chat_json_sync, api_key, OPENAI_TEXT_MODEL, step3_user_text
            )
        except Exception as ex:
            total = round(time.perf_counter() - t_start, 2)
            yield _ndjson({"event": "step_error", "step": "image_gen",
                           "elapsed": total, "step_elapsed": round(time.perf_counter() - t0, 2),
                           "message": f"OpenAI API 오류: {ex}"})
            yield _ndjson({"event": "complete", "elapsed": total, "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)
        total = round(time.perf_counter() - t_start, 2)
        yield _ndjson({"event": "step_done", "step": "image_gen",
                       "elapsed": total, "step_elapsed": step_elapsed,
                       "data": {
                           "layout_json": step3_json,
                           "layout_raw": step3_raw,
                           "parsed": isinstance(step3_json, dict),
                       }})
        yield _ndjson({"event": "complete", "elapsed": total, "status": "ok"})

    return _stream_response(gen())


@app.post("/api/step/image_gen_ocr")
async def api_step_image_gen_ocr(request: Request) -> StreamingResponse:
    form = await _parse_big_form(request)
    image_b64 = str(form.get("image_b64") or "")
    if not image_b64:
        raise HTTPException(status_code=400, detail="image_b64 가 비어 있습니다.")

    async def gen() -> AsyncIterator[bytes]:
        t_start = time.perf_counter()
        yield _ndjson({
            "event": "step_start", "step": "image_gen_ocr",
            "label": "3단계 결과 이미지 OCR (Document AI)",
            "elapsed": 0.0,
        })
        t0 = time.perf_counter()
        try:
            out = await asyncio.to_thread(_ocr_with_positions_sync, base64.b64decode(image_b64))
        except Exception as ex:
            total = round(time.perf_counter() - t_start, 2)
            yield _ndjson({"event": "step_error", "step": "image_gen_ocr",
                           "elapsed": total, "step_elapsed": round(time.perf_counter() - t0, 2),
                           "message": str(ex)})
            yield _ndjson({"event": "complete", "elapsed": total, "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)
        total = round(time.perf_counter() - t_start, 2)
        yield _ndjson({"event": "step_done", "step": "image_gen_ocr",
                       "elapsed": total, "step_elapsed": step_elapsed,
                       "data": {
                           "ocr_text": out["full_text"],
                           "ocr_chars": len(out["full_text"] or ""),
                           "lines": out["lines"],
                           "line_count": len(out["lines"]),
                       }})
        yield _ndjson({"event": "complete", "elapsed": total, "status": "ok"})

    return _stream_response(gen())


@app.post("/api/step/image_gen2")
async def api_step_image_gen2(request: Request) -> StreamingResponse:
    """4단계 (image_gen2) 재실행. 원본 제품 사진 + image_prompt + 모델/품질 선택값으로 한 장 생성."""
    form = await _parse_big_form(request)
    file = form.get("file")
    if file is None or not hasattr(file, "read"):
        raise HTTPException(status_code=400, detail="제품 사진(file) 누락")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="제품 사진이 비어 있습니다.")
    filename = getattr(file, "filename", None) or "upload.png"
    image_prompt = str(form.get("image_prompt") or "")
    if not image_prompt:
        raise HTTPException(status_code=400, detail="image_prompt 가 비어 있습니다.")
    try:
        width_px = int(form.get("width_px") or 860)
        height_px = int(form.get("height_px") or 2000)
    except (ValueError, TypeError):
        width_px, height_px = 860, 2000
    model_arg = (str(form.get("model") or "")).strip()
    quality_arg = (str(form.get("quality") or "")).strip()

    async def gen() -> AsyncIterator[bytes]:
        t_start = time.perf_counter()
        api_key = _openai_api_key()
        if not api_key:
            yield _ndjson({"event": "step_error", "step": "image_gen2",
                           "elapsed": 0.0,
                           "message": "OPENAI_API_KEY 가 config.py 또는 환경 변수에 없습니다."})
            yield _ndjson({"event": "complete", "elapsed": 0.0, "status": "error"})
            return
        gen_size = choose_api_size(width_px, height_px)
        use_model = model_arg or OPENAI_IMAGE_MODEL
        use_quality = quality_arg or "low"
        yield _ndjson({
            "event": "step_start", "step": "image_gen2",
            "label": f"상세페이지 이미지 생성 ({use_model}, {gen_size}, {use_quality})",
            "elapsed": 0.0,
            "data": {
                "model": use_model,
                "size": gen_size,
                "quality": use_quality,
                "available_options": IMAGE_GEN2_OPTIONS,
                "image_prompt": image_prompt,
                "width_px": width_px,
                "height_px": height_px,
            },
        })
        t0 = time.perf_counter()
        try:
            img_out = await asyncio.to_thread(
                _images_generate_sync, api_key, image_prompt, gen_size,
                raw, filename, use_model, use_quality,
            )
        except Exception as ex:
            total = round(time.perf_counter() - t_start, 2)
            yield _ndjson({"event": "step_error", "step": "image_gen2",
                           "elapsed": total, "step_elapsed": round(time.perf_counter() - t0, 2),
                           "message": str(ex)})
            yield _ndjson({"event": "complete", "elapsed": total, "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)
        total = round(time.perf_counter() - t_start, 2)
        yield _ndjson({
            "event": "step_done", "step": "image_gen2",
            "elapsed": total, "step_elapsed": step_elapsed,
            "data": {
                "first_image_b64": img_out["b64_json"],
                "first_image_width": int(img_out["output_width"]),
                "first_image_height": int(img_out["output_height"]),
            },
        })
        yield _ndjson({"event": "complete", "elapsed": total, "status": "ok"})

    return _stream_response(gen())


@app.post("/api/step/image_gen_text")
async def api_step_image_gen_text(request: Request) -> StreamingResponse:
    form = await _parse_big_form(request)
    image_b64 = str(form.get("image_b64") or "")
    if not image_b64:
        raise HTTPException(status_code=400, detail="image_b64 가 비어 있습니다.")
    first_json = str(form.get("first_json") or "{}")  # noqa: F841 (예비 보관)
    try:
        width_px = int(form.get("width_px") or 860)
        height_px = int(form.get("height_px") or 2000)
    except (ValueError, TypeError):
        width_px, height_px = 860, 2000
    prompt = str(form.get("prompt") or "")
    model = str(form.get("model") or "")
    quality_arg = str(form.get("quality") or "")
    # 페이지 N (2 이상) 재실행 지원: step_suffix="__pN" 이 오면 응답 step 이름에 붙임
    step_suffix = str(form.get("step_suffix") or "")
    try:
        page_num_arg: int | None = int(form.get("page") or 0) or None
    except (ValueError, TypeError):
        page_num_arg = None
    step_name = "image_gen_text" + step_suffix
    page_kw = {"page": page_num_arg} if page_num_arg else {}
    # 마스크용 사각형 좌표 (JSON). 없으면 마스크 없이 호출.
    mask_rects_json = str(form.get("mask_rects_json") or "")
    mask_rects: list[dict[str, Any]] = []
    if mask_rects_json:
        try:
            parsed_rects = json.loads(mask_rects_json)
            if isinstance(parsed_rects, list):
                mask_rects = [r for r in parsed_rects if isinstance(r, dict)]
        except json.JSONDecodeError:
            mask_rects = []
    # 6단계 글자 제거는 원본 제품 사진을 첨부하지 않는다 (마스크로 영역만 지정)

    async def gen() -> AsyncIterator[bytes]:
        t_start = time.perf_counter()
        api_key = _openai_api_key()
        if not api_key:
            yield _ndjson({"event": "step_error", "step": step_name,
                           "elapsed": 0.0,
                           "message": "OPENAI_API_KEY 가 config.py 또는 환경 변수에 없습니다.",
                           **page_kw})
            yield _ndjson({"event": "complete", "elapsed": 0.0, "status": "error",
                           **page_kw})
            return
        gen_size = choose_api_size(width_px, height_px)
        text_prompt = (prompt or "").strip() or STEP6_TEXT_REMOVE_PROMPT
        use_model = (model or "").strip() or OPENAI_IMAGE_MODEL
        use_quality = (quality_arg or "").strip() or "medium"
        # 마스크 빌드: step3 이미지의 실제 해상도로 만들어야 함
        step3_bytes = base64.b64decode(image_b64)
        try:
            with Image.open(BytesIO(step3_bytes)) as _im:
                src_w, src_h = _im.size
        except Exception:
            src_w, src_h = width_px, height_px
        coarse_png_bytes: bytes | None = None
        coarse_file_path: str | None = None
        refined_png_bytes: bytes | None = None
        refined_file_path: str | None = None
        refine_info: dict[str, Any] = {}
        if mask_rects:
            try:
                _expanded_rects = _expand_rects_with_margin(mask_rects, 5, src_w, src_h)
                coarse_png_bytes = _build_text_mask_png(src_w, src_h, _expanded_rects)
                mask_label = f"image_gen_text_rerun{step_suffix}_coarse" if step_suffix \
                    else "image_gen_text_rerun_coarse"
                coarse_file_path = _save_mask_to_session(
                    coarse_png_bytes, label=mask_label, page=page_num_arg
                )
            except Exception as ex:
                print(f"[image_gen_text rerun{step_suffix}] 거친 마스크 빌드 실패: {ex}")
                coarse_png_bytes = None
            if coarse_png_bytes:
                refine_label = f"image_gen_text_rerun{step_suffix}" if step_suffix \
                    else "image_gen_text_rerun"
                refined_png_bytes, refined_file_path, refine_info = _refine_mask_with_logging(
                    coarse_png_bytes, step3_bytes,
                    label=refine_label, page=page_num_arg,
                )
        # API 에 전달할 마스크는 coarse (사각) 마스크
        mask_png_bytes = coarse_png_bytes
        # API 입력 이미지: refined mask 영역을 주변 색으로 메꾼 step3 이미지
        api_input_bytes_rerun = step3_bytes
        inpainted_file_path: str | None = None
        if refined_png_bytes:
            inp_label = f"image_gen_text_rerun{step_suffix}" if step_suffix \
                else "image_gen_text_rerun"
            api_input_bytes_rerun, inpainted_file_path = await asyncio.to_thread(
                _inpaint_and_save, step3_bytes, refined_png_bytes,
                inp_label, page_num_arg,
            )
        yield _ndjson({
            "event": "step_start", "step": step_name,
            "label": f"글자를 제거한 이미지 생성 ({use_model}, {gen_size}, {use_quality})",
            "elapsed": 0.0,
            "data": {
                "model": use_model,
                "quality": use_quality,
                "available_options": IMAGE_GEN_TEXT_OPTIONS,
                "size": gen_size,
                "text_prompt": text_prompt,
                "mask_rect_count": len(mask_rects),
                "mask_rects": mask_rects,
                "mask_file": coarse_file_path,
                "refined_mask_file": refined_file_path,
                "refine_info": refine_info,
                "inpainted_image_file": inpainted_file_path,
                "inpainted_image_b64": base64.b64encode(api_input_bytes_rerun).decode("ascii"),
                "base_image_b64": image_b64,
                "base_image_width": src_w,
                "base_image_height": src_h,
            },
            **page_kw,
        })
        t0 = time.perf_counter()
        try:
            print("-" * 60)
            print(
                f"[image_gen_text] (rerun{step_suffix}) model={use_model}, "
                f"quality={use_quality}, size={gen_size}, "
                f"input_images=1 (primary=step3_inpainted.png), "
                f"mask_rects={len(mask_rects)}"
            )
            print(f"[image_gen_text] prompt ({len(text_prompt)} chars):")
            print(text_prompt)
            print("-" * 60)
            cp = await asyncio.to_thread(
                _crop_paste_api_call,
                api_key, text_prompt,
                api_input_bytes_rerun, mask_png_bytes,
                use_model, use_quality,
            )
        except Exception as ex:
            total = round(time.perf_counter() - t_start, 2)
            yield _ndjson({"event": "step_error", "step": step_name,
                           "elapsed": total, "step_elapsed": round(time.perf_counter() - t0, 2),
                           "message": str(ex), **page_kw})
            yield _ndjson({"event": "complete", "elapsed": total, "status": "error",
                           **page_kw})
            return
        final_bytes = cp["final_bytes"]
        final_b64 = base64.b64encode(final_bytes).decode("ascii")
        final_w = int(cp["final_width"])
        final_h = int(cp["final_height"])
        # [DEBUG] 5개 이미지 파일로 저장
        try:
            _dbg_dir = _session_dir_var.get() or (_LOG_ROOT / "ad_hoc")
            _dbg_dir.mkdir(parents=True, exist_ok=True)
            _dbg_stamp = time.strftime("%H%M%S") + "_" + uuid.uuid4().hex[:4]
            def _save(name: str, data: bytes | None) -> None:
                if not data:
                    return
                p = _dbg_dir / f"{step_name}__{name}__{_dbg_stamp}.png"
                p.write_bytes(data)
                print(f"[{step_name}][DEBUG] {name}: {p}")
            _save("1_api_input", api_input_bytes_rerun)
            _save("2_mask", mask_png_bytes)
            _save("3_cropped_sent", cp["cropped_input_bytes"])
            _save("4_api_response", cp["api_response_bytes"])
            _save("5_final", final_bytes)
            print(f"[{step_name}][DEBUG] bbox={cp['bbox']}")
        except Exception as _dbg_ex:
            print(f"[{step_name}][DEBUG] 디버그 저장 실패: {_dbg_ex}")
        stage_images_rerun = {
            "api_input_b64": base64.b64encode(api_input_bytes_rerun).decode("ascii"),
            "mask_b64": base64.b64encode(cp["cropped_mask_bytes"]).decode("ascii") if cp.get("cropped_mask_bytes") else (base64.b64encode(mask_png_bytes).decode("ascii") if mask_png_bytes else None),
            "cropped_sent_b64": base64.b64encode(cp["cropped_input_bytes"]).decode("ascii") if cp["cropped_input_bytes"] else None,
            "api_response_b64": base64.b64encode(cp["api_response_bytes"]).decode("ascii") if cp["api_response_bytes"] else None,
            "final_b64": final_b64,
            "bbox": list(cp["bbox"]) if cp["bbox"] else None,
        }
        step_elapsed = round(time.perf_counter() - t0, 2)
        total = round(time.perf_counter() - t_start, 2)
        yield _ndjson({"event": "step_done", "step": step_name,
                       "elapsed": total, "step_elapsed": step_elapsed,
                       "data": {
                           "image_b64": final_b64,
                           "image_width": final_w,
                           "image_height": final_h,
                           "stage_images": stage_images_rerun,
                       },
                       **page_kw})
        yield _ndjson({"event": "complete", "elapsed": total, "status": "ok",
                       **page_kw})

    return _stream_response(gen())


@app.post("/api/step/cleaned_ocr")
async def api_step_cleaned_ocr(request: Request) -> StreamingResponse:
    form = await _parse_big_form(request)
    image_b64 = str(form.get("image_b64") or "")
    if not image_b64:
        raise HTTPException(status_code=400, detail="image_b64 가 비어 있습니다.")

    async def gen() -> AsyncIterator[bytes]:
        t_start = time.perf_counter()
        yield _ndjson({
            "event": "step_start", "step": "cleaned_ocr",
            "label": "글자 제거 이미지 OCR (Document AI, 라인+좌표)",
            "elapsed": 0.0,
        })
        t0 = time.perf_counter()
        try:
            out = await asyncio.to_thread(_ocr_with_positions_sync, base64.b64decode(image_b64))
        except Exception as ex:
            total = round(time.perf_counter() - t_start, 2)
            yield _ndjson({"event": "step_error", "step": "cleaned_ocr",
                           "elapsed": total, "step_elapsed": round(time.perf_counter() - t0, 2),
                           "message": str(ex)})
            yield _ndjson({"event": "complete", "elapsed": total, "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)
        total = round(time.perf_counter() - t_start, 2)
        yield _ndjson({"event": "step_done", "step": "cleaned_ocr",
                       "elapsed": total, "step_elapsed": step_elapsed,
                       "data": {
                           "lines": out["lines"],
                           "line_count": len(out["lines"]),
                           "full_text": out["full_text"],
                           "image_width": out["image_width"],
                           "image_height": out["image_height"],
                       }})
        yield _ndjson({"event": "complete", "elapsed": total, "status": "ok"})

    return _stream_response(gen())


@app.post("/api/step/text_area_diff")
async def api_step_text_area_diff(request: Request) -> StreamingResponse:
    form = await _parse_big_form(request)
    before_json = str(form.get("before_json") or "[]")
    after_json = str(form.get("after_json") or "[]")
    try:
        iou_threshold = float(form.get("iou_threshold") or 0.3)
    except (ValueError, TypeError):
        iou_threshold = 0.3
    try:
        lines_a = json.loads(before_json)
        lines_b = json.loads(after_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"lines JSON 파싱 오류: {e}") from e

    async def gen() -> AsyncIterator[bytes]:
        t_start = time.perf_counter()
        yield _ndjson({
            "event": "step_start", "step": "text_area_diff",
            "label": "OCR 영역 차이 (원본에는 있고 글자 제거 이미지에는 없는 영역)",
            "elapsed": 0.0,
            "data": {
                "before_count": len(lines_a) if isinstance(lines_a, list) else 0,
                "after_count": len(lines_b) if isinstance(lines_b, list) else 0,
                "iou_threshold": iou_threshold,
            },
        })
        t0 = time.perf_counter()
        try:
            removed = await asyncio.to_thread(
                _text_area_diff_sync,
                lines_a if isinstance(lines_a, list) else [],
                lines_b if isinstance(lines_b, list) else [],
                iou_threshold,
            )
        except Exception as ex:
            total = round(time.perf_counter() - t_start, 2)
            yield _ndjson({"event": "step_error", "step": "text_area_diff",
                           "elapsed": total, "step_elapsed": round(time.perf_counter() - t0, 2),
                           "message": str(ex)})
            yield _ndjson({"event": "complete", "elapsed": total, "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)
        total = round(time.perf_counter() - t_start, 2)
        yield _ndjson({"event": "step_done", "step": "text_area_diff",
                       "elapsed": total, "step_elapsed": step_elapsed,
                       "data": {
                           "lines": removed,
                           "line_count": len(removed),
                           "iou_threshold": iou_threshold,
                       }})
        yield _ndjson({"event": "complete", "elapsed": total, "status": "ok"})

    return _stream_response(gen())


@app.post("/api/step/removed_text_overlay")
async def api_step_removed_text_overlay(request: Request) -> StreamingResponse:
    form = await _parse_big_form(request)
    base_image_b64 = str(form.get("base_image_b64") or "")
    lines_json = str(form.get("lines_json") or "[]")
    if not base_image_b64:
        raise HTTPException(status_code=400, detail="base_image_b64 가 비어 있습니다.")
    try:
        lines = json.loads(lines_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"lines_json 파싱 오류: {e}") from e

    async def gen() -> AsyncIterator[bytes]:
        t_start = time.perf_counter()
        try:
            with Image.open(BytesIO(base64.b64decode(base_image_b64))) as im:
                bw, bh = im.size
        except Exception:
            bw, bh = None, None
        yield _ndjson({
            "event": "step_start", "step": "removed_text_overlay",
            "label": "글자 제거 이미지에 (지워진 영역) 글자 합성 (canvas)",
            "elapsed": 0.0,
            "data": {
                "base_image_ref": "image_gen_text",
                "lines_ref": "text_area_diff",
                "line_count": len(lines) if isinstance(lines, list) else 0,
            },
        })
        total = round(time.perf_counter() - t_start, 2)
        yield _ndjson({"event": "step_done", "step": "removed_text_overlay",
                       "elapsed": total, "step_elapsed": 0.0,
                       "data": {
                           "base_image_b64": base_image_b64,
                           "base_image_width": bw,
                           "base_image_height": bh,
                           "lines": lines if isinstance(lines, list) else [],
                           "line_count": len(lines) if isinstance(lines, list) else 0,
                       }})
        yield _ndjson({"event": "complete", "elapsed": total, "status": "ok"})

    return _stream_response(gen())


@app.post("/api/step/overlay_refine")
async def api_step_overlay_refine(request: Request) -> StreamingResponse:
    """7단계 (합성 정보 다듬기) 재실행. base_image_b64 + lines_json + (width/height) 을 받아
    GPT 비전에 보내 다듬어진 lines 를 반환한다. step_suffix 가 있으면 응답의 step 이름에
    그대로 붙여 추가 페이지(__pN) 도 같은 페이지 그룹 UI 에 갱신된다."""
    form = await _parse_big_form(request)
    base_image_b64 = str(form.get("base_image_b64") or "")
    lines_json = str(form.get("lines_json") or "[]")
    step_suffix = str(form.get("step_suffix") or "")
    chosen_model = (str(form.get("model") or "").strip()
                    or OPENAI_VISION_MODEL)
    chosen_effort = (str(form.get("reasoning_effort") or "").strip()
                     or OVERLAY_REFINE_DEFAULT_EFFORT)
    try:
        page_num = int(form.get("page") or 0) or None
    except (ValueError, TypeError):
        page_num = None
    try:
        width = int(form.get("width") or 0) or None
    except (ValueError, TypeError):
        width = None
    try:
        height = int(form.get("height") or 0) or None
    except (ValueError, TypeError):
        height = None
    if not base_image_b64:
        raise HTTPException(status_code=400, detail="base_image_b64 가 비어 있습니다.")
    try:
        lines = json.loads(lines_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"lines_json 파싱 오류: {e}") from e
    if not isinstance(lines, list):
        raise HTTPException(status_code=400, detail="lines 는 배열이어야 합니다.")
    step_name = f"overlay_refine{step_suffix}"

    async def gen() -> AsyncIterator[bytes]:
        t_start = time.perf_counter()
        api_key = _openai_api_key()
        if not api_key:
            yield _ndjson({
                "event": "step_error", "step": step_name,
                "elapsed": 0.0,
                "message": "OPENAI_API_KEY 가 config.py 또는 환경 변수에 없습니다.",
                **({"page": page_num} if page_num else {}),
            })
            yield _ndjson({"event": "complete", "elapsed": 0.0, "status": "error"})
            return
        yield _ndjson({
            "event": "step_start", "step": step_name,
            "label": f"합성 정보 다듬기 ({chosen_model}, reasoning: {chosen_effort}) — 재실행",
            "elapsed": 0.0,
            "data": {
                "model": chosen_model,
                "reasoning_effort": chosen_effort,
                "available_options": OVERLAY_REFINE_OPTIONS,
                "input_line_count": len(lines),
                "input_lines": lines,
            },
            **({"page": page_num} if page_num else {}),
        })
        t0 = time.perf_counter()
        try:
            refined, prompt, raw = await asyncio.to_thread(
                _refine_overlay_layout_sync,
                api_key, chosen_model, base_image_b64, lines, width, height,
                chosen_effort,
            )
        except Exception as ex:
            total = round(time.perf_counter() - t_start, 2)
            yield _ndjson({
                "event": "step_error", "step": step_name,
                "elapsed": total,
                "step_elapsed": round(time.perf_counter() - t0, 2),
                "message": str(ex),
                **({"page": page_num} if page_num else {}),
            })
            yield _ndjson({"event": "complete", "elapsed": total, "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)
        total = round(time.perf_counter() - t_start, 2)
        yield _ndjson({
            "event": "step_done", "step": step_name,
            "elapsed": total, "step_elapsed": step_elapsed,
            "data": {
                "lines": refined,
                "line_count": len(refined),
                "input_lines": lines,
                "base_image_b64": base_image_b64,
                "base_image_width": width,
                "base_image_height": height,
                "refine_prompt": prompt,
                "refine_raw": raw,
            },
            **({"page": page_num} if page_num else {}),
        })
        yield _ndjson({"event": "complete", "elapsed": total, "status": "ok"})

    return _stream_response(gen())


@app.post("/api/next-page")
async def api_next_page(request: Request) -> StreamingResponse:
    """추가 페이지 한 장에 대해 4~7단계(상세이미지 생성 → OCR → 글자 제거 → 글자 합성)를 수행."""
    form = await _parse_big_form(request)
    file = form.get("file")
    image_prompt_json = str(form.get("image_prompt_json") or "")
    try:
        page_index = int(form.get("page_index") or 1)
    except (ValueError, TypeError):
        page_index = 1
    if file is None or not hasattr(file, "read"):
        raise HTTPException(status_code=400, detail="제품 사진(file) 누락")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="제품 사진이 비어 있습니다.")
    filename = getattr(file, "filename", None) or "upload.png"
    if not image_prompt_json:
        raise HTTPException(status_code=400, detail="image_prompt_json 누락")
    try:
        first = json.loads(image_prompt_json)
        if not isinstance(first, dict):
            raise ValueError("image_prompt 가 객체(dict)가 아닙니다.")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"image_prompt_json 파싱 오류: {e}") from e

    # 4단계 image_gen2 의 모델/품질 선택값 (없으면 기본값)
    image_gen2_model_arg = (str(form.get("image_gen2_model") or "")).strip()
    image_gen2_quality_arg = (str(form.get("image_gen2_quality") or "")).strip()

    sx = f"__p{page_index}"

    async def gen() -> AsyncIterator[bytes]:
        t_start = time.perf_counter()

        def elapsed_total() -> float:
            return round(time.perf_counter() - t_start, 2)

        def _complete(status: str) -> bytes:
            return _ndjson({
                "event": "complete", "elapsed": elapsed_total(),
                "status": status, "page": page_index,
            })

        api_key = _openai_api_key()
        if not api_key:
            yield _ndjson({
                "event": "step_error", "step": f"image_gen2{sx}",
                "elapsed": elapsed_total(),
                "message": "OPENAI_API_KEY 가 config.py 또는 환경 변수에 없습니다.",
                "page": page_index,
            })
            yield _complete("error")
            return

        # --- 내부: 레이아웃 JSON (image_gen 단계와 동일 로직) ---
        image_prompt_full = json.dumps(first, ensure_ascii=False, indent=2)
        step3_user_text = f"{STEP3_TEXT_LAYOUT_PROMPT}\n\n{image_prompt_full}"
        try:
            _step3_raw, step3_json = await asyncio.to_thread(
                _openai_chat_json_sync, api_key, OPENAI_TEXT_MODEL, step3_user_text
            )
        except Exception as ex:
            yield _ndjson({
                "event": "step_error", "step": f"image_gen2{sx}",
                "elapsed": elapsed_total(),
                "message": f"레이아웃 JSON 생성 오류: {ex}",
                "page": page_index,
            })
            yield _complete("error")
            return
        layout_labels = _extract_layout_labels(step3_json)
        layout_texts = [
            lab.get("text", "").strip() for lab in layout_labels
            if isinstance(lab, dict) and (lab.get("text") or "").strip()
        ]

        # --- 4단계: 상세페이지 이미지 생성 (image_gen2) ---
        texts_block = "\n".join(f"- {t}" for t in layout_texts)
        image2_prompt_en = (first.get("image2_prompt_en") or "").strip()
        image_gen2_prompt = f"{STEP3_5_IMAGE_GEN_PROMPT}\n{texts_block}".rstrip()
        if image2_prompt_en:
            image_gen2_prompt += f"\n\n{image2_prompt_en}"
        wpx = int(first.get("width_px") or 860)
        hpx = int(first.get("height_px") or 2000)
        gen_size = choose_api_size(wpx, hpx)
        image_gen2_model_use = image_gen2_model_arg or OPENAI_IMAGE_MODEL
        image_gen2_quality = image_gen2_quality_arg or "low"
        yield _ndjson({
            "event": "step_start", "step": f"image_gen2{sx}",
            "label": f"페이지 {page_index} · 4단계 — 상세페이지 이미지 생성 ({image_gen2_model_use}, {gen_size}, {image_gen2_quality})",
            "elapsed": elapsed_total(),
            "page": page_index,
            "data": {
                "model": image_gen2_model_use, "size": gen_size,
                "quality": image_gen2_quality,
                "available_options": IMAGE_GEN2_OPTIONS,
                "layout_texts": layout_texts,
                "image_prompt": image_gen2_prompt,
                "width_px": wpx,
                "height_px": hpx,
            },
        })
        t0 = time.perf_counter()
        try:
            img_out = await asyncio.to_thread(
                _images_generate_sync, api_key, image_gen2_prompt, gen_size,
                raw, filename, image_gen2_model_use, image_gen2_quality,
            )
            first_image_b64 = img_out["b64_json"]
            first_image_width = int(img_out["output_width"])
            first_image_height = int(img_out["output_height"])
        except Exception as ex:
            yield _ndjson({
                "event": "step_error", "step": f"image_gen2{sx}",
                "elapsed": elapsed_total(),
                "step_elapsed": round(time.perf_counter() - t0, 2),
                "message": str(ex), "page": page_index,
            })
            yield _complete("error")
            return
        yield _ndjson({
            "event": "step_done", "step": f"image_gen2{sx}",
            "elapsed": elapsed_total(),
            "step_elapsed": round(time.perf_counter() - t0, 2),
            "page": page_index,
            "data": {
                "first_image_b64": first_image_b64,
                "first_image_width": first_image_width,
                "first_image_height": first_image_height,
            },
        })

        # --- 5단계: 상세페이지 이미지 OCR (image_gen_ocr) ---
        yield _ndjson({
            "event": "step_start", "step": f"image_gen_ocr{sx}",
            "label": f"페이지 {page_index} · 5단계 — 상세페이지 이미지 OCR (Document AI)",
            "elapsed": elapsed_total(), "page": page_index,
        })
        t0 = time.perf_counter()
        try:
            ocr_step3 = await asyncio.to_thread(
                _ocr_with_positions_sync, base64.b64decode(first_image_b64)
            )
            step3_ocr_lines = ocr_step3["lines"]
            step3_ocr_text = ocr_step3["full_text"]
        except Exception as ex:
            yield _ndjson({
                "event": "step_error", "step": f"image_gen_ocr{sx}",
                "elapsed": elapsed_total(),
                "step_elapsed": round(time.perf_counter() - t0, 2),
                "message": str(ex), "page": page_index,
            })
            yield _complete("error")
            return
        yield _ndjson({
            "event": "step_done", "step": f"image_gen_ocr{sx}",
            "elapsed": elapsed_total(),
            "step_elapsed": round(time.perf_counter() - t0, 2),
            "page": page_index,
            "data": {
                "ocr_text": step3_ocr_text,
                "ocr_chars": len(step3_ocr_text or ""),
                "lines": step3_ocr_lines,
                "line_count": len(step3_ocr_lines),
            },
        })

        # --- 6단계: 글자 제거 이미지 재생성 (image_gen_text) ---
        matched_lines = _match_ocr_to_layout_texts(step3_ocr_lines, layout_texts)
        matched_ocr_lines = _expand_matched_to_ocr_lines(matched_lines)
        matched_lines_compact = [
            {
                "text": ln.get("text", ""),
                "x": ln.get("x", 0),
                "y": ln.get("y", 0),
                "width": ln.get("width", 0),
                "height": ln.get("height", 0),
            }
            for ln in matched_ocr_lines
            if isinstance(ln, dict)
        ]
        text_prompt = STEP6_TEXT_REMOVE_PROMPT
        # 지울 영역은 프롬프트가 아니라 마스크 PNG 로 API 에 전달
        coarse_png_bytes: bytes | None = None
        coarse_file_path: str | None = None
        refined_png_bytes: bytes | None = None
        refined_file_path: str | None = None
        refine_info: dict[str, Any] = {}
        if matched_lines_compact:
            try:
                _expanded_rects = _expand_rects_with_margin(
                    matched_lines_compact, 5, first_image_width, first_image_height
                )
                coarse_png_bytes = _build_text_mask_png(
                    first_image_width, first_image_height, _expanded_rects
                )
                coarse_file_path = _save_mask_to_session(
                    coarse_png_bytes,
                    label=f"image_gen_text_p{page_index}_coarse",
                    page=page_index,
                )
            except Exception as ex:
                print(f"[image_gen_text page] 거친 마스크 빌드 실패: {ex}")
                coarse_png_bytes = None
            if coarse_png_bytes:
                refined_png_bytes, refined_file_path, refine_info = _refine_mask_with_logging(
                    coarse_png_bytes, base64.b64decode(first_image_b64),
                    label=f"image_gen_text_p{page_index}", page=page_index,
                )
        # API 에 전달할 마스크는 coarse (사각) 마스크
        mask_png_bytes = coarse_png_bytes
        # API 입력 이미지: refined mask 영역을 주변 색으로 메꾼 4단계 이미지
        step3_bytes_page = base64.b64decode(first_image_b64)
        api_input_bytes_page = step3_bytes_page
        inpainted_file_path: str | None = None
        if refined_png_bytes:
            api_input_bytes_page, inpainted_file_path = await asyncio.to_thread(
                _inpaint_and_save, step3_bytes_page, refined_png_bytes,
                f"image_gen_text_p{page_index}", page_index,
            )
        step4_quality = "medium"
        yield _ndjson({
            "event": "step_start", "step": f"image_gen_text{sx}",
            "label": f"페이지 {page_index} · 6단계 — 글자를 제거한 이미지 생성 ({OPENAI_IMAGE_MODEL}, {gen_size}, {step4_quality})",
            "elapsed": elapsed_total(), "page": page_index,
            "data": {
                "model": OPENAI_IMAGE_MODEL,
                "quality": step4_quality,
                "available_options": IMAGE_GEN_TEXT_OPTIONS,
                "size": gen_size,
                "text_prompt": text_prompt,
                "mask_rect_count": len(matched_lines_compact),
                "mask_rects": matched_lines_compact,
                "mask_file": coarse_file_path,
                "refined_mask_file": refined_file_path,
                "refine_info": refine_info,
                "inpainted_image_file": inpainted_file_path,
                "inpainted_image_b64": base64.b64encode(api_input_bytes_page).decode("ascii"),
                "base_image_b64": first_image_b64,
                "base_image_width": first_image_width,
                "base_image_height": first_image_height,
                "width_px": wpx,
                "height_px": hpx,
            },
        })
        t0 = time.perf_counter()
        stage_images_page: dict[str, Any] = {}
        try:
            cp = await asyncio.to_thread(
                _crop_paste_api_call,
                api_key, text_prompt,
                api_input_bytes_page, mask_png_bytes,
                OPENAI_IMAGE_MODEL, OPENAI_IMAGE_QUALITY,
            )
            final_bytes = cp["final_bytes"]
            second_image_b64 = base64.b64encode(final_bytes).decode("ascii")
            second_image_width = int(cp["final_width"])
            second_image_height = int(cp["final_height"])
            # [DEBUG] 5개 이미지 파일로 저장
            try:
                _dbg_dir = _session_dir_var.get() or (_LOG_ROOT / "ad_hoc")
                _dbg_dir.mkdir(parents=True, exist_ok=True)
                _dbg_stamp = time.strftime("%H%M%S") + "_" + uuid.uuid4().hex[:4]
                def _save(name: str, data: bytes | None) -> None:
                    if not data:
                        return
                    p = _dbg_dir / f"image_gen_text{sx}__{name}__{_dbg_stamp}.png"
                    p.write_bytes(data)
                    print(f"[image_gen_text{sx}][DEBUG] {name}: {p}")
                _save("1_api_input", api_input_bytes_page)
                _save("2_mask", cp.get("cropped_mask_bytes") or mask_png_bytes)
                _save("3_cropped_sent", cp["cropped_input_bytes"])
                _save("4_api_response", cp["api_response_bytes"])
                _save("5_final", final_bytes)
                print(f"[image_gen_text{sx}][DEBUG] bbox={cp['bbox']}")
            except Exception as _dbg_ex:
                print(f"[image_gen_text{sx}][DEBUG] 디버그 저장 실패: {_dbg_ex}")
            stage_images_page = {
                "api_input_b64": base64.b64encode(api_input_bytes_page).decode("ascii"),
                "mask_b64": base64.b64encode(cp["cropped_mask_bytes"]).decode("ascii") if cp.get("cropped_mask_bytes") else (base64.b64encode(mask_png_bytes).decode("ascii") if mask_png_bytes else None),
                "cropped_sent_b64": base64.b64encode(cp["cropped_input_bytes"]).decode("ascii") if cp["cropped_input_bytes"] else None,
                "api_response_b64": base64.b64encode(cp["api_response_bytes"]).decode("ascii") if cp["api_response_bytes"] else None,
                "final_b64": second_image_b64,
                "bbox": list(cp["bbox"]) if cp["bbox"] else None,
            }
        except Exception as ex:
            yield _ndjson({
                "event": "step_error", "step": f"image_gen_text{sx}",
                "elapsed": elapsed_total(),
                "step_elapsed": round(time.perf_counter() - t0, 2),
                "message": str(ex), "page": page_index,
            })
            yield _complete("error")
            return
        # 1차 합성 정보 (5단계 OCR 좌표 + 4단계 이미지 픽셀 색상)
        # (step_done 직전에 계산하여 6단계 결과에 "다듬기 전 정보 preview" 를 포함)
        sampled_lines = await asyncio.to_thread(
            _apply_pixel_styles_to_lines,
            base64.b64decode(first_image_b64),
            matched_ocr_lines,
        )
        overlay_labels = []
        for ln in sampled_lines:
            if not isinstance(ln, dict):
                continue
            cleaned = _strip_leading_bullet(ln.get("text"))
            if not (isinstance(cleaned, str) and cleaned.strip()):
                continue
            overlay_labels.append({**ln, "align": "center", "text": cleaned})

        yield _ndjson({
            "event": "step_done", "step": f"image_gen_text{sx}",
            "elapsed": elapsed_total(),
            "step_elapsed": round(time.perf_counter() - t0, 2),
            "page": page_index,
            "data": {
                "image_b64": second_image_b64,
                "image_width": second_image_width,
                "image_height": second_image_height,
                "input_lines": overlay_labels,
                "base_image_b64": second_image_b64,
                "base_image_width": second_image_width,
                "base_image_height": second_image_height,
                "stage_images": stage_images_page,
            },
        })

        # --- 7단계: 합성 정보 다듬기 (GPT 비전) ---
        refined_labels = overlay_labels
        refine_raw = ""
        refine_prompt = ""
        yield _ndjson({
            "event": "step_start", "step": f"overlay_refine{sx}",
            "label": f"페이지 {page_index} · 7단계 — 합성 정보 다듬기 ({OPENAI_VISION_MODEL})",
            "elapsed": elapsed_total(), "page": page_index,
            "data": {
                "model": OPENAI_VISION_MODEL,
                "reasoning_effort": OVERLAY_REFINE_DEFAULT_EFFORT,
                "available_options": OVERLAY_REFINE_OPTIONS,
                "input_line_count": len(overlay_labels),
                "input_lines": overlay_labels,
            },
        })
        t0 = time.perf_counter()
        try:
            refined_labels, refine_prompt, refine_raw = await asyncio.to_thread(
                _refine_overlay_layout_sync,
                api_key, OPENAI_VISION_MODEL, second_image_b64, overlay_labels,
                second_image_width, second_image_height,
                OVERLAY_REFINE_DEFAULT_EFFORT,
            )
        except Exception as ex:
            yield _ndjson({
                "event": "step_error", "step": f"overlay_refine{sx}",
                "elapsed": elapsed_total(),
                "step_elapsed": round(time.perf_counter() - t0, 2),
                "message": str(ex), "page": page_index,
            })
            refined_labels = overlay_labels
        step_elapsed = round(time.perf_counter() - t0, 2)
        yield _ndjson({
            "event": "step_done", "step": f"overlay_refine{sx}",
            "elapsed": elapsed_total(), "step_elapsed": step_elapsed,
            "page": page_index,
            "data": {
                "lines": refined_labels,
                "line_count": len(refined_labels),
                "input_lines": overlay_labels,
                "base_image_b64": second_image_b64,
                "base_image_width": second_image_width,
                "base_image_height": second_image_height,
                "refine_prompt": refine_prompt,
                "refine_raw": refine_raw,
            },
        })

        # --- 8단계: 다듬어진 합성 정보로 글자 합성 (canvas) ---
        yield _ndjson({
            "event": "step_start", "step": f"removed_text_overlay{sx}",
            "label": f"페이지 {page_index} · 8단계 — 글자 제거 이미지에 글자 합성 (canvas)",
            "elapsed": elapsed_total(), "page": page_index,
            "data": {"line_count": len(refined_labels)},
        })
        yield _ndjson({
            "event": "step_done", "step": f"removed_text_overlay{sx}",
            "elapsed": elapsed_total(), "step_elapsed": 0.0,
            "page": page_index,
            "data": {
                "base_image_b64": second_image_b64,
                "base_image_width": second_image_width,
                "base_image_height": second_image_height,
                "lines": refined_labels,
                "line_count": len(refined_labels),
            },
        })
        yield _complete("ok")

    return _stream_response(gen())


@app.post("/api/step/image_diff")
async def api_step_image_diff(request: Request) -> StreamingResponse:
    form = await _parse_big_form(request)
    image_a_b64 = str(form.get("image_a_b64") or "")
    image_b_b64 = str(form.get("image_b_b64") or "")
    if not image_a_b64 or not image_b_b64:
        raise HTTPException(status_code=400, detail="비교할 이미지 두 개 모두 필요합니다.")

    async def gen() -> AsyncIterator[bytes]:
        t_start = time.perf_counter()
        yield _ndjson({
            "event": "step_start", "step": "image_diff",
            "label": "픽셀 단위 차이 이미지 (3단계 vs 4단계)",
            "elapsed": 0.0,
            "data": {},
        })
        t0 = time.perf_counter()
        try:
            diff_out = await asyncio.to_thread(
                _diff_images_sync, image_a_b64, image_b_b64
            )
        except Exception as ex:
            total = round(time.perf_counter() - t_start, 2)
            yield _ndjson({"event": "step_error", "step": "image_diff",
                           "elapsed": total, "step_elapsed": round(time.perf_counter() - t0, 2),
                           "message": str(ex)})
            yield _ndjson({"event": "complete", "elapsed": total, "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)
        total = round(time.perf_counter() - t_start, 2)
        yield _ndjson({"event": "step_done", "step": "image_diff",
                       "elapsed": total, "step_elapsed": step_elapsed,
                       "data": {
                           "image_b64": diff_out["b64_json"],
                           "image_width": diff_out["output_width"],
                           "image_height": diff_out["output_height"],
                           "diff_pixels": diff_out["diff_pixels"],
                           "bbox": diff_out["bbox"],
                       }})
        yield _ndjson({"event": "complete", "elapsed": total, "status": "ok"})

    return _stream_response(gen())


@app.post("/api/step/diff_refine")
async def api_step_diff_refine(request: Request) -> StreamingResponse:
    form = await _parse_big_form(request)
    image_b64 = str(form.get("image_b64") or "")
    if not image_b64:
        raise HTTPException(status_code=400, detail="image_b64 가 비어 있습니다.")
    try:
        threshold = int(form.get("threshold") or 60)
    except (ValueError, TypeError):
        threshold = 60

    async def gen() -> AsyncIterator[bytes]:
        t_start = time.perf_counter()
        yield _ndjson({
            "event": "step_start", "step": "diff_refine",
            "label": f"diff 정제 (임계값 < {threshold} 인 흐린 픽셀 제거)",
            "elapsed": 0.0,
            "data": {"input_image_ref": "image_diff", "threshold": threshold},
        })
        t0 = time.perf_counter()
        try:
            out = await asyncio.to_thread(_refine_diff_sync, image_b64, threshold)
        except Exception as ex:
            total = round(time.perf_counter() - t_start, 2)
            yield _ndjson({"event": "step_error", "step": "diff_refine",
                           "elapsed": total, "step_elapsed": round(time.perf_counter() - t0, 2),
                           "message": str(ex)})
            yield _ndjson({"event": "complete", "elapsed": total, "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)
        total = round(time.perf_counter() - t_start, 2)
        yield _ndjson({"event": "step_done", "step": "diff_refine",
                       "elapsed": total, "step_elapsed": step_elapsed,
                       "data": {
                           "image_b64": out["b64_json"],
                           "image_width": out["output_width"],
                           "image_height": out["output_height"],
                           "threshold": out["threshold"],
                           "kept_pixels": out["kept_pixels"],
                           "total_pixels": out["total_pixels"],
                           "bbox": out["bbox"],
                       }})
        yield _ndjson({"event": "complete", "elapsed": total, "status": "ok"})

    return _stream_response(gen())


@app.post("/api/step/ocr_diff")
async def api_step_ocr_diff(request: Request) -> StreamingResponse:
    form = await _parse_big_form(request)
    image_b64 = str(form.get("image_b64") or "")
    if not image_b64:
        raise HTTPException(status_code=400, detail="image_b64 가 비어 있습니다.")

    async def gen() -> AsyncIterator[bytes]:
        t_start = time.perf_counter()
        yield _ndjson({
            "event": "step_start", "step": "ocr_diff",
            "label": "정제된 diff 이미지 OCR (Document AI, 라인+좌표)",
            "elapsed": 0.0,
        })
        t0 = time.perf_counter()
        try:
            out = await asyncio.to_thread(_ocr_with_positions_sync, base64.b64decode(image_b64))
        except Exception as ex:
            total = round(time.perf_counter() - t_start, 2)
            yield _ndjson({"event": "step_error", "step": "ocr_diff",
                           "elapsed": total, "step_elapsed": round(time.perf_counter() - t0, 2),
                           "message": str(ex)})
            yield _ndjson({"event": "complete", "elapsed": total, "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)
        total = round(time.perf_counter() - t_start, 2)
        yield _ndjson({"event": "step_done", "step": "ocr_diff",
                       "elapsed": total, "step_elapsed": step_elapsed,
                       "data": {
                           "lines": out["lines"],
                           "line_count": len(out["lines"]),
                           "full_text": out["full_text"],
                           "image_width": out["image_width"],
                           "image_height": out["image_height"],
                       }})
        yield _ndjson({"event": "complete", "elapsed": total, "status": "ok"})

    return _stream_response(gen())


@app.post("/api/step/text_overlay")
async def api_step_text_overlay(request: Request) -> StreamingResponse:
    form = await _parse_big_form(request)
    base_image_b64 = str(form.get("base_image_b64") or "")
    lines_json = str(form.get("lines_json") or "[]")
    if not base_image_b64:
        raise HTTPException(status_code=400, detail="base_image_b64 가 비어 있습니다.")
    try:
        lines = json.loads(lines_json)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"lines_json 파싱 오류: {e}") from e

    async def gen() -> AsyncIterator[bytes]:
        t_start = time.perf_counter()
        try:
            with Image.open(BytesIO(base64.b64decode(base_image_b64))) as im:
                bw, bh = im.size
        except Exception:
            bw, bh = None, None
        yield _ndjson({
            "event": "step_start", "step": "text_overlay",
            "label": "3단계 이미지 위에 6단계 글자 합성 (canvas)",
            "elapsed": 0.0,
            "data": {
                "base_image_ref": "image_gen",
                "lines_ref": "ocr_diff",
                "line_count": len(lines) if isinstance(lines, list) else 0,
            },
        })
        total = round(time.perf_counter() - t_start, 2)
        yield _ndjson({"event": "step_done", "step": "text_overlay",
                       "elapsed": total, "step_elapsed": 0.0,
                       "data": {
                           "base_image_b64": base_image_b64,
                           "base_image_width": bw,
                           "base_image_height": bh,
                           "lines": lines if isinstance(lines, list) else [],
                           "line_count": len(lines) if isinstance(lines, list) else 0,
                       }})
        yield _ndjson({"event": "complete", "elapsed": total, "status": "ok"})

    return _stream_response(gen())


@app.post("/api/step/vision_layout")
async def api_step_vision_layout(request: Request) -> StreamingResponse:
    form = await _parse_big_form(request)
    meta_slice = str(form.get("meta_slice") or "{}")
    image_b64 = str(form.get("image_b64") or "")
    if not image_b64:
        raise HTTPException(status_code=400, detail="image_b64 가 비어 있습니다.")
    try:
        meta_obj = json.loads(meta_slice)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"meta_slice JSON 파싱 오류: {e}") from e

    async def gen() -> AsyncIterator[bytes]:
        t_start = time.perf_counter()
        api_key = _openai_api_key()
        if not api_key:
            yield _ndjson({"event": "step_error", "step": "vision_layout",
                           "elapsed": 0.0,
                           "message": "OPENAI_API_KEY 가 config.py 또는 환경 변수에 없습니다."})
            yield _ndjson({"event": "complete", "elapsed": 0.0, "status": "error"})
            return
        vision_prompt = build_vision_layout_prompt(meta_obj)
        try:
            from PIL import Image as _PIL
            with _PIL.open(BytesIO(base64.b64decode(image_b64))) as _im:
                w_in, h_in = _im.size
        except Exception:
            w_in, h_in = None, None
        yield _ndjson({
            "event": "step_start", "step": "vision_layout",
            "label": f"문구 좌표 추출 ({OPENAI_VISION_MODEL})",
            "elapsed": 0.0,
            "data": {
                "model": OPENAI_VISION_MODEL,
                "system_instruction": VISION_LAYOUT_SYSTEM_INSTRUCTION,
                "vision_prompt": vision_prompt,
                "meta_slice": meta_obj,
                "input_image_ref": "image_gen",
                "input_image_width": w_in,
                "input_image_height": h_in,
            },
        })
        t0 = time.perf_counter()
        try:
            layout_raw, layout_json = await asyncio.to_thread(
                _vision_layout_sync, api_key, OPENAI_VISION_MODEL, image_b64, vision_prompt
            )
        except Exception as ex:
            total = round(time.perf_counter() - t_start, 2)
            yield _ndjson({"event": "step_error", "step": "vision_layout",
                           "elapsed": total, "step_elapsed": round(time.perf_counter() - t0, 2),
                           "message": str(ex)})
            yield _ndjson({"event": "complete", "elapsed": total, "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)
        total = round(time.perf_counter() - t_start, 2)
        yield _ndjson({"event": "step_done", "step": "vision_layout",
                       "elapsed": total, "step_elapsed": step_elapsed,
                       "data": {
                           "layout_json": layout_json,
                           "layout_raw": layout_raw,
                           "parsed": layout_json is not None,
                           "vision_prompt": vision_prompt,
                       }})
        yield _ndjson({"event": "complete", "elapsed": total, "status": "ok"})

    return _stream_response(gen())


@app.post("/api/image-edit")
async def api_image_edit(
    file: UploadFile = File(...),
    prompt: str = Form(""),
) -> JSONResponse:
    """제품 이미지 편집: gpt-image-2 + medium (config)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    try:
        with Image.open(BytesIO(raw)) as im:
            im.load()
            ow, oh = im.size
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지를 읽을 수 없습니다: {e}") from e

    use_prompt = (prompt or "").strip() or "제품 패키지 이미지를 자연스럽게 다듬어 주세요."
    api_size = choose_api_size(ow, oh)
    api_key = _openai_api_key()
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY 가 config.py 또는 환경 변수에 없습니다.",
        )

    try:
        out = await asyncio.to_thread(
            _openai_image_edit_sync, api_key, raw, use_prompt, api_size
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI Images API 오류: {e}") from e

    return JSONResponse(
        {
            "b64_json": out["b64_json"],
            "filename": file.filename or "upload",
            "original_width": ow,
            "original_height": oh,
            "api_size": api_size,
            "output_width": out["output_width"],
            "output_height": out["output_height"],
            "model": OPENAI_IMAGE_MODEL,
            "quality": OPENAI_IMAGE_QUALITY,
        }
    )


@app.get("/health")
async def health() -> dict:
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)