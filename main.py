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
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from google import genai
from google.genai import types as genai_types
from openai import OpenAI
from PIL import Image

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

_ASSETS = Path(__file__).resolve().parent / "assets"
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
    """모델 응답에서 JSON 객체 1개를 파싱. 실패 시 (raw, None)."""
    text = (raw or "").strip()
    if not text:
        return text, None
    try:
        return text, json.loads(text)
    except json.JSONDecodeError:
        pass
    # ```json ... ``` 또는 본문 중 첫 { ... } 블록 시도
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        inner = fence.group(1).strip()
        try:
            return text, json.loads(inner)
        except json.JSONDecodeError:
            pass
    brace = re.search(r"\{[\s\S]*\}", text)
    if brace:
        try:
            return text, json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass
    return text, None


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

# 4단계 (image_gen_text) 에서 선택 가능한 이미지 모델 + 품질 조합 (첫 항목이 기본값)
IMAGE_GEN_TEXT_OPTIONS: list[dict[str, str]] = [
    {"model": OPENAI_IMAGE_MODEL, "quality": "low",    "label": f"{OPENAI_IMAGE_MODEL} (low)"},
    {"model": OPENAI_IMAGE_MODEL, "quality": "medium", "label": f"{OPENAI_IMAGE_MODEL} (medium)"},
    {"model": OPENAI_IMAGE_MODEL, "quality": "high",   "label": f"{OPENAI_IMAGE_MODEL} (high)"},
    {"model": GEMINI_IMAGE_MODEL, "quality": "",       "label": GEMINI_IMAGE_MODEL},
]


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
                "align": "left",
                "font_color": "#111111",
                "style": "bold",
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


def _images_generate_sync(
    api_key: str,
    prompt: str,
    size: str,
    product_image: bytes | None = None,
    product_filename: str | None = None,
    model: str | None = None,
    quality: str | None = None,
    extra_images: list[tuple[bytes, str]] | None = None,
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
    "문구는 아래 문구만 사용해"
)

STEP6_TEXT_REMOVE_PROMPT = "아래 영역을 지운 이미지 생성해.\n영역내 글자만 지워.\n영역 이외는 절대 수정하지 마.\n영역내에서도 도형, 로고, 이미지는 지우면 안 돼"


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


_INTERNAL_QUOTE_RE = re.compile(r"['\"‘’“”]")


def _strip_leading_decoration(s: str) -> str:
    """선행/후행 불릿·따옴표·문장부호·장식 문자 제거 (• · ♥ - * ' " . , ! ? 등)."""
    s = (s or "").strip()
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

    - 글자색: 배경 대비 가장 어두운/밝은 픽셀(글자에 해당) 의 평균 RGB
    - bold: 글자 픽셀 비율 (>= 0.18) 이면 'bold', 아니면 'normal'
    실패 시 ('#111111', 'bold') 반환.
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
        # 각 픽셀의 명도(luminance)
        lums = [0.299 * r + 0.587 * g + 0.114 * b for (r, g, b) in pixels]
        mean_lum = sum(lums) / len(lums)
        # 배경이 밝으면(>128) 글자는 어두운 픽셀, 반대면 밝은 픽셀
        if mean_lum >= 128:
            text_pixels = [
                pixels[i] for i, lm in enumerate(lums) if lm < mean_lum - 25
            ]
        else:
            text_pixels = [
                pixels[i] for i, lm in enumerate(lums) if lm > mean_lum + 25
            ]
        if not text_pixels:
            # 컨트라스트가 약하면 quantile 기반으로 다시 시도
            sorted_idx = sorted(range(len(lums)), key=lambda k: lums[k])
            cut = max(1, len(sorted_idx) // 6)
            text_pixels = [pixels[i] for i in sorted_idx[:cut]]
        n = len(text_pixels)
        ar = sum(p[0] for p in text_pixels) / n
        ag = sum(p[1] for p in text_pixels) / n
        ab = sum(p[2] for p in text_pixels) / n
        color = f"#{int(ar):02X}{int(ag):02X}{int(ab):02X}"
        density = n / len(pixels)
        weight = "bold" if density >= 0.18 else "normal"
        return (color, weight)
    except Exception:
        return ("#111111", "bold")


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
        "align": first.get("align", "left"),
        "font_color": first.get("font_color", "#111111"),
        "style": first.get("style", "bold"),
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


INDEX_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>상세페이지 프롬프트 생성</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Pretendard",
        "Noto Sans KR", sans-serif;
      margin: 0; padding: 24px;
      background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
      color: #1f2937;
    }
    h1 { font-size: 24px; margin: 0 0 16px; }
    .container { max-width: none; margin: 0; }
    .progress-row {
      display: flex; gap: 18px; align-items: flex-start; flex-wrap: nowrap;
      margin-bottom: 18px;
    }
    .progress-row > .progress-left {
      flex: 0 0 800px; max-width: 800px; min-width: 0;
      display: flex; flex-direction: column; gap: 18px;
      max-height: calc(100vh - 110px); overflow-y: auto;
    }
    .progress-row > .progress-left > .card { margin-bottom: 0; }
    .progress-row > #overlayHistoryCard {
      flex: 0 0 1000px; max-width: 1000px; min-width: 0; margin-bottom: 0;
      max-height: calc(100vh - 110px); overflow-y: auto;
    }
    @media (max-width: 1850px) {
      .progress-row { flex-wrap: wrap; }
      .progress-row > .progress-left {
        flex: 1 1 100%; max-width: 800px;
        max-height: none; overflow-y: visible;
      }
      .progress-row > #overlayHistoryCard {
        flex: 1 1 100%; max-width: 1000px;
        max-height: none; overflow-y: visible;
      }
    }
    .overlay-history-entry {
      margin-top: 12px; padding: 8px 10px;
      border: 1px solid #e5e7eb; border-radius: 6px; background: #f8fafc;
    }
    .overlay-history-entry:first-of-type { margin-top: 0; }
    .overlay-history-entry > .ohe-title {
      font-weight: 600; font-size: 13px; margin-bottom: 6px; color: #1e293b;
    }
    .overlay-history-entry > .ohe-meta {
      font-size: 12px; color: #64748b; margin-bottom: 6px;
    }
    .overlay-history-entry .ohe-preview { max-width: 100%; }
    .card {
      background: #fff; border-radius: 14px; padding: 22px;
      box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
      margin-bottom: 18px;
    }
    label { display: block; font-weight: 600; font-size: 14px; margin-bottom: 6px; }
    textarea {
      width: 100%; padding: 10px 12px; border: 1px solid #d1d5db;
      border-radius: 8px; font-size: 14px; font-family: inherit;
      background: #fff;
    }
    textarea { min-height: 120px; resize: vertical; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    @media (max-width: 720px) { .row { grid-template-columns: 1fr; } }
    button {
      background: #2563eb; color: #fff; border: 0; padding: 12px 22px;
      border-radius: 10px; font-size: 15px; font-weight: 600; cursor: pointer;
    }
    button:disabled { background: #94a3b8; cursor: not-allowed; }
    .progress-line {
      padding: 6px 10px; border-left: 3px solid #2563eb;
      margin: 4px 0; font-size: 13px; color: #374151; background: #f8fafc;
      border-radius: 4px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    .progress-line.running { border-color: #f59e0b; color: #92400e; background: #fffbeb; }
    .progress-line.done { border-color: #16a34a; color: #166534; background: #f0fdf4; }
    .progress-line.error { border-color: #dc2626; color: #b91c1c; background: #fef2f2; }
    .progress-line.skip { border-color: #94a3b8; color: #475569; background: #f1f5f9; }
    .progress-line .badge {
      display: inline-block; min-width: 64px; text-align: center;
      padding: 1px 6px; border-radius: 4px; margin-right: 6px;
      font-size: 11px; font-weight: 700; letter-spacing: .3px;
      background: #e5e7eb; color: #374151;
    }
    .progress-line.running .badge { background: #fef3c7; color: #92400e; }
    .progress-line.done .badge { background: #dcfce7; color: #166534; }
    .progress-line.error .badge { background: #fee2e2; color: #b91c1c; }
    .progress-line.skip .badge { background: #e2e8f0; color: #475569; }
    .progress-line .dur { float: right; color: #6b7280; }
    .progress-line .step-num {
      display: inline-block; min-width: 22px; margin-right: 4px;
      color: #1f2937; font-weight: 700;
    }
    .label-edit-wrap {
      position: relative; display: block; max-width: 100%;
      margin: 0 auto;
      line-height: 0; container-type: inline-size;
    }
    .label-edit-wrap > img.label-edit-base {
      display: block; width: 100%; height: auto;
      border-radius: 8px; border: 1px solid #e5e7eb; background: #fff;
      user-select: none;
    }
    .label-edit-box {
      position: absolute; box-sizing: border-box; padding: 0; margin: 0;
      line-height: 1.2; white-space: nowrap; overflow: visible;
      font-family: "Noto Sans KR","Pretendard","Malgun Gothic",sans-serif;
      outline: none; background: transparent;
      cursor: move; user-select: none;
    }
    .label-edit-box.selected {
      outline: 2px solid #2563eb;
      background: rgba(37,99,235,0.06);
    }
    .label-edit-box[contenteditable="true"] {
      cursor: text; user-select: text;
      outline: 2px solid #16a34a !important;
      background: rgba(22,163,74,0.06);
    }
    /* 글자 영역 테두리 표시 (체크박스 토글) */
    .show-text-bounds .label-edit-box::after {
      content: "";
      position: absolute;
      inset: 0;
      border: 1px dashed #ef4444;
      pointer-events: none;
    }
    .step-result {
      margin: 4px 0 10px 12px; padding: 8px 10px;
      background: #fff; border: 1px solid #e5e7eb; border-radius: 6px;
      font-size: 12px; color: #374151;
    }
    .step-result pre {
      margin: 6px 0 0; padding: 8px; background: #0f172a; color: #e2e8f0;
      border-radius: 6px; max-height: 220px; overflow: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: pre-wrap; word-break: break-word; font-size: 12px;
    }
    .meta { color: #6b7280; font-size: 13px; margin: 0 0 12px; line-height: 1.5; }
    .dropzone {
      position: relative;
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      gap: 6px;
      min-height: 140px; padding: 18px;
      border: 2px dashed #cbd5e1; border-radius: 12px;
      background: #f8fafc; color: #475569;
      cursor: pointer; text-align: center;
      transition: border-color .15s, background-color .15s;
    }
    .dropzone:hover { border-color: #94a3b8; background: #f1f5f9; }
    .dropzone.dragover {
      border-color: #2563eb; background: #eff6ff; color: #1d4ed8;
    }
    .dropzone .dz-title { font-weight: 600; font-size: 14px; display: block; }
    .dropzone .dz-sub { font-size: 12px; color: #64748b; display: block; margin-top: 4px; }
    .dropzone .dz-fname {
      display: block; margin-top: 8px; font-size: 12px; color: #1d4ed8; font-weight: 600;
      word-break: break-all;
    }
    .dropzone .dz-preview {
      display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; justify-content: center;
      width: 100%;
    }
    .dropzone .dz-preview img {
      max-width: 160px; max-height: 160px;
      border-radius: 8px; border: 1px solid #e5e7eb; object-fit: contain; background: #fff;
    }
    .gpt-out {
      background: #0f172a; color: #e2e8f0; font-size: 13px; line-height: 1.55;
      border-radius: 8px; padding: 14px; max-height: 520px; overflow: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: pre-wrap; word-break: break-word;
      margin: 0; border: 1px solid #1e293b;
    }
    .gen-img-wrap {
      border: 1px solid #e5e7eb; border-radius: 12px; padding: 14px;
      background: #f8fafc; text-align: center; margin-bottom: 14px;
    }
    .gen-img-wrap img {
      max-width: 100%; height: auto; border-radius: 8px;
      border: 1px solid #e5e7eb; object-fit: contain; background: #fff;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>상세페이지 이미지 프롬프트 생성</h1>

    <div class="progress-row" id="progressRow">
      <div class="progress-left">
        <div class="card" id="inputCard">
          <form id="genForm">
            <div>
              <label style="display:block">제품 사진</label>
              <label class="dropzone" id="productDz" for="file">
                <span class="dz-title">클릭 또는 파일을 여기로 드래그</span>
                <span class="dz-fname" id="fileName"></span>
                <span class="dz-preview" id="previewBox"></span>
              </label>
              <input type="file" id="file" name="file" accept="image/*" required style="display:none" />
            </div>
            <div style="margin-top:14px">
              <label for="prompt">추가 프롬프트 (선택)</label>
              <textarea id="prompt" name="prompt" placeholder="상품에 대한 추가 설명이 있으면 입력하세요. OCR 결과 아래에 함께 붙습니다."></textarea>
            </div>

            <div style="margin-top:16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap">
              <button type="submit" id="submitBtn">생성</button>
            </div>
          </form>
        </div>
        <div class="card" id="progressCard" style="display:none">
          <h2 style="margin-top:0;font-size:18px">진행 상황</h2>
          <div id="progressBox"></div>
        </div>
      </div>
      <div class="card" id="overlayHistoryCard" style="display:none">
        <div style="display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:6px;flex-wrap:wrap">
          <h2 style="margin:0;font-size:18px">글자 합성 결과 (페이지별)</h2>
          <label style="display:inline-flex;align-items:center;gap:6px;font-size:13px;font-weight:500;color:#374151;cursor:pointer;user-select:none">
            <input type="checkbox" id="toggleTextBounds" style="margin:0">
            글자 영역 테두리 표시
          </label>
        </div>
        <p class="meta" style="font-size:12px;color:#64748b;margin:0 0 10px">
          글자 제거 이미지 + 합성된 글자(canvas) — 페이지마다 누적 표시
        </p>
        <div id="overlayHistoryBox"></div>
      </div>
    </div>

    <div class="card" id="vizCard" style="display:none">
      <h3 id="cleanedOcrTitle" style="display:none;margin:0 0 8px;font-size:16px">6단계 결과: 글자 제거 이미지 OCR (라인+좌표)</h3>
      <p class="meta" id="cleanedOcrMeta" style="display:none"></p>
      <pre id="cleanedOcrOut" class="gpt-out" style="max-height:280px;margin-top:6px;display:none"></pre>

      <h3 id="areaDiffTitle" style="display:none;margin:18px 0 8px;font-size:16px">7단계 결과: 사라진 OCR 영역 (원본 ∖ 글자 제거)</h3>
      <p class="meta" id="areaDiffMeta" style="display:none"></p>
      <pre id="areaDiffOut" class="gpt-out" style="max-height:280px;margin-top:6px;display:none"></pre>

      <h3 id="labeledTitle" style="display:none;margin:18px 0 8px;font-size:16px">문구를 합성한 이미지 (좌표 JSON 적용)</h3>
      <div class="gen-img-wrap" id="labeledImgWrap" style="display:none"></div>
    </div>
  </div>

  <script>
    window.addEventListener("error", (ev) => {
      const box = document.getElementById("progressBox");
      if (!box) return;
      const card = document.getElementById("progressCard");
      if (card) card.style.display = "block";
      const div = document.createElement("div");
      div.className = "progress-line error";
      div.textContent = "JS 오류: " + (ev.message || ev.error || "");
      box.appendChild(div);
    });

    const $ = (id) => document.getElementById(id);
    const stepRows = {}; // step -> { line, result }
    let lastGeneratedImageB64 = null;
    let stepCounter = 0;
    const stepInputs = { ocr: null, gpt: null, image_gen: null, image_gen2: null,
                         image_gen_ocr: null, image_gen_text: null, cleaned_ocr: null,
                         text_area_diff: null, removed_text_overlay: null, vision_layout: null };
    // 다음 페이지 생성용 상태
    let allImagePrompts = null;     // gpt 단계 결과의 image_prompts (배열)
    let nextPageToGen = 2;          // 다음에 생성할 페이지 번호 (1-based, 첫 페이지=1 은 메인 파이프라인)

    function extractLabels(layoutJson) {
      if (!layoutJson) return [];
      if (Array.isArray(layoutJson)) return layoutJson;
      for (const k of Object.keys(layoutJson)) {
        if (Array.isArray(layoutJson[k])) return layoutJson[k];
      }
      return [];
    }

    function renderLabelsOnImage(b64, labels, onDone) {
      if (!b64) return onDone(null);
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        const fontFamily = '"Noto Sans KR","Pretendard","Malgun Gothic",sans-serif';
        for (const lab of labels) {
          if (!lab || typeof lab !== "object") continue;
          const text = String(lab.text == null ? "" : lab.text);
          if (!text) continue;
          const x = Number(lab.x || 0);
          const y = Number(lab.y || 0);
          const fontSize = Number(lab.font_size || 32);
          const styleStr = String(lab.style || "").toLowerCase();
          const isBold = styleStr.includes("bold") || styleStr.includes("heavy");
          const isItalic = styleStr.includes("italic");
          const color = String(lab.font_color || "#111");
          const align = (lab.align === "center" || lab.align === "right") ? lab.align : "left";
          let weight = isBold ? "bold" : "normal";
          let italic = isItalic ? "italic " : "";
          ctx.font = italic + weight + " " + fontSize + "px " + fontFamily;
          ctx.fillStyle = color;
          ctx.textAlign = align;
          ctx.textBaseline = "top";
          const lineH = fontSize * 1.25;
          const lines = text.split(/\\r?\\n/);
          let drawX = x;
          if (lab.width && align === "center") drawX = x + Number(lab.width) / 2;
          else if (lab.width && align === "right") drawX = x + Number(lab.width);
          lines.forEach((line, i) => {
            ctx.fillText(line, drawX, y + i * lineH);
          });
        }
        onDone(canvas);
      };
      img.onerror = () => onDone(null);
      img.src = "data:image/png;base64," + b64;
    }

    let activeOverlayBox = null;

    function _deselectOverlayBox() {
      if (activeOverlayBox) {
        activeOverlayBox.classList.remove("selected");
        activeOverlayBox.contentEditable = "false";
        try { activeOverlayBox.blur(); } catch (_) {}
      }
      activeOverlayBox = null;
    }

    function _selectOverlayBox(box) {
      if (activeOverlayBox && activeOverlayBox !== box) _deselectOverlayBox();
      activeOverlayBox = box;
      box.classList.add("selected");
    }

    function _startDragOverlayBox(box, startEvt) {
      const wrap = box.parentElement;
      if (!wrap) return;
      const wrapRect = wrap.getBoundingClientRect();
      const startX = startEvt.clientX;
      const startY = startEvt.clientY;
      const origLeftPct = parseFloat(box.style.left) || 0;
      const origTopPct = parseFloat(box.style.top) || 0;
      function onMove(e) {
        const dxPx = e.clientX - startX;
        const dyPx = e.clientY - startY;
        const dxPct = wrapRect.width > 0 ? (dxPx / wrapRect.width * 100) : 0;
        const dyPct = wrapRect.height > 0 ? (dyPx / wrapRect.height * 100) : 0;
        box.style.left = (origLeftPct + dxPct).toFixed(3) + "%";
        box.style.top = (origTopPct + dyPct).toFixed(3) + "%";
      }
      function onUp() {
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
      }
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
    }

    function _outsideMouseDownDeselect(e) {
      if (!activeOverlayBox) return;
      if (e.target.closest && e.target.closest(".label-edit-box") === activeOverlayBox) return;
      _deselectOverlayBox();
    }
    document.addEventListener("mousedown", _outsideMouseDownDeselect);

    function renderLabelsAsDOM(b64, labels, container) {
      if (!container) return;
      container.innerHTML = "";
      if (!b64) return;
      const img = new Image();
      img.onload = () => {
        const natW = img.naturalWidth || 1;
        const natH = img.naturalHeight || 1;

        const wrap = document.createElement("div");
        wrap.className = "label-edit-wrap";
        wrap.style.width = natW + "px";
        wrap.style.aspectRatio = natW + " / " + natH;

        const baseImg = document.createElement("img");
        baseImg.className = "label-edit-base";
        baseImg.src = "data:image/png;base64," + b64;
        baseImg.draggable = false;
        wrap.appendChild(baseImg);

        for (const lab of (labels || [])) {
          if (!lab || typeof lab !== "object") continue;
          const text = String(lab.text == null ? "" : lab.text);
          const x = Number(lab.x || 0);
          const y = Number(lab.y || 0);
          const w = Number(lab.width || 100);
          const h = Number(lab.height || 30);
          const fontSize = Number(lab.font_size || Math.max(12, h * 0.85));
          const color = String(lab.font_color || "#111");
          const styleStr = String(lab.style || "").toLowerCase();
          const isBold = styleStr.includes("bold") || styleStr.includes("heavy");
          const isItalic = styleStr.includes("italic");
          const align = (lab.align === "center" || lab.align === "right") ? lab.align : "left";

          const box = document.createElement("div");
          box.className = "label-edit-box";
          box.textContent = text;
          box.contentEditable = "false";
          box.spellcheck = false;
          box.dataset.origX = String(x);
          box.dataset.origY = String(y);
          box.dataset.origW = String(w);
          box.dataset.origH = String(h);
          box.style.left = (x / natW * 100).toFixed(3) + "%";
          box.style.top = (y / natH * 100).toFixed(3) + "%";
          box.style.width = (w / natW * 100).toFixed(3) + "%";
          box.style.minHeight = (h / natH * 100).toFixed(3) + "%";
          // 컨테이너 가로 크기 대비 비율로 폰트 크기 (cqi: 컨테이너 가로 1% 단위)
          box.style.fontSize = (fontSize / natW * 100).toFixed(3) + "cqi";
          box.style.color = color;
          box.style.fontWeight = isBold ? "bold" : "normal";
          box.style.fontStyle = isItalic ? "italic" : "normal";
          box.style.textAlign = align;

          box.addEventListener("mousedown", (e) => {
            if (box.contentEditable === "true") return;
            e.stopPropagation();
            e.preventDefault();
            _selectOverlayBox(box);
            _startDragOverlayBox(box, e);
          });
          box.addEventListener("dblclick", (e) => {
            e.stopPropagation();
            e.preventDefault();
            _selectOverlayBox(box);
            box.contentEditable = "true";
            box.focus();
            // 전체 선택
            try {
              const range = document.createRange();
              range.selectNodeContents(box);
              const sel = window.getSelection();
              sel.removeAllRanges();
              sel.addRange(range);
            } catch (_) {}
          });
          box.addEventListener("blur", () => {
            box.contentEditable = "false";
          });

          wrap.appendChild(box);
        }

        container.appendChild(wrap);
      };
      img.onerror = () => { container.textContent = "이미지 로드 실패"; };
      img.src = "data:image/png;base64," + b64;
    }

    function renderLabeledCanvas(b64, labels) {
      renderLabelsOnImage(b64, labels, (canvas) => {
        if (!canvas) return;
        canvas.style.maxWidth = "100%";
        canvas.style.height = "auto";
        canvas.style.borderRadius = "8px";
        canvas.style.border = "1px solid #e5e7eb";
        canvas.style.background = "#fff";
        const wrap = $("labeledImgWrap");
        wrap.innerHTML = "";
        wrap.appendChild(canvas);
        wrap.style.display = "block";
        $("labeledTitle").style.display = "block";
      });
    }

    let lastImageWithTextB64 = null;

    // 다음 페이지 생성: 페이지 접미사(__pN)가 붙은 step 이벤트 처리.
    // 메인 파이프라인의 단일-페이지 상태(stepInputs, lastImageWithTextB64 등)는 건드리지 않는다.
    function handlePageStreamEvent(ev) {
      const stepKey = ev.step || "";
      const m = stepKey.match(/^(.+?)__p(\d+)$/);
      // 페이지 단위 complete 이벤트는 step 이 없고 page 필드로 식별
      if (!m && ev.event === "complete" && ev.page != null) {
        appendInfo("페이지 " + ev.page + " 완료 — 소요 " + fmtSec(ev.elapsed) +
          " (상태: " + (ev.status || "ok") + ")",
          ev.status === "ok" ? "done" : "error");
        return true;
      }
      if (!m) return false;
      const baseStep = m[1];
      const pageNum = Number(m[2]);
      if (ev.event === "step_start") {
        startStep(stepKey, ev.label || stepKey, ev.elapsed);
        const sd = ev.data || {};
        if (sd.model) {
          appendStepDetail(stepKey, "사용 AI 모델:", sd.model);
        }
        if (baseStep === "image_gen2" && sd.image_prompt) {
          appendStepDetail(stepKey,
            "이미지 생성 API 전달 프롬프트 (제품 사진 + 문구):",
            sd.image_prompt);
        }
        return true;
      }
      if (ev.event === "step_done") {
        // 재실행 버튼이 step_done 후 finishStep 에서 표시되도록 stepInputs 에 컨텍스트 저장.
        // image_gen2__pN 만 재실행을 지원 (페이지 N 의 4~7 단계 전체를 다시 실행).
        if (baseStep === "image_gen2") {
          stepInputs[stepKey] = { kind: "next-page", pageNum: pageNum };
        }
        finishStep(stepKey, "완료", "done", ev.elapsed, ev.step_elapsed);
        const d = ev.data || {};
        const row = stepRows[stepKey];
        if (!row) return true;
        row.result.style.display = "block";
        if (baseStep === "image_gen2") {
          const note = document.createElement("div");
          note.textContent = "페이지 " + pageNum + " 이미지 " +
            (d.first_image_width || "?") + "×" + (d.first_image_height || "?") + " 생성됨.";
          note.style.fontWeight = "600";
          note.style.marginTop = "6px";
          const thumb = document.createElement("img");
          thumb.src = "data:image/png;base64," + d.first_image_b64;
          thumb.style.maxWidth = "540px";
          thumb.style.marginTop = "4px";
          thumb.style.border = "1px solid #e5e7eb";
          thumb.style.borderRadius = "6px";
          row.result.appendChild(note);
          row.result.appendChild(thumb);
        } else if (baseStep === "image_gen_ocr") {
          appendStepDetail(stepKey,
            "OCR 라인 (" + (d.line_count || 0) + "개, " + (d.ocr_chars || 0) + "자):",
            JSON.stringify(d.lines || [], null, 2));
        } else if (baseStep === "image_gen_text") {
          const note = document.createElement("div");
          note.textContent = "페이지 " + pageNum + " 글자 제거 이미지 " +
            (d.image_width || "?") + "×" + (d.image_height || "?") + ".";
          note.style.fontWeight = "600";
          note.style.marginTop = "6px";
          const thumb = document.createElement("img");
          thumb.src = "data:image/png;base64," + d.image_b64;
          thumb.style.maxWidth = "540px";
          thumb.style.marginTop = "4px";
          thumb.style.border = "1px solid #e5e7eb";
          thumb.style.borderRadius = "6px";
          row.result.appendChild(note);
          row.result.appendChild(thumb);
        } else if (baseStep === "removed_text_overlay") {
          const lines = d.lines || [];
          const base = d.base_image_b64;
          appendStepDetail(stepKey,
            "합성 라인 수:", String(d.line_count || lines.length || 0));
          // 미리보기는 오른쪽 오버레이 히스토리 패널에 누적
          addOverlayHistoryEntry(pageNum, base, lines, {
            width: d.base_image_width,
            height: d.base_image_height,
          });
        }
        return true;
      }
      if (ev.event === "step_skip") {
        startStep(stepKey, stepKey, ev.elapsed);
        finishStep(stepKey, "건너뜀", "skip", ev.elapsed, null, ev.message);
        return true;
      }
      if (ev.event === "step_error") {
        if (!stepRows[stepKey]) startStep(stepKey, stepKey, ev.elapsed);
        finishStep(stepKey, "오류", "error", ev.elapsed, ev.step_elapsed, ev.message);
        return true;
      }
      return false;
    }

    // 오른쪽 "글자 합성 결과" 패널에 페이지별 항목을 추가/갱신.
    // 동일 페이지 번호의 항목이 있으면 in-place 로 교체한다.
    function addOverlayHistoryEntry(pageNum, base, lines, meta) {
      const box = $("overlayHistoryBox");
      const card = $("overlayHistoryCard");
      if (!box || !card) return;
      card.style.display = "block";
      const id = "ohe-p" + pageNum;
      let entry = document.getElementById(id);
      if (!entry) {
        entry = document.createElement("div");
        entry.id = id;
        entry.className = "overlay-history-entry";
        entry.dataset.page = String(pageNum);
        // 페이지 번호 순으로 정렬 삽입
        const siblings = Array.from(box.querySelectorAll(".overlay-history-entry"));
        let inserted = false;
        for (const sib of siblings) {
          if (Number(sib.dataset.page || 0) > pageNum) {
            box.insertBefore(entry, sib);
            inserted = true;
            break;
          }
        }
        if (!inserted) box.appendChild(entry);
      } else {
        entry.innerHTML = "";
      }
      const title = document.createElement("div");
      title.className = "ohe-title";
      title.textContent = "페이지 " + pageNum;
      entry.appendChild(title);
      const metaLine = document.createElement("div");
      metaLine.className = "ohe-meta";
      const w = (meta && meta.width) || "?";
      const h = (meta && meta.height) || "?";
      metaLine.textContent =
        "이미지 " + w + "×" + h + " px · 라벨 " + (lines ? lines.length : 0) + "개";
      entry.appendChild(metaLine);
      const previewWrap = document.createElement("div");
      previewWrap.className = "ohe-preview";
      entry.appendChild(previewWrap);
      renderLabelsAsDOM(base, lines || [], previewWrap);
    }

    // 8단계(removed_text_overlay) 결과 영역에 "다음 페이지 생성" 컨트롤을 (재)삽입.
    function maybeAddNextPageControls(container) {
      if (!container) return;
      container.querySelectorAll(".next-page-controls").forEach((el) => el.remove());
      if (!Array.isArray(allImagePrompts) || allImagePrompts.length <= 1) return;
      const remaining = allImagePrompts.length - (nextPageToGen - 1);
      if (remaining <= 0) {
        const done = document.createElement("div");
        done.className = "next-page-controls";
        done.style.cssText = "margin-top:14px;padding:8px 10px;border:1px dashed #94a3b8;border-radius:6px;background:#f1f5f9;color:#475569;font-size:13px";
        done.textContent = "모든 페이지(" + allImagePrompts.length + "장) 생성 완료.";
        container.appendChild(done);
        return;
      }
      const wrap = document.createElement("div");
      wrap.className = "next-page-controls";
      wrap.style.cssText = "margin-top:14px;padding:10px 12px;border:1px solid #cbd5e1;border-radius:8px;background:#f8fafc;display:flex;align-items:center;gap:8px;flex-wrap:wrap";

      const label = document.createElement("label");
      label.textContent = "추가 페이지 수:";
      label.style.cssText = "font-weight:600;font-size:13px;margin:0";
      wrap.appendChild(label);

      const sel = document.createElement("select");
      sel.className = "next-page-count";
      sel.style.cssText = "padding:4px 8px;border:1px solid #cbd5e1;border-radius:6px;font-size:13px";
      const opts = [];
      if (remaining >= 1) opts.push({ v: "1", t: "1" });
      if (remaining >= 2) opts.push({ v: "2", t: "2" });
      opts.push({ v: "all", t: "전체 (" + remaining + ")" });
      opts.forEach((o) => {
        const op = document.createElement("option");
        op.value = o.v;
        op.textContent = o.t;
        sel.appendChild(op);
      });
      wrap.appendChild(sel);

      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = "다음 페이지 생성";
      btn.style.cssText = "padding:6px 14px;font-size:13px;background:#2563eb;color:#fff;border:0;border-radius:6px;cursor:pointer";
      btn.addEventListener("click", () => {
        const v = sel.value;
        let count = v === "all" ? remaining : Number(v);
        if (!count || count < 1) return;
        if (count > remaining) count = remaining;
        btn.disabled = true;
        sel.disabled = true;
        const orig = btn.textContent;
        btn.textContent = "생성 중...";
        runNextPages(count).finally(() => {
          btn.disabled = false;
          sel.disabled = false;
          btn.textContent = orig;
          // 컨트롤을 다시 그려 남은 페이지 수 반영
          maybeAddNextPageControls(container);
        });
      });
      wrap.appendChild(btn);

      const hint = document.createElement("span");
      hint.style.cssText = "color:#64748b;font-size:12px;margin-left:auto";
      hint.textContent = "남은 페이지 " + remaining + " 장 (전체 " + allImagePrompts.length + " 장)";
      wrap.appendChild(hint);

      container.appendChild(wrap);
    }

    async function runNextPages(count) {
      const fileEl = $("file");
      const file = fileEl && fileEl.files && fileEl.files[0];
      if (!file) {
        appendInfo("원본 제품 사진을 찾을 수 없습니다. 폼에서 다시 선택해주세요.", "error");
        return;
      }
      if (!Array.isArray(allImagePrompts) || allImagePrompts.length === 0) {
        appendInfo("image_prompts 목록이 없습니다.", "error");
        return;
      }
      // 작업 목록을 먼저 만들고 nextPageToGen 을 한 번에 전진시킨 뒤, 모든 페이지를 병렬로 시작.
      // 각 페이지의 스트림은 독립적으로 흐르며, 합성(removed_text_overlay) 이벤트가 도착하는 즉시
      // addOverlayHistoryEntry(pageNum, ...) 가 오른쪽 패널에 페이지 번호 순으로 삽입한다.
      const tasks = [];
      for (let i = 0; i < count; i++) {
        if (nextPageToGen > allImagePrompts.length) break;
        const promptIdx = nextPageToGen - 1;
        const pageNum = nextPageToGen;
        nextPageToGen += 1;
        tasks.push({ promptIdx, pageNum, prompt: allImagePrompts[promptIdx] });
      }
      if (!tasks.length) return;
      appendInfo("페이지 " + tasks.map((t) => t.pageNum).join(", ") + " 병렬 생성 시작 (" + tasks.length + "개)");
      await Promise.all(tasks.map(async (t) => {
        const fd = new FormData();
        fd.append("file", file, file.name);
        fd.append("image_prompt_json", JSON.stringify(t.prompt));
        fd.append("page_index", String(t.pageNum));
        try {
          const resp = await fetch("/api/next-page", { method: "POST", body: fd });
          await consumeStream(resp);
        } catch (e) {
          appendInfo("페이지 " + t.pageNum + " 생성 오류: " + e, "error");
        }
      }));
    }

    function handleStreamEvent(ev) {
      if (handlePageStreamEvent(ev)) return;
      if (ev.event === "step_start") {
        startStep(ev.step, ev.label || ev.step, ev.elapsed);
        const sd = ev.data || {};
        if (sd.model && ev.step !== "image_gen_text") {
          appendStepDetail(ev.step, "사용 AI 모델:", sd.model);
        }
        if (ev.step === "image_gen_text" && Array.isArray(sd.available_options) && sd.available_options.length) {
          const row = stepRows["image_gen_text"];
          if (row) {
            row.result.style.display = "block";
            const wrap = document.createElement("div");
            wrap.style.cssText = "display:flex;align-items:center;gap:8px;margin-top:6px";
            const lab = document.createElement("span");
            lab.style.fontWeight = "600";
            lab.textContent = "사용 AI 모델:";
            const sel = document.createElement("select");
            sel.style.cssText = "padding:4px 8px;font-size:13px;border:1px solid #cbd5e1;border-radius:6px;background:#fff";
            const curM = String(sd.model || "");
            const curQ = String(sd.quality || "");
            for (const opt of sd.available_options) {
              if (!opt || typeof opt !== "object") continue;
              const o = document.createElement("option");
              o.value = (opt.model || "") + "|" + (opt.quality || "");
              o.textContent = opt.label || ((opt.model || "") + (opt.quality ? " (" + opt.quality + ")" : ""));
              o.dataset.model = opt.model || "";
              o.dataset.quality = opt.quality || "";
              if ((opt.model || "") === curM && (opt.quality || "") === curQ) o.selected = true;
              sel.appendChild(o);
            }
            wrap.appendChild(lab);
            wrap.appendChild(sel);
            row.result.appendChild(wrap);
            row.modelSelect = sel;
          }
        }
        if (ev.step === "gpt") {
          if (sd.user_message) {
            appendStepDetail(
              "gpt",
              "Gemini 전달 입력 메시지 (" + (sd.user_message_chars || sd.user_message.length) + "자)" +
              (sd.has_product_image ? " + 업로드 제품 사진" : "") + ":",
              sd.user_message
            );
          }
          if ((sd.ui_prompt || "").trim()) {
            appendStepDetail("gpt", "UI 추가 프롬프트:", sd.ui_prompt);
          }
        }
        if (ev.step === "image_gen") {
          if (sd.user_message) {
            appendStepDetail("image_gen",
              "OpenAI 챗 전달 프롬프트 (image_prompts[0] 포함):",
              sd.user_message);
          } else if (sd.image_prompt) {
            appendStepDetail("image_gen",
              "image_prompts[0] (JSON):",
              sd.image_prompt);
          }
        }
        if (ev.step === "image_gen2" && sd.image_prompt) {
          appendStepDetail("image_gen2",
            "이미지 생성 API 전달 프롬프트 (제품 사진 + 문구):",
            sd.image_prompt);
        }
        if (ev.step === "image_gen_text") {
          if (sd.text_prompt) {
            const ta = appendEditableStepDetail("image_gen_text",
              "글자 제거 프롬프트 (4단계 이미지 + 5단계 OCR 중 4단계 입력 문구와 일치하는 영역 좌표, 편집 가능):",
              sd.text_prompt);
            if (stepRows["image_gen_text"]) stepRows["image_gen_text"].promptEdit = ta;
          }
        }
        if (ev.step === "image_diff") {
        }
        if (ev.step === "vision_layout") {
          if (sd.input_image_ref === "image_gen" && lastGeneratedImageB64) {
            const row = stepRows["vision_layout"];
            if (row) {
              row.result.style.display = "block";
              const lab = document.createElement("div");
              lab.style.fontWeight = "600";
              lab.style.marginTop = "6px";
              lab.textContent =
                "입력 이미지 (image_gen 단계에서 생성된 PNG) (" +
                (sd.input_image_width || "?") + "×" +
                (sd.input_image_height || "?") + " px)";
              row.result.appendChild(lab);
            }
          }
          if (sd.vision_prompt) {
            appendStepDetail("vision_layout", "비전 API 전달 프롬프트 (메타 JSON 포함):", sd.vision_prompt);
          }
        }
      } else if (ev.event === "step_done") {
        finishStep(ev.step, "완료", "done", ev.elapsed, ev.step_elapsed);
        const d = ev.data || {};
        if (ev.step === "ocr") {
          appendStepDetail("ocr", "OCR 결과 (" + (d.ocr_chars || 0) + "자):", previewText(d.ocr_text, 800));
          const baseFile = (stepInputs.ocr && stepInputs.ocr.file) || ($("file").files && $("file").files[0]);
          if (baseFile) {
            stepInputs.gpt = {
              file: baseFile, filename: baseFile.name,
              ocr_text: d.ocr_text || "", ui_prompt: $("prompt").value || ""
            };
          }
        } else if (ev.step === "gpt") {
          if (d.parsed && d.gpt_json) {
            appendStepDetail("gpt", "GPT JSON 파싱 성공:", JSON.stringify(d.gpt_json, null, 2));
            const prompts = d.gpt_json && d.gpt_json.image_prompts;
            allImagePrompts = Array.isArray(prompts) ? prompts : null;
            const first = (prompts && prompts[0]) || null;
            if (first) {
              stepInputs.image_gen = {
                image_prompt: JSON.stringify(first, null, 2),
              };
              stepInputs.image_gen_text = {
                first: first,
                image_b64: null,
                width_px: Number(first.width_px || 860),
                height_px: Number(first.height_px || 2000),
              };
            }
          } else {
            appendStepDetail("gpt", "GPT JSON 파싱 실패 — raw 출력:", d.gpt_raw || "");
          }
        } else if (ev.step === "image_gen") {
          const layout = d.layout_json || null;
          const jsonText = layout
            ? JSON.stringify(layout, null, 2)
            : (d.layout_raw || "");
          const row = stepRows["image_gen"];
          if (row) {
            row.result.style.display = "block";
            appendStepDetail("image_gen",
              d.parsed ? "문구 정보 JSON:" : "OpenAI 원본 출력 (파싱 실패):",
              jsonText);
          }
          stepInputs.image_gen2 = { layout_json: layout };
        } else if (ev.step === "image_gen2") {
          lastGeneratedImageB64 = d.first_image_b64;
          const row = stepRows["image_gen2"];
          if (row) {
            row.result.style.display = "block";
            const note = document.createElement("div");
            note.textContent =
              "이미지 " + (d.first_image_width || "?") + "×" + (d.first_image_height || "?") +
              " 생성됨.";
            const thumb = document.createElement("img");
            thumb.src = "data:image/png;base64," + d.first_image_b64;
            thumb.style.maxWidth = "540px";
            thumb.style.marginTop = "6px";
            thumb.style.border = "1px solid #e5e7eb";
            thumb.style.borderRadius = "6px";
            row.result.appendChild(note);
            row.result.appendChild(thumb);
          }
          if (stepInputs.image_gen_text) stepInputs.image_gen_text.image_b64 = d.first_image_b64;
          stepInputs.image_gen_ocr = { image_b64: d.first_image_b64 };
        } else if (ev.step === "image_gen_ocr") {
          appendStepDetail("image_gen_ocr",
            "상세페이지 이미지(4단계) OCR 라인 (" + (d.line_count || 0) + "개, " + (d.ocr_chars || 0) + "자):",
            JSON.stringify(d.lines || [], null, 2));
          // 라인 좌표를 보존: 6단계 글자 제거 프롬프트의 좌표 필터 입력으로 사용
          if (stepInputs.image_gen_ocr) stepInputs.image_gen_ocr.lines = d.lines || [];
          else stepInputs.image_gen_ocr = { image_b64: lastGeneratedImageB64, lines: d.lines || [] };
        } else if (ev.step === "image_gen_text") {
          lastImageWithTextB64 = d.image_b64;
          const row = stepRows["image_gen_text"];
          if (row) {
            row.result.style.display = "block";
            const thumb = document.createElement("img");
            thumb.src = "data:image/png;base64," + d.image_b64;
            thumb.style.maxWidth = "540px";
            thumb.style.marginTop = "6px";
            thumb.style.border = "1px solid #e5e7eb";
            thumb.style.borderRadius = "6px";
            row.result.appendChild(thumb);
          }
          // 5단계 결과(클린 이미지) 가 6단계(cleaned_ocr) 의 입력
          stepInputs.cleaned_ocr = { image_b64: d.image_b64 };
        } else if (ev.step === "cleaned_ocr") {
          $("vizCard").style.display = "block";
          $("cleanedOcrTitle").style.display = "block";
          $("cleanedOcrMeta").style.display = "block";
          $("cleanedOcrOut").style.display = "block";
          $("cleanedOcrMeta").textContent =
            "라인 " + (d.line_count || 0) + "개, 크기 " +
            (d.image_width || "?") + "×" + (d.image_height || "?") + " px";
          $("cleanedOcrOut").textContent = JSON.stringify(d.lines || [], null, 2);
          appendStepDetail("cleaned_ocr",
            "글자 제거 이미지 OCR 라인 (" + (d.line_count || 0) + "개):",
            JSON.stringify(d.lines || [], null, 2));
          // 7단계(text_area_diff) 의 입력: 원본 OCR(3.5단계) vs 클린 OCR(이번 단계)
          const beforeLines = (stepInputs.image_gen_ocr && stepInputs.image_gen_ocr.lines) ||
                              (stepInputs.text_area_diff && stepInputs.text_area_diff.before_lines) || [];
          stepInputs.text_area_diff = {
            before_lines: beforeLines,
            after_lines: d.lines || [],
            iou_threshold: 0.3,
          };
        } else if (ev.step === "text_area_diff") {
          $("vizCard").style.display = "block";
          $("areaDiffTitle").style.display = "block";
          $("areaDiffMeta").style.display = "block";
          $("areaDiffOut").style.display = "block";
          $("areaDiffMeta").textContent =
            "사라진 라인 " + (d.line_count || 0) + "개 (IoU < " +
            (d.iou_threshold != null ? d.iou_threshold : 0.3) + ")";
          $("areaDiffOut").textContent = JSON.stringify(d.lines || [], null, 2);
          appendStepDetail("text_area_diff",
            "사라진 OCR 영역 (" + (d.line_count || 0) + "개):",
            JSON.stringify(d.lines || [], null, 2));
          // 8단계(removed_text_overlay) 의 입력
          if (lastImageWithTextB64) {
            stepInputs.removed_text_overlay = {
              base_image_b64: lastImageWithTextB64,
              lines: d.lines || [],
            };
          }
        } else if (ev.step === "removed_text_overlay") {
          const base = d.base_image_b64 || lastImageWithTextB64;
          const lines = d.lines || [];
          stepInputs.removed_text_overlay = {
            base_image_b64: base,
            lines: lines,
          };
          // 합성 결과 미리보기는 진행 상황 영역에 두지 않고 오른쪽 패널에 누적
          addOverlayHistoryEntry(1, base, lines, {
            width: d.base_image_width,
            height: d.base_image_height,
          });
          const row = stepRows["removed_text_overlay"];
          if (row) {
            row.result.style.display = "block";
            appendStepDetail("removed_text_overlay",
              "합성 라인 수:", String(d.line_count || lines.length || 0));
            appendStepDetail("removed_text_overlay",
              "합성에 사용된 글자 정보 (text/좌표/font_size=5단계 OCR, font_color/style=4단계 이미지 픽셀 샘플링, align=left):",
              JSON.stringify(lines, null, 2));
            // 8단계 완료 → 다음 페이지 생성 컨트롤 추가
            maybeAddNextPageControls(row.result);
          }
        } else if (ev.step === "vision_layout") {
          if (d.parsed && d.layout_json) {
            appendStepDetail("vision_layout", "문구 좌표 JSON 파싱 성공:",
              JSON.stringify(d.layout_json, null, 2));
          } else {
            appendStepDetail("vision_layout", "문구 좌표 JSON 파싱 실패 — raw 출력:",
              previewText(d.layout_raw, 700));
          }
        }
      } else if (ev.event === "step_skip") {
        startStep(ev.step, ev.step, ev.elapsed);
        finishStep(ev.step, "건너뜀", "skip", ev.elapsed, null, ev.message);
      } else if (ev.event === "step_error") {
        if (!stepRows[ev.step]) startStep(ev.step, ev.step, ev.elapsed);
        finishStep(ev.step, "오류", "error", ev.elapsed, ev.step_elapsed, ev.message);
      } else if (ev.event === "complete") {
        appendInfo(
          "전체 완료 — 총 소요 시간: " + fmtSec(ev.elapsed) +
          " (상태: " + (ev.status || "ok") + ")",
          ev.status === "ok" ? "done" : "error"
        );
      }
    }

    async function consumeStream(resp) {
      if (!resp.ok || !resp.body) {
        let detail = "HTTP " + resp.status;
        try {
          const j = await resp.json();
          if (j && j.detail) detail += " — " + (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail));
        } catch (_) {}
        appendInfo("요청 실패: " + detail, "error");
        return;
      }
      for await (const ev of readNdjson(resp)) {
        handleStreamEvent(ev);
      }
    }

    function rerunStep(step) {
      const input = stepInputs[step];
      if (!input) {
        appendInfo("재실행 불가: " + step + " 단계의 저장된 입력이 없습니다.", "error");
        return;
      }
      const fd = new FormData();
      let url = "";
      // 다음 페이지 4단계(image_gen2__pN) 재실행 — /api/next-page 를 같은 page_index 로 다시 호출.
      // 페이지 N 의 4~7 단계 행이 그대로 갱신된다.
      if (input.kind === "next-page") {
        const pageNum = Number(input.pageNum || 0);
        const promptIdx = pageNum - 1;
        const fileEl = $("file");
        const file = fileEl && fileEl.files && fileEl.files[0];
        if (!file || !Array.isArray(allImagePrompts) || !allImagePrompts[promptIdx]) {
          appendInfo("재실행 불가: 원본 사진 또는 image_prompts[" + promptIdx + "] 누락", "error");
          return;
        }
        fd.append("file", file, file.name);
        fd.append("image_prompt_json", JSON.stringify(allImagePrompts[promptIdx]));
        fd.append("page_index", String(pageNum));
        fetch("/api/next-page", { method: "POST", body: fd })
          .then(consumeStream)
          .catch((err) => appendInfo("재실행 오류: " + err, "error"));
        return;
      }
      if (step === "ocr") {
        if (!input.file) return appendInfo("저장된 파일이 없습니다.", "error");
        fd.append("file", input.file, input.filename || "upload.png");
        url = "/api/step/ocr";
      } else if (step === "gpt") {
        fd.append("file", input.file, input.filename || "upload.png");
        fd.append("ocr_text", input.ocr_text || "");
        fd.append("prompt", input.ui_prompt || "");
        url = "/api/step/gpt";
      } else if (step === "image_gen") {
        if (!input.image_prompt) return appendInfo("image_prompts[0] 가 없습니다.", "error");
        fd.append("image_prompt", input.image_prompt);
        url = "/api/step/image_gen";
      } else if (step === "image_gen_ocr") {
        if (!input.image_b64) return appendInfo("3단계 이미지가 없습니다.", "error");
        fd.append("image_b64", input.image_b64);
        url = "/api/step/image_gen_ocr";
      } else if (step === "image_gen_text") {
        if (!input.image_b64) return appendInfo("3단계 이미지가 없습니다.", "error");
        fd.append("image_b64", input.image_b64);
        fd.append("first_json", JSON.stringify(input.first || {}));
        fd.append("width_px", String(input.width_px || 860));
        fd.append("height_px", String(input.height_px || 2000));
        // 원본 제품 사진도 함께 입력으로 전달
        const origFile = (stepInputs.ocr && stepInputs.ocr.file) ||
                         ($("file").files && $("file").files[0]);
        if (origFile) fd.append("original", origFile, origFile.name);
        const promptTa = stepRows["image_gen_text"] && stepRows["image_gen_text"].promptEdit;
        const editedPrompt = promptTa ? promptTa.value : "";
        if ((editedPrompt || "").trim()) fd.append("prompt", editedPrompt);
        const modelSel = stepRows["image_gen_text"] && stepRows["image_gen_text"].modelSelect;
        if (modelSel) {
          const opt = modelSel.options[modelSel.selectedIndex];
          const chosenModel = (opt && (opt.dataset.model || "")) || "";
          const chosenQuality = (opt && (opt.dataset.quality || "")) || "";
          if (chosenModel.trim()) fd.append("model", chosenModel);
          if (chosenQuality.trim()) fd.append("quality", chosenQuality);
        }
        url = "/api/step/image_gen_text";
      } else if (step === "cleaned_ocr") {
        if (!input.image_b64) return appendInfo("글자 제거 이미지가 없습니다.", "error");
        fd.append("image_b64", input.image_b64);
        url = "/api/step/cleaned_ocr";
      } else if (step === "text_area_diff") {
        fd.append("before_json", JSON.stringify(input.before_lines || []));
        fd.append("after_json", JSON.stringify(input.after_lines || []));
        fd.append("iou_threshold", String(input.iou_threshold || 0.3));
        url = "/api/step/text_area_diff";
      } else if (step === "removed_text_overlay") {
        // 항상 최신 6단계(image_gen_text) 결과 이미지를 base 로 사용
        const baseB64 = lastImageWithTextB64 || input.base_image_b64;
        if (!baseB64) return appendInfo("글자 제거 이미지가 없습니다.", "error");
        if (lastImageWithTextB64 && stepInputs.removed_text_overlay) {
          stepInputs.removed_text_overlay.base_image_b64 = lastImageWithTextB64;
        }
        fd.append("base_image_b64", baseB64);
        fd.append("lines_json", JSON.stringify(input.lines || []));
        url = "/api/step/removed_text_overlay";
      } else if (step === "vision_layout") {
        fd.append("meta_slice", JSON.stringify(input.meta_slice || {}));
        fd.append("image_b64", input.image_b64);
        url = "/api/step/vision_layout";
      } else {
        return;
      }
      fetch(url, { method: "POST", body: fd })
        .then(consumeStream)
        .catch((err) => appendInfo("재실행 오류: " + err, "error"));
    }

    function regenOverlayFromDom() {
      // 페이지 1 항목의 편집 가능한 오버레이(label-edit-wrap)를 오버레이 히스토리 패널에서 찾는다.
      const entry = document.getElementById("ohe-p1");
      const overlayWrap = entry && entry.querySelector(".label-edit-wrap");
      if (!overlayWrap) {
        appendInfo("재생성 대상 오버레이를 찾지 못했습니다.", "error");
        return;
      }
      const baseImg = overlayWrap.querySelector("img.label-edit-base");
      const natW = (baseImg && baseImg.naturalWidth) || 0;
      const natH = (baseImg && baseImg.naturalHeight) || 0;
      if (!natW || !natH) {
        appendInfo("이미지 원본 크기를 확인할 수 없습니다.", "error");
        return;
      }
      const prev = (stepInputs.removed_text_overlay &&
                    stepInputs.removed_text_overlay.lines) || [];
      const boxes = overlayWrap.querySelectorAll(".label-edit-box");
      const updated = [];
      boxes.forEach((box, idx) => {
        const src = prev[idx] || {};
        const leftPct = parseFloat(box.style.left) || 0;
        const topPct = parseFloat(box.style.top) || 0;
        const widthPct = parseFloat(box.style.width) || 0;
        updated.push({
          ...src,
          text: box.textContent || "",
          x: Math.round(leftPct / 100 * natW),
          y: Math.round(topPct / 100 * natH),
          width: Math.round(widthPct / 100 * natW) || src.width || 0,
          height: src.height || 0,
        });
      });
      if (!stepInputs.removed_text_overlay) {
        appendInfo("이전 합성 입력이 없습니다.", "error");
        return;
      }
      stepInputs.removed_text_overlay = {
        base_image_b64: stepInputs.removed_text_overlay.base_image_b64,
        lines: updated,
      };
      rerunStep("removed_text_overlay");
    }

    function appendStepImage(step, label, b64, w, h) {
      const row = stepRows[step];
      if (!row || !b64) return;
      row.result.style.display = "block";
      const lab = document.createElement("div");
      lab.style.fontWeight = "600";
      lab.style.marginTop = "6px";
      lab.textContent = label + " (" + (w || "?") + "×" + (h || "?") + " px)";
      const thumb = document.createElement("img");
      thumb.src = "data:image/png;base64," + b64;
      thumb.style.maxWidth = "480px";
      thumb.style.marginTop = "4px";
      thumb.style.border = "1px solid #e5e7eb";
      thumb.style.borderRadius = "6px";
      row.result.appendChild(lab);
      row.result.appendChild(thumb);
    }

    function fmtSec(s) {
      if (s == null || isNaN(s)) return "?";
      return Number(s).toFixed(2) + "s";
    }

    function renderPreview(files) {
      const box = $("previewBox");
      const fn = $("fileName");
      if (box) box.innerHTML = "";
      if (fn) fn.textContent = "";
      const f = files && files[0];
      if (!f || !box) return;
      if (fn) fn.textContent = "선택됨: " + f.name + " (" + Math.round(f.size/1024) + " KB)";
      const img = document.createElement("img");
      img.alt = f.name || "preview";
      box.appendChild(img);
      try {
        if (typeof URL !== "undefined" && URL.createObjectURL) {
          img.src = URL.createObjectURL(f);
          img.onload = () => { try { URL.revokeObjectURL(img.src); } catch (_) {} };
        } else { throw new Error("URL.createObjectURL 미지원"); }
      } catch (_) {
        const reader = new FileReader();
        reader.onload = (ev) => { img.src = String(ev.target.result || ""); };
        reader.readAsDataURL(f);
      }
    }

    const fileInput = $("file");
    if (fileInput) {
      fileInput.addEventListener("change", (e) => {
        renderPreview(e.target.files);
      });
    } else {
      console.warn("file input not found at script load");
    }

    function setupDropzone(dzId, inputId) {
      const dz = $(dzId);
      const input = $(inputId);
      if (!dz || !input) { console.warn("dropzone/input not found", dzId, inputId); return; }
      const stop = (ev) => { ev.preventDefault(); ev.stopPropagation(); };

      // 클릭은 <label for="file">의 네이티브 동작이 처리. JS는 드래그·드롭만 담당.

      ["dragenter", "dragover"].forEach(t => dz.addEventListener(t, (e) => {
        stop(e);
        if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
        dz.classList.add("dragover");
      }));
      ["dragleave", "dragend"].forEach(t => dz.addEventListener(t, (e) => {
        stop(e); dz.classList.remove("dragover");
      }));
      dz.addEventListener("drop", (e) => {
        stop(e); dz.classList.remove("dragover");
        const dropped = Array.from(e.dataTransfer && e.dataTransfer.files || [])
          .filter(f => f.type.startsWith("image/"));
        if (dropped.length === 0) { alert("이미지 파일만 끌어넣어 주세요."); return; }
        try {
          const dt = new DataTransfer();
          dt.items.add(dropped[0]);
          input.files = dt.files;
        } catch (_) { /* DataTransfer 미지원 환경 폴백 */ }
        renderPreview([dropped[0]]);
        input.dispatchEvent(new Event("change", { bubbles: true }));
      });
    }

    setupDropzone("productDz", "file");
    ["dragover", "drop"].forEach(t => window.addEventListener(t, (e) => e.preventDefault()));

    // "글자 영역 테두리 표시" 체크박스 — overlayHistoryBox 에 클래스 토글
    const toggleBoundsCb = $("toggleTextBounds");
    if (toggleBoundsCb) {
      toggleBoundsCb.addEventListener("change", () => {
        const box = $("overlayHistoryBox");
        if (!box) return;
        box.classList.toggle("show-text-bounds", toggleBoundsCb.checked);
      });
    }

    const applyBtn = $("applyLabelsBtn");
    if (applyBtn) {
      applyBtn.addEventListener("click", () => {
        const msg = $("applyLabelsMsg");
        if (!lastGeneratedImageB64) {
          msg.style.color = "#b91c1c";
          msg.textContent = "먼저 이미지 생성을 완료해주세요.";
          return;
        }
        const raw = $("layoutOut").value || "";
        let parsed;
        try {
          parsed = JSON.parse(raw);
        } catch (e) {
          msg.style.color = "#b91c1c";
          msg.textContent = "JSON 파싱 오류: " + e.message;
          return;
        }
        const labels = extractLabels(parsed);
        if (!labels.length) {
          msg.style.color = "#b91c1c";
          msg.textContent = "라벨 배열을 찾을 수 없습니다.";
          return;
        }
        renderLabeledCanvas(lastGeneratedImageB64, labels);
        msg.style.color = "#166534";
        msg.textContent = labels.length + "개 라벨로 다시 그렸습니다.";
      });
    }

    const copyLayoutBtn = $("copyLayoutBtn");
    if (copyLayoutBtn) {
      copyLayoutBtn.addEventListener("click", () => {
        copyToClipboard($("layoutOut").value || "", copyLayoutBtn);
      });
    }

    function appendInfo(msg, cls) {
      const div = document.createElement("div");
      div.className = "progress-line" + (cls ? " " + cls : "");
      div.textContent = msg;
      $("progressBox").appendChild(div);
      $("progressBox").scrollTop = $("progressBox").scrollHeight;
      return div;
    }

    function startStep(step, label, elapsed) {
      if (stepRows[step]) {
        const row = stepRows[step];
        row.line.className = "progress-line running";
        row.line.querySelector(".badge").textContent = "실행중";
        row.line.querySelector(".dur").textContent = "";
        row.line.querySelector("b").textContent = label || step;
        const btn = row.line.querySelector(".rerun-btn");
        if (btn) btn.style.display = "none";
        row.result.innerHTML = "";
        row.result.style.display = "none";
        return;
      }
      stepCounter += 1;
      const line = document.createElement("div");
      line.className = "progress-line running";
      const isOverlay = step === "removed_text_overlay";
      line.innerHTML =
        '<span class="step-num"></span>' +
        '<span class="badge">실행중</span>' +
        '<b></b>' +
        '<span class="dur"></span>' +
        '<button type="button" class="rerun-btn" style="display:none;margin-left:8px;padding:2px 10px;' +
        'font-size:11px;border:1px solid #2563eb;background:#fff;color:#2563eb;' +
        'border-radius:6px;cursor:pointer">재실행</button>' +
        (isOverlay
          ? '<button type="button" class="regen-btn" style="display:none;margin-left:6px;padding:2px 10px;' +
            'font-size:11px;border:0;background:#2563eb;color:#fff;' +
            'border-radius:6px;cursor:pointer">재생성</button>'
          : '');
      line.querySelector(".step-num").textContent = stepCounter + ".";
      line.querySelector("b").textContent = label || step;
      const rerunBtn = line.querySelector(".rerun-btn");
      rerunBtn.addEventListener("click", () => rerunStep(step));
      if (isOverlay) {
        const regenBtn = line.querySelector(".regen-btn");
        regenBtn.addEventListener("click", () => regenOverlayFromDom());
      }
      $("progressBox").appendChild(line);
      const result = document.createElement("div");
      result.className = "step-result";
      result.style.display = "none";
      $("progressBox").appendChild(result);
      $("progressBox").scrollTop = $("progressBox").scrollHeight;
      stepRows[step] = { line, result };
    }

    function setStepResult(step, htmlOrText, isHtml=false) {
      const row = stepRows[step];
      if (!row) return;
      row.result.style.display = "block";
      if (isHtml) row.result.innerHTML = htmlOrText;
      else row.result.textContent = htmlOrText;
    }

    function copyToClipboard(text, btn) {
      const ok = (msg) => {
        if (!btn) return;
        const orig = btn.textContent;
        btn.textContent = msg || "복사됨";
        btn.disabled = true;
        setTimeout(() => { btn.textContent = orig; btn.disabled = false; }, 1200);
      };
      const fail = () => ok("복사 실패");
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(String(text == null ? "" : text)).then(() => ok(), fail);
          return;
        }
      } catch (_) { /* fallthrough */ }
      // 폴백: 임시 textarea + execCommand
      try {
        const ta = document.createElement("textarea");
        ta.value = String(text == null ? "" : text);
        ta.style.position = "fixed"; ta.style.left = "-1000px"; ta.style.top = "-1000px";
        document.body.appendChild(ta);
        ta.select();
        const done = document.execCommand("copy");
        document.body.removeChild(ta);
        if (done) ok(); else fail();
      } catch (_) { fail(); }
    }

    function makeCopyBtn(getText) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "copy-btn";
      btn.textContent = "복사";
      btn.style.cssText =
        "margin-left:8px;padding:1px 8px;font-size:11px;border:1px solid #94a3b8;" +
        "background:#fff;color:#475569;border-radius:5px;cursor:pointer;vertical-align:middle";
      btn.addEventListener("click", (e) => {
        e.preventDefault(); e.stopPropagation();
        copyToClipboard(typeof getText === "function" ? getText() : getText, btn);
      });
      return btn;
    }

    function appendEditableStepDetail(step, label, content) {
      const row = stepRows[step];
      if (!row) return null;
      row.result.style.display = "block";
      const wrap = document.createElement("div");
      const lab = document.createElement("div");
      lab.style.fontWeight = "600";
      lab.style.marginTop = "6px";
      lab.style.display = "flex";
      lab.style.alignItems = "center";
      lab.style.gap = "4px";
      const labText = document.createElement("span");
      labText.textContent = label;
      lab.appendChild(labText);
      const ta = document.createElement("textarea");
      ta.value = String(content == null ? "" : content);
      ta.style.cssText =
        "width:100%;min-height:90px;max-height:280px;padding:10px;" +
        "background:#0f172a;color:#e2e8f0;border:1px solid #1e293b;border-radius:8px;" +
        "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;line-height:1.5;" +
        "white-space:pre-wrap;word-break:break-word;resize:vertical;display:block;margin-top:4px";
      lab.appendChild(makeCopyBtn(() => ta.value));
      wrap.appendChild(lab);
      wrap.appendChild(ta);
      row.result.appendChild(wrap);
      return ta;
    }

    function appendStepDetail(step, label, content) {
      const row = stepRows[step];
      if (!row) return;
      row.result.style.display = "block";
      const wrap = document.createElement("div");
      const lab = document.createElement("div");
      lab.style.fontWeight = "600";
      lab.style.marginTop = "6px";
      lab.style.display = "flex";
      lab.style.alignItems = "center";
      lab.style.gap = "4px";
      const labText = document.createElement("span");
      labText.textContent = label;
      lab.appendChild(labText);
      const pre = document.createElement("pre");
      pre.textContent = content;
      lab.appendChild(makeCopyBtn(() => pre.textContent));
      wrap.appendChild(lab);
      wrap.appendChild(pre);
      row.result.appendChild(wrap);
    }

    function finishStep(step, badge, cls, elapsed, stepElapsed, message) {
      const row = stepRows[step];
      if (!row) {
        appendInfo("[" + step + "] " + (message || ""), cls);
        return;
      }
      row.line.className = "progress-line " + cls;
      row.line.querySelector(".badge").textContent = badge;
      row.line.querySelector(".dur").textContent =
        stepElapsed != null ? "+" + fmtSec(stepElapsed) : "";
      const rerunBtn = row.line.querySelector(".rerun-btn");
      if (rerunBtn) {
        rerunBtn.style.display = (cls === "done" || cls === "error") && stepInputs[step]
          ? "inline-block" : "none";
      }
      const regenBtn = row.line.querySelector(".regen-btn");
      if (regenBtn) {
        regenBtn.style.display = (cls === "done" || cls === "error") && stepInputs[step]
          ? "inline-block" : "none";
      }
      if (message) {
        const note = document.createElement("div");
        note.style.marginTop = "4px";
        note.style.fontSize = "12px";
        note.textContent = message;
        row.line.appendChild(note);
      }
    }

    function previewText(s, max=600) {
      if (s == null) return "";
      const str = String(s);
      return str.length > max ? str.slice(0, max) + "…(생략)" : str;
    }

    async function* readNdjson(resp) {
      const reader = resp.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\\n")) >= 0) {
          const line = buf.slice(0, idx).trim();
          buf = buf.slice(idx + 1);
          if (!line) continue;
          try { yield JSON.parse(line); }
          catch (_) { /* ignore malformed line */ }
        }
      }
      const tail = buf.trim();
      if (tail) {
        try { yield JSON.parse(tail); } catch (_) {}
      }
    }

    $("genForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      const input = $("file");
      const files = input.files || [];
      if (!files.length) { alert("제품 사진을 선택해주세요"); return; }

      const fd = new FormData();
      fd.append("file", files[0], files[0].name);
      fd.append("prompt", $("prompt").value || "");

      // 입력 저장: OCR은 사진만, 나머지는 각 단계 완료 후 채워짐
      stepInputs.ocr = { file: files[0], filename: files[0].name };
      stepInputs.gpt = null;
      stepInputs.image_gen = null;
      stepInputs.image_gen_ocr = null;
      stepInputs.image_gen_text = null;
      stepInputs.cleaned_ocr = null;
      stepInputs.text_area_diff = null;
      stepInputs.removed_text_overlay = null;
      stepInputs.vision_layout = null;
      // 다음 페이지 생성 상태 초기화
      allImagePrompts = null;
      nextPageToGen = 2;

      $("progressCard").style.display = "block";
      $("progressBox").innerHTML = "";
      // 오버레이 히스토리 패널 초기화 (다음 단계 결과가 들어오면 다시 표시)
      $("overlayHistoryBox").innerHTML = "";
      $("overlayHistoryCard").style.display = "none";
      for (const k of Object.keys(stepRows)) delete stepRows[k];
      lastGeneratedImageB64 = null;
      lastImageWithTextB64 = null;
      stepCounter = 0;
      $("vizCard").style.display = "none";
      $("cleanedOcrTitle").style.display = "none";
      $("cleanedOcrMeta").style.display = "none";
      $("cleanedOcrOut").style.display = "none";
      $("cleanedOcrOut").textContent = "";
      $("areaDiffTitle").style.display = "none";
      $("areaDiffMeta").style.display = "none";
      $("areaDiffOut").style.display = "none";
      $("areaDiffOut").textContent = "";
      $("labeledTitle").style.display = "none";
      $("labeledImgWrap").style.display = "none";
      $("labeledImgWrap").innerHTML = "";
      $("submitBtn").disabled = true;
      appendInfo("요청 시작 — 단계별 진행 상황과 소요 시간을 실시간으로 보여줍니다.");

      try {
        const resp = await fetch("/api/generate", { method: "POST", body: fd });
        if (!resp.ok || !resp.body) {
          let detail = "HTTP " + resp.status;
          try {
            const j = await resp.json();
            if (j && j.detail) detail += " — " + (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail));
          } catch (_) {}
          appendInfo("요청 실패: " + detail, "error");
          $("submitBtn").disabled = false;
          return;
        }

        for await (const ev of readNdjson(resp)) {
          handleStreamEvent(ev);
        }
      } catch (err) {
        appendInfo("스트림 오류: " + err, "error");
      } finally {
        $("submitBtn").disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


def _ndjson(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


async def _generate_stream(
    raw: bytes,
    filename: str,
    suffix: str,
    prompt: str,
) -> AsyncIterator[bytes]:
    t_start = time.perf_counter()

    def elapsed_total() -> float:
        return round(time.perf_counter() - t_start, 2)

    tmp_path: Path | None = None
    try:
        fd_tmp, tmp_name = tempfile.mkstemp(suffix=suffix, prefix="product_")
        os.close(fd_tmp)
        tmp_path = Path(tmp_name)
        tmp_path.write_bytes(raw)

        # 1) OCR
        yield _ndjson({"event": "step_start", "step": "ocr", "label": "OCR (Document AI)", "elapsed": elapsed_total()})
        t0 = time.perf_counter()
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
            yield _ndjson({
                "event": "step_skip", "step": "image_gen",
                "elapsed": elapsed_total(),
                "message": "image_prompts 가 비어 있어 3단계를 건너뜁니다.",
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
                "image_prompt": image_gen2_prompt,
                "layout_texts": layout_texts,
            },
        })
        t0 = time.perf_counter()
        first_image_b64: str | None = None
        first_image_width: int | None = None
        first_image_height: int | None = None
        try:
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
        text_prompt = STEP6_TEXT_REMOVE_PROMPT
        if matched_lines_compact:
            text_prompt += (
                "\n\n[지울 영역 좌표 (이미지 좌상단 원점, 픽셀 단위)]\n"
                + json.dumps(matched_lines_compact, ensure_ascii=False, indent=2)
            )
        step4_quality = "medium"
        yield _ndjson({
            "event": "step_start", "step": "image_gen_text",
            "label": f"글자 제거 이미지 재생성 ({OPENAI_IMAGE_MODEL}, {gen_size}, {step4_quality})",
            "elapsed": elapsed_total(),
            "data": {
                "model": OPENAI_IMAGE_MODEL,
                "quality": step4_quality,
                "available_options": IMAGE_GEN_TEXT_OPTIONS,
                "size": gen_size,
                "text_prompt": text_prompt,
            },
        })
        t0 = time.perf_counter()
        second_image_b64: str | None = None
        second_image_width: int | None = None
        second_image_height: int | None = None
        try:
            step3_bytes = base64.b64decode(first_image_b64)
            extra_for_step4 = [(raw, filename)] if raw else None
            num_input_imgs = 1 + (len(extra_for_step4) if extra_for_step4 else 0)
            extra_names = (
                [n for (_b, n) in extra_for_step4] if extra_for_step4 else []
            )
            """
            print("-" * 60)
            print(
                f"[image_gen_text] (main) model={OPENAI_IMAGE_MODEL}, "
                f"quality={OPENAI_IMAGE_QUALITY}, size={gen_size}, "
                f"input_images={num_input_imgs} (primary=step3.png, extras={extra_names})"
            )
            print(f"[image_gen_text] prompt ({len(text_prompt)} chars):")
            print(text_prompt)
            print("-" * 60)
            """
            img_out_v2 = await asyncio.to_thread(
                _images_generate_sync, api_key, text_prompt, gen_size,
                step3_bytes, "step3.png", OPENAI_IMAGE_MODEL, OPENAI_IMAGE_QUALITY,
                extra_for_step4,
            )
            second_image_b64 = img_out_v2["b64_json"]
            second_image_width = int(img_out_v2["output_width"])
            second_image_height = int(img_out_v2["output_height"])
        except Exception as ex:
            yield _ndjson({
                "event": "step_error", "step": "image_gen_text",
                "elapsed": elapsed_total(), "step_elapsed": round(time.perf_counter() - t0, 2),
                "message": str(ex),
            })
            yield _ndjson({"event": "complete", "elapsed": elapsed_total(), "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)
        yield _ndjson({
            "event": "step_done", "step": "image_gen_text",
            "elapsed": elapsed_total(), "step_elapsed": step_elapsed,
            "data": {
                "image_b64": second_image_b64,
                "image_width": second_image_width,
                "image_height": second_image_height,
            },
        })

        # 9) 글자 제거 이미지 위에 합성: 5단계 OCR 의 text/좌표/font_size 사용,
        #    글자색 + bold 여부는 4단계 이미지의 OCR 영역 픽셀에서 추출
        sampled_lines = await asyncio.to_thread(
            _apply_pixel_styles_to_lines,
            base64.b64decode(first_image_b64),
            matched_ocr_lines,
        )
        overlay_labels = [
            {**ln, "align": "left"}
            for ln in sampled_lines
            if isinstance(ln, dict)
        ]
        yield _ndjson({
            "event": "step_start", "step": "removed_text_overlay",
            "label": "글자 제거 이미지에 글자 합성 (canvas) — 전부 5단계 OCR 정보 사용",
            "elapsed": elapsed_total(),
            "data": {
                "base_image_ref": "image_gen_text",
                "lines_ref": "image_gen_ocr",
                "line_count": len(overlay_labels),
            },
        })
        yield _ndjson({
            "event": "step_done", "step": "removed_text_overlay",
            "elapsed": elapsed_total(), "step_elapsed": 0.0,
            "data": {
                "base_image_b64": second_image_b64,
                "base_image_width": second_image_width,
                "base_image_height": second_image_height,
                "lines": overlay_labels,
                "line_count": len(overlay_labels),
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

    return StreamingResponse(
        _generate_stream(raw, file.filename or "upload", suffix, prompt),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


def _stream_response(gen: AsyncIterator[bytes]) -> StreamingResponse:
    return StreamingResponse(
        gen,
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
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
    # 원본 제품 이미지 (선택). form field "original" (UploadFile) 또는 "original_b64" (base64 문자열)
    extra_imgs: list[tuple[bytes, str]] = []
    orig_uf = form.get("original")
    if orig_uf is not None and hasattr(orig_uf, "read"):
        try:
            orig_bytes = await orig_uf.read()
            if orig_bytes:
                extra_imgs.append((orig_bytes, getattr(orig_uf, "filename", None) or "original.png"))
        except Exception:
            pass
    if not extra_imgs:
        orig_b64 = str(form.get("original_b64") or "")
        if orig_b64:
            try:
                extra_imgs.append((base64.b64decode(orig_b64), "original.png"))
            except Exception:
                pass

    async def gen() -> AsyncIterator[bytes]:
        t_start = time.perf_counter()
        api_key = _openai_api_key()
        if not api_key:
            yield _ndjson({"event": "step_error", "step": "image_gen_text",
                           "elapsed": 0.0,
                           "message": "OPENAI_API_KEY 가 config.py 또는 환경 변수에 없습니다."})
            yield _ndjson({"event": "complete", "elapsed": 0.0, "status": "error"})
            return
        gen_size = choose_api_size(width_px, height_px)
        text_prompt = (prompt or "").strip() or STEP6_TEXT_REMOVE_PROMPT
        use_model = (model or "").strip() or OPENAI_IMAGE_MODEL
        use_quality = (quality_arg or "").strip() or "medium"
        yield _ndjson({
            "event": "step_start", "step": "image_gen_text",
            "label": f"글자 제거 이미지 재생성 ({use_model}, {gen_size}, {use_quality})",
            "elapsed": 0.0,
            "data": {
                "model": use_model,
                "quality": use_quality,
                "available_options": IMAGE_GEN_TEXT_OPTIONS,
                "size": gen_size,
                "text_prompt": text_prompt,
            },
        })
        t0 = time.perf_counter()
        try:
            step3_bytes = base64.b64decode(image_b64)
            num_input_imgs = 1 + (len(extra_imgs) if extra_imgs else 0)
            extra_names = [n for (_b, n) in extra_imgs] if extra_imgs else []
            print("-" * 60)
            print(
                f"[image_gen_text] (rerun) model={use_model}, "
                f"quality={use_quality}, size={gen_size}, "
                f"input_images={num_input_imgs} (primary=step3.png, extras={extra_names})"
            )
            print(f"[image_gen_text] prompt ({len(text_prompt)} chars):")
            print(text_prompt)
            print("-" * 60)
            img_out_v2 = await asyncio.to_thread(
                _images_generate_sync, api_key, text_prompt, gen_size,
                step3_bytes, "step3.png", use_model, use_quality,
                extra_imgs or None,
            )
        except Exception as ex:
            total = round(time.perf_counter() - t_start, 2)
            yield _ndjson({"event": "step_error", "step": "image_gen_text",
                           "elapsed": total, "step_elapsed": round(time.perf_counter() - t0, 2),
                           "message": str(ex)})
            yield _ndjson({"event": "complete", "elapsed": total, "status": "error"})
            return
        step_elapsed = round(time.perf_counter() - t0, 2)
        total = round(time.perf_counter() - t_start, 2)
        yield _ndjson({"event": "step_done", "step": "image_gen_text",
                       "elapsed": total, "step_elapsed": step_elapsed,
                       "data": {
                           "image_b64": img_out_v2["b64_json"],
                           "image_width": int(img_out_v2["output_width"]),
                           "image_height": int(img_out_v2["output_height"]),
                       }})
        yield _ndjson({"event": "complete", "elapsed": total, "status": "ok"})

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
        image_gen2_quality = "low"
        yield _ndjson({
            "event": "step_start", "step": f"image_gen2{sx}",
            "label": f"페이지 {page_index} · 4단계 — 상세페이지 이미지 생성 ({OPENAI_IMAGE_MODEL}, {gen_size}, {image_gen2_quality})",
            "elapsed": elapsed_total(),
            "page": page_index,
            "data": {
                "model": OPENAI_IMAGE_MODEL, "size": gen_size,
                "quality": image_gen2_quality, "layout_texts": layout_texts,
                "image_prompt": image_gen2_prompt,
            },
        })
        t0 = time.perf_counter()
        try:
            img_out = await asyncio.to_thread(
                _images_generate_sync, api_key, image_gen2_prompt, gen_size,
                raw, filename, OPENAI_IMAGE_MODEL, image_gen2_quality,
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
        if matched_lines_compact:
            text_prompt += (
                "\n\n[지울 영역 좌표 (이미지 좌상단 원점, 픽셀 단위)]\n"
                + json.dumps(matched_lines_compact, ensure_ascii=False, indent=2)
            )
        step4_quality = "medium"
        yield _ndjson({
            "event": "step_start", "step": f"image_gen_text{sx}",
            "label": f"페이지 {page_index} · 6단계 — 글자 제거 이미지 재생성 ({OPENAI_IMAGE_MODEL}, {gen_size}, {step4_quality})",
            "elapsed": elapsed_total(), "page": page_index,
            "data": {"text_prompt": text_prompt},
        })
        t0 = time.perf_counter()
        try:
            step3_bytes = base64.b64decode(first_image_b64)
            img_out_v2 = await asyncio.to_thread(
                _images_generate_sync, api_key, text_prompt, gen_size,
                step3_bytes, "step3.png", OPENAI_IMAGE_MODEL, OPENAI_IMAGE_QUALITY,
                [(raw, filename)] if raw else None,
            )
            second_image_b64 = img_out_v2["b64_json"]
            second_image_width = int(img_out_v2["output_width"])
            second_image_height = int(img_out_v2["output_height"])
        except Exception as ex:
            yield _ndjson({
                "event": "step_error", "step": f"image_gen_text{sx}",
                "elapsed": elapsed_total(),
                "step_elapsed": round(time.perf_counter() - t0, 2),
                "message": str(ex), "page": page_index,
            })
            yield _complete("error")
            return
        yield _ndjson({
            "event": "step_done", "step": f"image_gen_text{sx}",
            "elapsed": elapsed_total(),
            "step_elapsed": round(time.perf_counter() - t0, 2),
            "page": page_index,
            "data": {
                "image_b64": second_image_b64,
                "image_width": second_image_width,
                "image_height": second_image_height,
            },
        })

        # --- 7단계: 글자 합성 (removed_text_overlay) ---
        sampled_lines = await asyncio.to_thread(
            _apply_pixel_styles_to_lines,
            base64.b64decode(first_image_b64),
            matched_ocr_lines,
        )
        overlay_labels = [
            {**ln, "align": "left"}
            for ln in sampled_lines
            if isinstance(ln, dict)
        ]
        yield _ndjson({
            "event": "step_start", "step": f"removed_text_overlay{sx}",
            "label": f"페이지 {page_index} · 7단계 — 글자 제거 이미지에 글자 합성 (canvas)",
            "elapsed": elapsed_total(), "page": page_index,
            "data": {"line_count": len(overlay_labels)},
        })
        yield _ndjson({
            "event": "step_done", "step": f"removed_text_overlay{sx}",
            "elapsed": elapsed_total(), "step_elapsed": 0.0,
            "page": page_index,
            "data": {
                "base_image_b64": second_image_b64,
                "base_image_width": second_image_width,
                "base_image_height": second_image_height,
                "lines": overlay_labels,
                "line_count": len(overlay_labels),
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
