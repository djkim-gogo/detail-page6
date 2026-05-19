"""Google Document AI 기반 OCR 헬퍼.

`D:\\cursor-workspace\\detail-page4\\ocr_util.py` 와 동일.
"""
from __future__ import annotations

import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List, Optional

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1
from PIL import Image

from config import LOCATION, OCR_MAX_DIM, OCR_PARALLELISM, PROCESSOR_ID, PROJECT_ID

_CLIENT: documentai_v1.DocumentProcessorServiceClient | None = None
_CLIENT_LOCK = threading.Lock()


def _get_client() -> documentai_v1.DocumentProcessorServiceClient:
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                opts = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
                _CLIENT = documentai_v1.DocumentProcessorServiceClient(
                    client_options=opts
                )
    return _CLIENT


def _ocr_bytes(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    client = _get_client()
    processor_name = client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)
    raw_document = documentai_v1.RawDocument(content=image_bytes, mime_type=mime_type)
    request = documentai_v1.ProcessRequest(name=processor_name, raw_document=raw_document)
    result = client.process_document(request=request)
    return (result.document.text or "").strip()


def _resize_for_ocr(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    if w <= max_dim and h <= max_dim:
        return img
    scale = min(max_dim / w, max_dim / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _encode_for_ocr(img: Image.Image, max_bytes: int = 28 * 1024 * 1024) -> tuple[bytes, str]:
    quality = 90
    cur = img
    while True:
        buf = io.BytesIO()
        cur.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data, "image/jpeg"
        if quality > 60:
            quality -= 10
            continue
        new_w = max(1, int(cur.size[0] * 0.85))
        new_h = max(1, int(cur.size[1] * 0.85))
        if new_w == cur.size[0] and new_h == cur.size[1]:
            return data, "image/jpeg"
        cur = cur.resize((new_w, new_h), Image.LANCZOS)
        quality = 90


ProgressFn = Callable[[str], None]


def _noop(_msg: str) -> None:  # pragma: no cover
    pass


def ocr_image_file(path: Path, progress: ProgressFn = _noop) -> tuple[str, Optional[str]]:
    try:
        file_size = path.stat().st_size
    except Exception:
        file_size = -1

    try:
        img = Image.open(path).convert("RGB")
    except Exception as ex:
        msg = f"이미지 열기 실패: {ex}"
        progress(f"[ocr] {path.name} {msg}")
        return "", msg

    orig_w, orig_h = img.size
    img_ocr = _resize_for_ocr(img, OCR_MAX_DIM)
    if img_ocr.size != (orig_w, orig_h):
        progress(
            f"[ocr] {path.name} 리사이즈 {orig_w}x{orig_h} → {img_ocr.size[0]}x{img_ocr.size[1]} "
            f"(max_dim={OCR_MAX_DIM})"
        )

    try:
        data, mime = _encode_for_ocr(img_ocr)
    except Exception as ex:
        msg = f"인코딩 실패: {ex}"
        progress(f"[ocr] {path.name} {msg}")
        return "", msg

    progress(
        f"[ocr] {path.name} 전송: 원본 {file_size//1024 if file_size>=0 else '?'}KB, "
        f"전송 {len(data)//1024}KB, dim={img_ocr.size[0]}x{img_ocr.size[1]} mime={mime}"
    )

    try:
        text = _ocr_bytes(data, mime_type=mime)
    except Exception as ex:
        msg = f"Document AI 호출 실패: {ex}"
        progress(f"[ocr] {path.name} {msg}")
        return "", msg

    if not text:
        return "", "OCR 결과가 비어 있음"

    return text, None


def ocr_images(
    paths: List[Path],
    progress: ProgressFn = _noop,
    max_workers: Optional[int] = None,
) -> str:
    if not paths:
        return ""

    workers = max(1, min(max_workers or OCR_PARALLELISM, len(paths)))
    lock = threading.Lock()

    def safe_progress(msg: str) -> None:
        with lock:
            try:
                progress(msg)
            except Exception:
                pass

    safe_progress(f"[ocr] 병렬 OCR 시작: 파일 {len(paths)}개 / 워커 {workers}개")

    results: List[Optional[tuple[str, Optional[str]]]] = [None] * len(paths)
    started = time.monotonic()

    def _worker(idx: int, path: Path) -> tuple[int, str, Optional[str]]:
        text, err = ocr_image_file(path, progress=safe_progress)
        return idx, text, err

    completed = 0
    total = len(paths)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ocr") as ex:
        futures = [ex.submit(_worker, i, p) for i, p in enumerate(paths)]
        for fut in as_completed(futures):
            try:
                idx, text, err = fut.result()
            except Exception as ex_inner:
                idx = -1
                text, err = "", f"예외: {ex_inner}"
                for j, slot in enumerate(results):
                    if slot is None:
                        results[j] = (text, err)
                        idx = j
                        break
            else:
                results[idx] = (text, err)

            completed += 1
            name = paths[idx].name if 0 <= idx < total else "?"
            status = f"실패: {err}" if err else f"{len(text)} 글자"
            safe_progress(f"[ocr] 진행 {completed}/{total} - {name} ({status})")

    elapsed = time.monotonic() - started
    safe_progress(f"[ocr] 병렬 OCR 완료: {total}개, {elapsed:.1f}s")

    pieces: List[str] = []
    for p, slot in zip(paths, results):
        text, err = slot if slot is not None else ("", "결과 누락")
        if err:
            header = f"--- {p.name} (실패: {err}) ---"
            body = ""
        else:
            header = f"--- {p.name} ---"
            body = text
        pieces.append(f"{header}\n{body}".rstrip())
    return "\n\n".join(pieces).strip()
