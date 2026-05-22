"""거친 사각 마스크와 원본 이미지를 받아, 마스크 안의 실제 글자 픽셀만 남긴
세부 마스크 PNG bytes 를 반환하는 헬퍼.

규약:
- 입력/출력 마스크 모두 OpenAI images.edit 의 mask 규약을 따른다:
  alpha=0  → 모델이 재생성할 영역 (글자 자리)
  alpha=255 → 원본 보존 영역
- 거친 마스크는 사각형들로 구성됐다고 가정. 사각형 안에서 배경색과 색차가
  threshold 이상인 픽셀만 글자로 본다.

호출 측 예:
    refined_png = refine_mask_to_text_pixels(
        image_bytes=first_image_b64_bytes,
        coarse_mask_png=mask_png_bytes,
    )
"""
from __future__ import annotations

from collections import deque
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


DEFAULT_THRESHOLD = 35.0
DEFAULT_DILATE = 2
DEFAULT_BG_MARGIN = 8
DEFAULT_MIN_AREA = 16

# 인페인트 기본 반복 블러 반경 (큰 → 작은 순)
DEFAULT_INPAINT_RADII: tuple[int, ...] = (40, 28, 20, 14, 10, 7, 5, 3, 2)


def find_mask_components(
    mask: np.ndarray, min_area: int = DEFAULT_MIN_AREA,
) -> list[tuple[int, int, int, int, int]]:
    """boolean 2D mask 의 4-연결 connected component 들의 (x, y, w, h, pixel_count).

    scipy 가 있으면 사용, 없으면 BFS 폴백.
    """
    try:
        from scipy.ndimage import label as _label, find_objects  # type: ignore
        labels, _n = _label(mask)
        rects: list[tuple[int, int, int, int, int]] = []
        for i, sl in enumerate(find_objects(labels), start=1):
            if sl is None:
                continue
            ys, xs = sl
            count = int(np.sum(labels[ys, xs] == i))
            if count < min_area:
                continue
            rects.append((xs.start, ys.start,
                          xs.stop - xs.start, ys.stop - ys.start, count))
        return rects
    except ImportError:
        pass

    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    rects = []
    for y0 in range(h):
        row_mask = mask[y0]
        row_visited = visited[y0]
        for x0 in range(w):
            if not row_mask[x0] or row_visited[x0]:
                continue
            q = deque([(y0, x0)])
            visited[y0, x0] = True
            min_x = max_x = x0
            min_y = max_y = y0
            count = 0
            while q:
                y, x = q.popleft()
                count += 1
                if x < min_x:
                    min_x = x
                elif x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                elif y > max_y:
                    max_y = y
                if y > 0 and mask[y - 1, x] and not visited[y - 1, x]:
                    visited[y - 1, x] = True
                    q.append((y - 1, x))
                if y + 1 < h and mask[y + 1, x] and not visited[y + 1, x]:
                    visited[y + 1, x] = True
                    q.append((y + 1, x))
                if x > 0 and mask[y, x - 1] and not visited[y, x - 1]:
                    visited[y, x - 1] = True
                    q.append((y, x - 1))
                if x + 1 < w and mask[y, x + 1] and not visited[y, x + 1]:
                    visited[y, x + 1] = True
                    q.append((y, x + 1))
            if count >= min_area:
                rects.append((min_x, min_y,
                              max_x - min_x + 1, max_y - min_y + 1, count))
    return rects


def sample_bg_color(
    img: np.ndarray, rx: int, ry: int, rw: int, rh: int, margin: int,
) -> np.ndarray:
    """박스 바깥 ring 영역 픽셀의 중앙값을 배경색(RGB)으로 반환."""
    h, w = img.shape[:2]
    y1 = max(0, ry - margin)
    y2 = min(h, ry + rh + margin)
    x1 = max(0, rx - margin)
    x2 = min(w, rx + rw + margin)
    outer = img[y1:y2, x1:x2]
    mask = np.ones((y2 - y1, x2 - x1), dtype=bool)
    mask[ry - y1: (ry + rh) - y1, rx - x1: (rx + rw) - x1] = False
    samples = outer[mask]
    if samples.size == 0:
        return img[ry:ry + rh, rx:rx + rw].astype(np.float32).reshape(-1, 3).mean(axis=0)
    return np.median(samples.astype(np.float32), axis=0)


def detect_text_pixels(
    box: np.ndarray, bg: np.ndarray, threshold: float,
) -> np.ndarray:
    """박스 내부에서 배경과의 RGB 색차가 threshold 이상인 픽셀을 True 로."""
    diff = box.astype(np.float32) - bg
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return dist >= threshold


def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    """numpy 전용 사각형 dilation. 의존성 추가 없음."""
    if radius <= 0 or not mask.any():
        return mask
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            y1 = max(0, dy)
            y2 = min(h, h + dy)
            x1 = max(0, dx)
            x2 = min(w, w + dx)
            sy1 = max(0, -dy)
            sy2 = sy1 + (y2 - y1)
            sx1 = max(0, -dx)
            sx2 = sx1 + (x2 - x1)
            out[y1:y2, x1:x2] |= mask[sy1:sy2, sx1:sx2]
    return out


def inpaint_with_mask(
    image_bytes: bytes,
    mask_png: bytes,
    *,
    radii: tuple[int, ...] = DEFAULT_INPAINT_RADII,
) -> bytes:
    """mask_png (alpha=0 영역) 을 주변 색으로 자연스럽게 메꾼 RGB PNG bytes 반환.

    의도: 6단계 글자 제거 API 에 보낼 입력을 미리 깨끗한 배경으로 시드해서
          모델이 글자를 새로 상상해 그리는 위험을 줄인다.

    알고리즘 (PIL + numpy 만 사용):
      1) 마스크 안을 일단 큰 블러 색으로 1차 채움 (시드)
      2) 큰 반경 → 작은 반경 순으로 Gaussian blur 반복.
         매 반복마다 "마스크 안" 만 블러값으로 교체, 밖은 원본 유지.
         결과적으로 마스크 외곽 원본 픽셀이 점진적으로 안쪽으로 번지면서
         자연스럽게 메꿔진다.
    """
    pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    mask_img = Image.open(BytesIO(mask_png)).convert("RGBA")
    if mask_img.size != pil.size:
        mask_img = mask_img.resize(pil.size, Image.NEAREST)

    alpha = np.array(mask_img.split()[-1])
    fill_mask = (alpha == 0)
    if not fill_mask.any():
        # 메꿀 영역이 없으면 원본을 그대로 PNG 로 인코딩해 반환
        bio = BytesIO()
        pil.save(bio, format="PNG")
        return bio.getvalue()

    img = np.array(pil, dtype=np.float32)
    mask3 = fill_mask.astype(np.float32)[..., None]
    keep3 = 1.0 - mask3
    orig = img.copy()

    # 1차 시드: 큰 블러 색을 마스크 안에 채워 본 색상 단서 제공
    seed = np.array(
        pil.filter(ImageFilter.GaussianBlur(radius=40)),
        dtype=np.float32,
    )
    work = orig * keep3 + seed * mask3

    # 반복 블러 — 마스크 외곽 원본 픽셀이 점진적으로 침투
    for r in radii:
        work_pil = Image.fromarray(np.clip(work, 0, 255).astype(np.uint8))
        blurred = np.array(
            work_pil.filter(ImageFilter.GaussianBlur(radius=r)),
            dtype=np.float32,
        )
        work = orig * keep3 + blurred * mask3

    out = np.clip(work, 0, 255).astype(np.uint8)
    bio = BytesIO()
    Image.fromarray(out, mode="RGB").save(bio, format="PNG")
    return bio.getvalue()


def refine_mask_to_text_pixels(
    image_bytes: bytes,
    coarse_mask_png: bytes,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    dilate: int = DEFAULT_DILATE,
    bg_margin: int = DEFAULT_BG_MARGIN,
    min_area: int = DEFAULT_MIN_AREA,
) -> tuple[bytes, dict[str, Any]]:
    """원본 이미지 + 거친 사각 마스크 → 글자 픽셀만 alpha=0 인 세부 마스크 PNG bytes.

    반환:
        (refined_png_bytes, info_dict)
        info_dict 에는 component_count / text_pixel_count / coarse_pixel_count / ratio_percent
    """
    img_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = np.array(img_pil)
    h, w = img.shape[:2]

    mask_pil = Image.open(BytesIO(coarse_mask_png)).convert("RGBA")
    if mask_pil.size != (w, h):
        mask_pil = mask_pil.resize((w, h), Image.NEAREST)
    alpha = np.array(mask_pil.split()[-1])
    coarse = (alpha == 0)
    coarse_pixels = int(coarse.sum())

    rects = find_mask_components(coarse, min_area)

    text_mask = np.zeros((h, w), dtype=bool)
    for (rx, ry, rw, rh, _cnt) in rects:
        bg = sample_bg_color(img, rx, ry, rw, rh, bg_margin)
        box = img[ry:ry + rh, rx:rx + rw]
        comp_mask = coarse[ry:ry + rh, rx:rx + rw]
        text_in_box = detect_text_pixels(box, bg, threshold) & comp_mask
        if dilate > 0:
            text_in_box = dilate_bool(text_in_box, dilate) & comp_mask
        text_mask[ry:ry + rh, rx:rx + rw] |= text_in_box

    alpha_out = np.where(text_mask, 0, 255).astype(np.uint8)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = alpha_out
    bio = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(bio, format="PNG")
    text_pixels = int(text_mask.sum())
    info = {
        "component_count": len(rects),
        "text_pixel_count": text_pixels,
        "coarse_pixel_count": coarse_pixels,
        "ratio_percent": round(text_pixels / max(1, coarse_pixels) * 100, 2),
        "image_width": w,
        "image_height": h,
    }
    return bio.getvalue(), info
