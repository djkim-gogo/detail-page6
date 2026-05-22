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
      const lines = text.split(/\r?\n/);
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

function renderLabelsAsDOM(b64, labels, container, opts) {
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
    if (opts && opts.maxWidth) wrap.style.maxWidth = opts.maxWidth;
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
    // 페이지 완료 → 다음 페이지 생성 컨트롤을 progressBox 맨 아래에 갱신
    maybeAddNextPageControls($("progressBox"));
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
    if (baseStep === "image_gen_text") {
      if (Array.isArray(sd.available_options) && sd.available_options.length) {
        const row = stepRows[stepKey];
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
      if (sd.text_prompt) {
        const ta = appendEditableStepDetail(stepKey,
          "글자 제거 프롬프트 (편집 가능):",
          sd.text_prompt);
        if (stepRows[stepKey]) stepRows[stepKey].promptEdit = ta;
      }
      // 페이지 N 6단계 재실행에 필요한 컨텍스트 저장
      stepInputs[stepKey] = {
        kind: "page-image-gen-text",
        pageNum: pageNum,
        image_b64: sd.base_image_b64 || null,
        first: { width_px: sd.width_px || 860, height_px: sd.height_px || 2000 },
        width_px: sd.width_px || 860,
        height_px: sd.height_px || 2000,
        mask_rects: Array.isArray(sd.mask_rects) ? sd.mask_rects : [],
      };
    }
    return true;
  }
  if (ev.event === "step_done") {
    // 재실행 버튼이 step_done 후 finishStep 에서 표시되도록 stepInputs 에 컨텍스트 저장.
    // image_gen2__pN: 페이지 N 의 4~7 단계 전체 재실행.
    // image_gen_text__pN: 페이지 N 의 6단계만 재실행 (step_start 에서 이미 저장됨).
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
      row.result.appendChild(note);
      if (d.stage_images) {
        renderStageImages(row.result, d.stage_images);
      } else {
        const thumb = document.createElement("img");
        thumb.src = "data:image/png;base64," + d.image_b64;
        thumb.style.maxWidth = "540px";
        thumb.style.marginTop = "4px";
        thumb.style.border = "1px solid #e5e7eb";
        thumb.style.borderRadius = "6px";
        row.result.appendChild(thumb);
      }
    } else if (baseStep === "overlay_refine") {
      const inputLines = d.input_lines || [];
      const base = d.base_image_b64;
      if (base && inputLines.length) {
        const lab = document.createElement("div");
        lab.style.cssText = "margin-top:6px;font-weight:600;font-size:13px";
        lab.textContent = "다듬기 전 OCR 정보로 합성한 결과 미리보기:";
        row.result.appendChild(lab);
        const preview = document.createElement("div");
        preview.style.cssText = "margin-top:4px;display:inline-block;border:1px solid #e5e7eb;border-radius:6px;overflow:hidden";
        row.result.appendChild(preview);
        renderLabelsAsDOM(base, inputLines, preview, { maxWidth: "540px" });
      }
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
  if (remaining >= 3) opts.push({ v: "3", t: "3" });
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
      // mask_rects 저장: 재실행 시 처음 생성과 동일한 마스크 사용
      if (Array.isArray(sd.mask_rects) && stepInputs.image_gen_text) {
        stepInputs.image_gen_text.mask_rects = sd.mask_rects;
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
        if (d.stage_images) {
          renderStageImages(row.result, d.stage_images);
        } else {
          const thumb = document.createElement("img");
          thumb.src = "data:image/png;base64," + d.image_b64;
          thumb.style.maxWidth = "540px";
          thumb.style.marginTop = "6px";
          thumb.style.border = "1px solid #e5e7eb";
          thumb.style.borderRadius = "6px";
          row.result.appendChild(thumb);
        }
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
    } else if (ev.step === "overlay_refine") {
      const row = stepRows["overlay_refine"];
      if (row) {
        row.result.style.display = "block";
        const inputLines = d.input_lines || [];
        const base = d.base_image_b64;
        if (base && inputLines.length) {
          const lab = document.createElement("div");
          lab.style.cssText = "margin-top:6px;font-weight:600;font-size:13px";
          lab.textContent = "다듬기 전 OCR 정보로 합성한 결과 미리보기:";
          row.result.appendChild(lab);
          const preview = document.createElement("div");
          preview.style.cssText = "margin-top:4px;display:inline-block;border:1px solid #e5e7eb;border-radius:6px;overflow:hidden";
          row.result.appendChild(preview);
          renderLabelsAsDOM(base, inputLines, preview, { maxWidth: "540px" });
        }
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
        // 8단계 완료 → 다음 페이지 생성 컨트롤을 progressBox 맨 아래에 추가
        maybeAddNextPageControls($("progressBox"));
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
  // 페이지 N 의 6단계(image_gen_text__pN) 만 재실행 — /api/step/image_gen_text 를 step_suffix 와 함께 호출.
  if (input.kind === "page-image-gen-text") {
    const pageNum = Number(input.pageNum || 0);
    if (!input.image_b64) {
      appendInfo("재실행 불가: 페이지 " + pageNum + " 의 4단계 이미지가 없습니다.", "error");
      return;
    }
    fd.append("image_b64", input.image_b64);
    fd.append("first_json", JSON.stringify(input.first || {}));
    fd.append("width_px", String(input.width_px || 860));
    fd.append("height_px", String(input.height_px || 2000));
    fd.append("step_suffix", "__p" + pageNum);
    fd.append("page", String(pageNum));
    if (Array.isArray(input.mask_rects)) {
      fd.append("mask_rects_json", JSON.stringify(input.mask_rects));
    }
    const promptTa = stepRows[step] && stepRows[step].promptEdit;
    const editedPrompt = promptTa ? promptTa.value : "";
    if ((editedPrompt || "").trim()) fd.append("prompt", editedPrompt);
    const modelSel = stepRows[step] && stepRows[step].modelSelect;
    if (modelSel) {
      const opt = modelSel.options[modelSel.selectedIndex];
      const chosenModel = (opt && (opt.dataset.model || "")) || "";
      const chosenQuality = (opt && (opt.dataset.quality || "")) || "";
      if (chosenModel.trim()) fd.append("model", chosenModel);
      if (chosenQuality.trim()) fd.append("quality", chosenQuality);
    }
    fetch("/api/step/image_gen_text", { method: "POST", body: fd })
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
    // 처음 생성 시 사용된 마스크 좌표를 그대로 전달 (재실행도 동일 마스크 사용)
    if (Array.isArray(input.mask_rects)) {
      fd.append("mask_rects_json", JSON.stringify(input.mask_rects));
    }
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

function getOrCreatePageContainer(pageNum) {
  const id = "page-container-" + pageNum;
  let container = document.getElementById(id);
  if (!container) {
    container = document.createElement("div");
    container.id = id;
    container.className = "page-container";
    container.style.cssText = "margin-top:16px;padding:12px;border:2px solid #cbd5e1;border-radius:8px;background:#f8fafc";
    const header = document.createElement("div");
    header.style.cssText = "font-weight:700;font-size:15px;color:#1e3a8a;margin-bottom:8px";
    header.textContent = "페이지 " + pageNum;
    container.appendChild(header);
    // 페이지 번호 순으로 정렬 삽입
    const siblings = Array.from($("progressBox").querySelectorAll(".page-container"));
    let inserted = false;
    for (const sib of siblings) {
      const sibNum = Number((sib.id || "").replace("page-container-", ""));
      if (sibNum > pageNum) {
        $("progressBox").insertBefore(container, sib);
        inserted = true;
        break;
      }
    }
    if (!inserted) $("progressBox").appendChild(container);
  }
  return container;
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
  // 페이지 접미사(__pN) 가 있으면 해당 페이지 컨테이너에 추가
  const m = step.match(/__p(\d+)$/);
  const targetBox = m ? getOrCreatePageContainer(Number(m[1])) : $("progressBox");
  targetBox.appendChild(line);
  const result = document.createElement("div");
  result.className = "step-result";
  result.style.display = "none";
  targetBox.appendChild(result);
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

function renderStageImages(targetEl, stages) {
  if (!targetEl || !stages) return;
  const items = [
    { key: "api_input_b64",    label: "① 원본 입력 (api_input_bytes_main)" },
    { key: "mask_b64",         label: "② 마스크 (mask_png_bytes)" },
    { key: "cropped_sent_b64", label: "③ 마스크 외 투명 처리 (API 에 전달)" },
    { key: "api_response_b64", label: "④ API 응답" },
    { key: "final_b64",        label: "⑤ 최종 (응답의 마스크 영역만 원본에 합성)" },
  ];
  const gallery = document.createElement("div");
  gallery.style.display = "flex";
  gallery.style.flexWrap = "wrap";
  gallery.style.gap = "12px";
  gallery.style.marginTop = "8px";
  for (const it of items) {
    const b64 = stages[it.key];
    if (!b64) continue;
    const cell = document.createElement("div");
    cell.style.flex = "0 0 auto";
    cell.style.maxWidth = "220px";
    cell.style.display = "flex";
    cell.style.flexDirection = "column";
    cell.style.alignItems = "center";
    cell.style.fontSize = "12px";
    cell.style.color = "#374151";
    const cap = document.createElement("div");
    cap.textContent = it.label;
    cap.style.fontWeight = "600";
    cap.style.marginBottom = "4px";
    cap.style.textAlign = "center";
    const img = document.createElement("img");
    img.src = "data:image/png;base64," + b64;
    img.style.maxWidth = "220px";
    img.style.maxHeight = "300px";
    img.style.border = "1px solid #e5e7eb";
    img.style.borderRadius = "6px";
    img.style.background = "#fff";
    img.style.objectFit = "contain";
    img.style.cursor = "zoom-in";
    img.addEventListener("click", () => {
      // data URL 을 Blob URL 로 변환 (브라우저가 큰 data URL 을 새 탭에서 못 여는 경우 대응)
      fetch(img.src)
        .then((r) => r.blob())
        .then((blob) => {
          const url = URL.createObjectURL(blob);
          const w = window.open(url, "_blank");
          if (!w) {
            // 팝업 차단 시 다운로드로 폴백
            const a = document.createElement("a");
            a.href = url;
            a.target = "_blank";
            a.rel = "noopener";
            a.click();
          }
        })
        .catch(() => window.open(img.src, "_blank"));
    });
    cell.appendChild(cap);
    cell.appendChild(img);
    gallery.appendChild(cell);
  }
  if (stages.bbox) {
    const bb = document.createElement("div");
    bb.textContent = "bbox = (x=" + stages.bbox[0] + ", y=" + stages.bbox[1] +
                     ", w=" + stages.bbox[2] + ", h=" + stages.bbox[3] + ")";
    bb.style.fontSize = "12px";
    bb.style.color = "#64748b";
    bb.style.marginTop = "6px";
    targetEl.appendChild(bb);
  }
  targetEl.appendChild(gallery);
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
    while ((idx = buf.indexOf("\n")) >= 0) {
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
  // '이전 결과 재사용' 체크 시 1~5단계는 캐시된 결과 사용
  const reuseEl = $("reusePrevResult");
  if (reuseEl && reuseEl.checked) {
    fd.append("use_cache", "1");
  }

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
