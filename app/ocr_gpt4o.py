from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import List

from openai import OpenAI

# 可选：优先提取“文本型 PDF”
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None  # type: ignore[assignment]

# 可选：扫描版 PDF 转图片后再 OCR
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore[assignment]


IMAGE_MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}
TEXT_SUFFIX = {".txt", ".md", ".csv", ".json", ".xml"}


def _to_data_url(image_bytes: bytes, mime: str) -> str:
    """把图片二进制转换为 data URL。"""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _read_chat_content(content) -> str:
    """兼容不同 SDK 返回格式，稳定拿到文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content).strip()


def _ocr_image_with_gpt4o(
    client: OpenAI,
    image_bytes: bytes,
    mime: str,
    model: str = "gpt-4o-mini",
) -> str:
    """使用 GPT-4o 识别单张图片文本。"""
    data_url = _to_data_url(image_bytes, mime)

    system_prompt = (
        "你是专业 OCR 引擎。"
        "任务是逐行转写图片中的文字。"
        "要求："
        "1) 保持原文顺序；"
        "2) 不要总结；"
        "3) 不要补充推断；"
        "4) 无法识别处写 [不清晰]；"
        "5) 只输出纯文本。"
    )

    user_content = [
        {"type": "text", "text": "请提取这张图片中的全部文字，按原有顺序输出。"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=3000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    return _read_chat_content(resp.choices[0].message.content)


def _extract_pdf_text_direct(pdf_bytes: bytes) -> str:
    """优先尝试直接提取文本型 PDF。"""
    if PdfReader is None:
        return ""

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts: List[str] = []
        for page in reader.pages:
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    except Exception:
        return ""


def _pdf_to_png_pages(pdf_bytes: bytes, max_pages: int = 8) -> List[bytes]:
    """扫描版 PDF 转 PNG（每页一张），供 GPT-4o OCR。"""
    if fitz is None:
        return []

    images: List[bytes] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = min(len(doc), max_pages)
        for i in range(page_count):
            page = doc[i]
            pix = page.get_pixmap(dpi=180, alpha=False)
            images.append(pix.tobytes("png"))
        doc.close()
    except Exception:
        return []

    return images


def _extract_pdf_with_gpt4o(
    client: OpenAI,
    pdf_bytes: bytes,
    model: str = "gpt-4o-mini",
    max_pages: int = 8,
) -> str:
    """
    PDF OCR 策略：
    1) 先直接抽取文本型 PDF
    2) 若文本太少，则按页转图片再用 GPT-4o OCR
    """
    direct_text = _extract_pdf_text_direct(pdf_bytes)

    # 文本型 PDF 直接抽取足够时，优先使用，成本最低
    if len(direct_text) >= 80:
        return direct_text

    page_images = _pdf_to_png_pages(pdf_bytes, max_pages=max_pages)
    if not page_images:
        return direct_text or "[PDF 无法解析：缺少可用解析器]"

    blocks: List[str] = []
    for idx, img in enumerate(page_images, start=1):
        page_text = _ocr_image_with_gpt4o(
            client=client,
            image_bytes=img,
            mime="image/png",
            model=model,
        )
        blocks.append(f"[第{idx}页]\n{page_text}")

    merged_ocr = "\n\n".join(blocks).strip()
    if direct_text:
        return f"{direct_text}\n\n{merged_ocr}".strip()
    return merged_ocr


def extract_text_from_upload_with_gpt4o(
    client: OpenAI,
    filename: str,
    content: bytes,
    model: str = "gpt-4o-mini",
    max_pdf_pages: int = 8,
) -> str:
    """
    上传文件统一文本提取入口：
    - txt/md/csv/json/xml：直接解码
    - image：GPT-4o OCR
    - pdf：先文本提取，再 GPT-4o OCR 扫描页
    """
    suffix = Path(filename).suffix.lower()

    if suffix in TEXT_SUFFIX:
        return content.decode("utf-8", errors="ignore").strip()

    if suffix in IMAGE_MIME_MAP:
        return _ocr_image_with_gpt4o(
            client=client,
            image_bytes=content,
            mime=IMAGE_MIME_MAP[suffix],
            model=model,
        )

    if suffix == ".pdf":
        return _extract_pdf_with_gpt4o(
            client=client,
            pdf_bytes=content,
            model=model,
            max_pages=max_pdf_pages,
        )

    return "[暂不支持的文件类型]"
