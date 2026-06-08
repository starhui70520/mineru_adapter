from __future__ import annotations

import base64
import copy
import io
import re
from enum import Enum
from typing import Any

from PIL import Image


class MinerUTask(str, Enum):
    layout = "layout"
    text = "text"
    table = "table"
    formula = "formula"
    image = "image"
    unknown = "unknown"


TASK_MARKERS: dict[MinerUTask, str] = {
    MinerUTask.layout: "Layout Detection:",
    MinerUTask.text: "Text Recognition:",
    MinerUTask.table: "Table Recognition:",
    MinerUTask.formula: "Formula Recognition:",
    MinerUTask.image: "Image Analysis:",
}

MINERU_LAYOUT_TYPES = (
    "text, title, table, equation, formula_number, code, algorithm, aside_text, "
    "ref_text, phonetic, list_item, table_caption, image_caption, code_caption, "
    "table_footnote, image_footnote, header, footer, page_number, page_footnote, "
    "image, chart, list, image_block, equation_block, unknown"
)

TASK_PROMPTS: dict[MinerUTask, str] = {
    MinerUTask.layout: (
        "Layout Detection:\n"
        "Detect visible document layout blocks in reading order. Return only a JSON array, "
        "with no Markdown fence and no explanation. Each item must have this exact shape: "
        '{"bbox_2d":[x1,y1,x2,y2],"type":"text"}. '
        "bbox_2d may be image pixel coordinates, normalized 0-1000 coordinates, or normalized "
        "0-1 coordinates. Use only these MinerU block types: "
        f"{MINERU_LAYOUT_TYPES}. "
        "Do not invent other type names. Prefer title for document/section headings, table for "
        "table bodies, image/chart for figures, equation for display formulas, header/footer/"
        "page_number for repeated page furniture, and text for normal paragraphs. Include all "
        "major readable regions, but do not split every text line into a separate block unless "
        "the page visually separates them."
    ),
    MinerUTask.text: (
        "Text Recognition:\n"
        "Recognize all visible text in reading order. Return only plain text. Do not wrap the "
        "answer in Markdown code fences and do not explain your reasoning."
    ),
    MinerUTask.table: (
        "Table Recognition:\n"
        "Recognize the table in the image. Return only HTML table markup. Do not wrap the "
        "answer in Markdown code fences and do not include explanation."
    ),
    MinerUTask.formula: (
        "Formula Recognition:\n"
        "Recognize the formula in the image. Return only the LaTeX string. Do not wrap the "
        "answer in Markdown code fences and do not include explanation."
    ),
    MinerUTask.image: (
        "Image Analysis:\n"
        "Describe the image or chart content concisely for document extraction. Return only "
        "plain text without Markdown code fences."
    ),
}


def extract_text_from_messages(messages: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
    return "\n".join(chunks)


def detect_task(messages: list[dict[str, Any]]) -> MinerUTask:
    text = extract_text_from_messages(messages)
    for task, marker in TASK_MARKERS.items():
        if marker in text:
            return task
    return MinerUTask.unknown


def rewrite_messages_for_task(messages: list[dict[str, Any]], task: MinerUTask) -> list[dict[str, Any]]:
    prompt = TASK_PROMPTS.get(task)
    if not prompt:
        return copy.deepcopy(messages)

    rewritten = copy.deepcopy(messages)
    for message in reversed(rewritten):
        if message.get("role") != "user":
            continue
        if _replace_first_text_part(message, prompt):
            return rewritten
        message["content"] = [{"type": "text", "text": prompt}, {"type": "text", "text": ""}]
        return rewritten
    rewritten.append({"role": "user", "content": prompt})
    return rewritten


def _replace_first_text_part(message: dict[str, Any], prompt: str) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = prompt
        return True
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = prompt
                return True
        content.insert(0, {"type": "text", "text": prompt})
        return True
    return False


def first_image_size(messages: list[dict[str, Any]]) -> tuple[int, int] | None:
    for url in _iter_image_urls(messages):
        size = _image_size_from_data_url(url)
        if size:
            return size
    return None


def downsample_data_url_images(messages: list[dict[str, Any]], max_side: int) -> tuple[list[dict[str, Any]], tuple[int, int] | None]:
    rewritten = copy.deepcopy(messages)
    first_size: tuple[int, int] | None = None
    if max_side <= 0:
        return rewritten, first_image_size(rewritten)

    for message in rewritten:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if isinstance(image_url, str):
                resized_url, size = _downsample_data_url(image_url, max_side)
                part["image_url"] = resized_url
            elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                resized_url, size = _downsample_data_url(image_url["url"], max_side)
                image_url["url"] = resized_url
            else:
                size = None
            if first_size is None and size:
                first_size = size
    return rewritten, first_size


def _iter_image_urls(messages: list[dict[str, Any]]):
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if isinstance(image_url, str):
                yield image_url
            elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                yield image_url["url"]


_DATA_URL_RE = re.compile(r"^data:image/[^;]+;base64,(?P<data>.+)$", re.IGNORECASE | re.DOTALL)


def _image_size_from_data_url(url: str) -> tuple[int, int] | None:
    match = _DATA_URL_RE.match(url.strip())
    if not match:
        return None
    try:
        raw = base64.b64decode(match.group("data"), validate=False)
        with Image.open(io.BytesIO(raw)) as image:
            return image.size
    except Exception:
        return None


def _downsample_data_url(url: str, max_side: int) -> tuple[str, tuple[int, int] | None]:
    match = _DATA_URL_RE.match(url.strip())
    if not match:
        return url, None
    try:
        raw = base64.b64decode(match.group("data"), validate=False)
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            width, height = image.size
            if width <= 0 or height <= 0:
                return url, None
            if max(width, height) <= max_side:
                return url, (width, height)

            resized = image.copy()
            resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            resized.save(output, format="PNG", optimize=True)
            encoded = base64.b64encode(output.getvalue()).decode("ascii")
            return f"data:image/png;base64,{encoded}", resized.size
    except Exception:
        return url, None
