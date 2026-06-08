from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


class LayoutParseError(ValueError):
    pass


@dataclass(slots=True)
class LayoutBlock:
    bbox_2d: tuple[int, int, int, int]
    label: str


MINERU_BLOCK_TYPES = {
    "text",
    "title",
    "table",
    "equation",
    "formula_number",
    "code",
    "algorithm",
    "aside_text",
    "ref_text",
    "phonetic",
    "list_item",
    "table_caption",
    "image_caption",
    "code_caption",
    "table_footnote",
    "image_footnote",
    "header",
    "footer",
    "page_number",
    "page_footnote",
    "image",
    "chart",
    "list",
    "image_block",
    "equation_block",
    "unknown",
}

LABEL_MAP = {
    "normal_text": "text",
    "plain_text": "text",
    "paragraph": "text",
    "body": "text",
    "content": "text",
    "abstract": "text",
    "caption": "text",
    "subtitle": "text",
    "sub_title": "text",
    "section": "text",
    "logo": "header",
    "page_header": "header",
    "header_image": "header",
    "header": "header",
    "foot": "footer",
    "page_footer": "footer",
    "footer_image": "footer",
    "footer": "footer",
    "title": "title",
    "doc_title": "title",
    "document_title": "title",
    "paragraph_title": "title",
    "section_title": "title",
    "heading": "title",
    "page_no": "page_number",
    "page_number": "page_number",
    "pagenumber": "page_number",
    "page_num": "page_number",
    "page": "page_number",
    "table": "table",
    "table_body": "table",
    "table_region": "table",
    "table_title": "table_caption",
    "table_caption": "table_caption",
    "table_note": "table_footnote",
    "table_footnote": "table_footnote",
    "image": "image",
    "img": "image",
    "figure": "image",
    "picture": "image",
    "image_body": "image",
    "figure_body": "image",
    "figure_caption": "image_caption",
    "image_caption": "image_caption",
    "figure_note": "image_footnote",
    "image_footnote": "image_footnote",
    "chart": "chart",
    "plot": "chart",
    "graph": "chart",
    "chart_body": "chart",
    "plot": "chart",
    "formula": "equation",
    "equation": "equation",
    "interline_equation": "equation",
    "display_formula": "equation",
    "math": "equation",
    "inline_formula": "equation",
    "equation_number": "formula_number",
    "formula_number": "formula_number",
    "equation_block": "equation_block",
    "formula_block": "equation_block",
    "code": "code",
    "code_body": "code",
    "source_code": "code",
    "code_caption": "code_caption",
    "algorithm": "algorithm",
    "aside": "aside_text",
    "aside_text": "aside_text",
    "side_note": "aside_text",
    "marginalia": "aside_text",
    "reference": "ref_text",
    "references": "ref_text",
    "ref": "ref_text",
    "ref_text": "ref_text",
    "bibliography": "ref_text",
    "phonetic": "phonetic",
    "list_item": "list_item",
    "bullet": "list_item",
    "bullet_item": "list_item",
    "numbered_item": "list_item",
    "list": "text",
    "list_block": "list",
    "page_footnote": "page_footnote",
    "footnote": "page_footnote",
    "unknown": "text",
}

DEDUP_IOU_THRESHOLD = 0.92


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    fence = re.fullmatch(r"```[a-zA-Z0-9_-]*\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()

    first_fence = re.search(r"```[a-zA-Z0-9_-]*\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if first_fence:
        return first_fence.group(1).strip()
    return stripped


def layout_json_to_mineru_tags(raw_text: str, image_size: tuple[int, int] | None = None) -> str:
    payload = parse_layout_json(raw_text)
    blocks = [_coerce_block(item, image_size) for item in payload]
    valid_blocks = _sort_blocks(_deduplicate_blocks([block for block in blocks if block is not None]))
    return "\n".join(
        f"<|box_start|>{x1} {y1} {x2} {y2}<|box_end|><|ref_start|>{block.label}<|ref_end|><|rotate_up|>"
        for block in valid_blocks
        for x1, y1, x2, y2 in [block.bbox_2d]
    )


def parse_layout_json(raw_text: str) -> list[Any]:
    cleaned = strip_markdown_fence(raw_text)
    try:
        loaded = json.loads(cleaned)
    except json.JSONDecodeError:
        array_text = _extract_first_json_array(cleaned)
        if array_text is None:
            raise LayoutParseError("Upstream layout output does not contain a JSON array")
        try:
            loaded = json.loads(array_text)
        except json.JSONDecodeError as exc:
            raise LayoutParseError(f"Upstream layout JSON is invalid: {exc}") from exc

    if isinstance(loaded, dict):
        for key in ("blocks", "layout", "items", "regions"):
            value = loaded.get(key)
            if isinstance(value, list):
                return value
        raise LayoutParseError("Upstream layout JSON object does not contain a blocks/layout/items/regions list")
    if not isinstance(loaded, list):
        raise LayoutParseError("Upstream layout JSON root is not a list")
    return loaded


def _extract_first_json_array(text: str) -> str | None:
    start = text.find("[")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _coerce_block(item: Any, image_size: tuple[int, int] | None) -> LayoutBlock | None:
    if not isinstance(item, dict):
        return None
    bbox = item.get("bbox_2d") or item.get("bbox") or item.get("box") or item.get("coordinates")
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return None
    try:
        coords = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None

    normalized = _normalize_bbox(coords, image_size)
    if normalized is None:
        return None
    label = normalize_label(str(item.get("label") or item.get("type") or "text"))
    return LayoutBlock(normalized, label)


def _normalize_bbox(coords: list[float], image_size: tuple[int, int] | None) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = coords
    if max(coords) <= 1.0 and min(coords) >= 0.0:
        x1, y1, x2, y2 = [coord * 1000 for coord in coords]
    elif max(coords) > 1000:
        if image_size:
            width, height = image_size
            if width > 0 and height > 0:
                x1, x2 = x1 / width * 1000, x2 / width * 1000
                y1, y2 = y1 / height * 1000, y2 / height * 1000
        else:
            max_x = max(abs(x1), abs(x2), 1.0)
            max_y = max(abs(y1), abs(y2), 1.0)
            x1, x2 = x1 / max_x * 1000, x2 / max_x * 1000
            y1, y2 = y1 / max_y * 1000, y2 / max_y * 1000

    x1, x2 = sorted((_clamp_round(x1), _clamp_round(x2)))
    y1, y2 = sorted((_clamp_round(y1), _clamp_round(y2)))
    if x1 == x2 or y1 == y2:
        return None
    return x1, y1, x2, y2


def _clamp_round(value: float) -> int:
    return max(0, min(1000, int(round(value))))


def normalize_label(label: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    if key in MINERU_BLOCK_TYPES and key != "unknown":
        return key
    normalized = LABEL_MAP.get(key, "text")
    return normalized if normalized in MINERU_BLOCK_TYPES else "text"


def _sort_blocks(blocks: list[LayoutBlock]) -> list[LayoutBlock]:
    return sorted(blocks, key=lambda block: (block.bbox_2d[1] // 20, block.bbox_2d[0], block.bbox_2d[1]))


def _deduplicate_blocks(blocks: list[LayoutBlock]) -> list[LayoutBlock]:
    result: list[LayoutBlock] = []
    for block in blocks:
        if any(block.label == existing.label and _iou(block.bbox_2d, existing.bbox_2d) >= DEDUP_IOU_THRESHOLD for existing in result):
            continue
        result.append(block)
    return result


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0
