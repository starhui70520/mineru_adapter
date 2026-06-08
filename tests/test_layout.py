from __future__ import annotations

import pytest

from mineru_adapter.layout import LayoutParseError, layout_json_to_mineru_tags, normalize_label, parse_layout_json, strip_markdown_fence


def test_strip_markdown_fence() -> None:
    assert strip_markdown_fence("```json\n[{\"a\": 1}]\n```") == '[{"a": 1}]'


def test_layout_json_to_mineru_tags_basic() -> None:
    raw = """
    ```json
    [
      {"bbox_2d": [146, 123, 429, 180], "label": "logo"},
      {"bbox_2d": [146, 420, 658, 452], "label": "title"},
      {"bbox_2d": [185, 766, 293, 810], "label": "subtitle"}
    ]
    ```
    """
    assert layout_json_to_mineru_tags(raw) == "\n".join(
        [
            "<|box_start|>146 123 429 180<|box_end|><|ref_start|>header<|ref_end|><|rotate_up|>",
            "<|box_start|>146 420 658 452<|box_end|><|ref_start|>title<|ref_end|><|rotate_up|>",
            "<|box_start|>185 766 293 810<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>",
        ]
    )


def test_layout_json_to_mineru_tags_normalizes_pixel_bbox() -> None:
    raw = '[{"bbox_2d": [100, 200, 900, 1800], "label": "table"}]'
    assert (
        layout_json_to_mineru_tags(raw, image_size=(1000, 2000))
        == "<|box_start|>100 100 900 900<|box_end|><|ref_start|>table<|ref_end|><|rotate_up|>"
    )


def test_layout_json_to_mineru_tags_normalizes_ratio_bbox() -> None:
    raw = '[{"bbox_2d": [0.1, 0.2, 0.9, 0.8], "type": "image"}]'
    assert (
        layout_json_to_mineru_tags(raw)
        == "<|box_start|>100 200 900 800<|box_end|><|ref_start|>image<|ref_end|><|rotate_up|>"
    )


def test_layout_json_to_mineru_tags_defaults_missing_label_to_text() -> None:
    raw = '[{"bbox_2d": [1, 2, 3, 4]}]'
    assert (
        layout_json_to_mineru_tags(raw)
        == "<|box_start|>1 2 3 4<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>"
    )


def test_layout_json_to_mineru_tags_sorts_and_deduplicates_shape() -> None:
    raw = """
    [
      {"bbox_2d": [100, 400, 500, 500], "type": "text"},
      {"bbox_2d": [100, 100, 500, 180], "type": "doc_title"},
      {"bbox_2d": [101, 101, 501, 181], "type": "paragraph_title"}
    ]
    """
    assert layout_json_to_mineru_tags(raw) == "\n".join(
        [
            "<|box_start|>100 100 500 180<|box_end|><|ref_start|>title<|ref_end|><|rotate_up|>",
            "<|box_start|>100 400 500 500<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>",
        ]
    )


def test_normalize_label_keeps_mineru_types_and_maps_synonyms() -> None:
    assert normalize_label("table_caption") == "table_caption"
    assert normalize_label("Display Formula") == "equation"
    assert normalize_label("Header Image") == "header"
    assert normalize_label("not-a-real-label") == "text"


def test_parse_layout_json_extracts_array_from_surrounding_text() -> None:
    assert parse_layout_json('Here is the JSON:\n[{"bbox_2d":[1,2,3,4],"label":"text"}]') == [
        {"bbox_2d": [1, 2, 3, 4], "label": "text"}
    ]


def test_parse_layout_json_rejects_invalid_json() -> None:
    with pytest.raises(LayoutParseError):
        parse_layout_json("not json")
