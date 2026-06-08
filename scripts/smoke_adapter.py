from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

import httpx


def image_part(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def completion_payload(model: str, prompt: str, image_path: Path) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, image_part(image_path)]}],
        "temperature": 0,
        "top_p": 0.01,
        "max_tokens": 2048,
        "vllm_xargs": {"no_repeat_ngram_size": 100},
        "skip_special_tokens": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the adapter against one image.")
    parser.add_argument("--adapter-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="vl-model")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--task", choices=["layout", "text"], default="layout")
    args = parser.parse_args()

    prompt = "\nLayout Detection:" if args.task == "layout" else "\nText Recognition:"
    payload = completion_payload(args.model, prompt, args.image)
    with httpx.Client(timeout=180) as client:
        response = client.post(f"{args.adapter_url.rstrip('/')}/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    print(json.dumps({"task": args.task, "content": content, "block_count": content.count("<|box_start|>")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
