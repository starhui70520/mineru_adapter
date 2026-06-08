from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from mineru_vl_utils import MinerUClient


def run(image_path: Path, server_url: str, model: str) -> dict:
    client = MinerUClient(backend="http-client", model_name=model, server_url=server_url)
    with Image.open(image_path) as image:
        result = client.layout_detect(image.convert("RGB"))
    blocks = getattr(result, "blocks", result)
    return {"image": str(image_path), "block_count": len(blocks), "blocks": [dict(block) for block in blocks]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MinerUClient.layout_detect through the adapter.")
    parser.add_argument("--adapter-url", default="http://127.0.0.1:18000")
    parser.add_argument("--model", default="vl-model")
    parser.add_argument("--image", required=True, type=Path)
    args = parser.parse_args()

    result = run(args.image, args.adapter_url, args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
