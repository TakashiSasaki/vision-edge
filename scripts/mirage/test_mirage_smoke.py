#!/usr/bin/env python3
"""
Phase 1 Smoke Test & CLI Bridge for Cactus Compute Mirage (51.3M / ~21.6MB .cact)
Repository: https://huggingface.co/Cactus-Compute/mirage
Library: cactus-compute (Apache-2.0)

Usage:
  1. Standalone smoke test:
     python3 scripts/mirage/test_mirage_smoke.py
  2. JSON CLI bridge for MirageVisionAdapter / benchmark:
     python3 scripts/mirage/test_mirage_smoke.py --image <path> [--mode describe|extract|both] [--form-json <json>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

EXPECTED_MIRAGE_REVISION = "76c1e9e53381aca8fc3843c636216c96ff220122"
EXPECTED_MIRAGE_BYTES = 21611043
EXPECTED_MIRAGE_SHA256 = "4249a0f7515272ff76b9414bec17d7e7daf84b15b66c7fb4f17fce1e66f9f237"

DEFAULT_SAMPLE_IMAGE = Path("public/assets/aistudio/image_c2b3e52e.png")

DEFAULT_DECISIONS_FORM: Dict[str, Any] = {
    "scene_domain": {
        "enum": [
            "application_ui_screenshot",
            "natural_photography",
            "document_or_form",
            "industrial_inspection",
        ]
    },
    "primary_subject": {
        "enum": [
            "person",
            "vehicle",
            "animal",
            "software_interface",
            "food_or_beverage",
        ]
    },
    "contains_readable_text": {
        "enum": ["yes", "no"]
    },
    "person_count": {
        "type": "integer",
        "minimum": 0,
        "maximum": 10,
    },
}


def resolve_cached_mirage_weights() -> Optional[Path]:
    """Return local cached mirage.cact path if present to avoid unnecessary hub network lookups."""
    candidates = [
        Path.home()
        / ".cache"
        / "cactus"
        / "models"
        / "mirage"
        / EXPECTED_MIRAGE_REVISION
        / "mirage.cact",
        Path("/root/.cache/cactus/models/mirage")
        / EXPECTED_MIRAGE_REVISION
        / "mirage.cact",
    ]
    env_override = os.environ.get("MIRAGE_WEIGHTS_PATH")
    if env_override:
        candidates.insert(0, Path(env_override))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # Search any revision under ~/.cache/cactus/models/mirage
    cache_root = Path.home() / ".cache" / "cactus" / "models" / "mirage"
    if cache_root.is_dir():
        for found in sorted(cache_root.rglob("mirage.cact")):
            if found.is_file():
                return found
    return None


def compute_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(usage / (1024 * 1024), 2)
    return round(usage / 1024.0, 2)


import re

_NAN_TOKEN_RE = re.compile(r"(:\s*)-?nan\b", re.IGNORECASE)


def _parse_mirage_c_buffer(vision: Any) -> Dict[str, Any]:
    raw_text = vision._buffer.value.decode("utf-8", errors="replace")
    sanitized = _NAN_TOKEN_RE.sub(r"\g<1>0.0", raw_text)
    return json.loads(sanitized)


def safe_mirage_describe(
    vision: Any, image: Any, **kwargs: Any
) -> Tuple[Dict[str, Any], bool]:
    try:
        return vision.describe(image, **kwargs), False
    except json.JSONDecodeError:
        return _parse_mirage_c_buffer(vision), True


def safe_mirage_extract(
    vision: Any, image: Any, form: Dict[str, Any], **kwargs: Any
) -> Tuple[Dict[str, Any], bool]:
    try:
        return vision.extract(image, form, **kwargs), False
    except json.JSONDecodeError:
        return _parse_mirage_c_buffer(vision), True


def run_mirage_inference(
    image_path: Path,
    mode: str = "both",
    form_schema: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import cactus  # type: ignore

    if not image_path.is_file():
        raise FileNotFoundError(f"Sample image not found: {image_path}")

    cached_weights = resolve_cached_mirage_weights()
    rss_before_mb = get_peak_rss_mb()

    t_init_start = time.perf_counter()
    if cached_weights is not None:
        vision = cactus.Mirage(weights=str(cached_weights))
    else:
        vision = cactus.Mirage()
        cached_weights = resolve_cached_mirage_weights()
    init_ms = round((time.perf_counter() - t_init_start) * 1000.0, 2)

    artifact_meta: Dict[str, Any] = {
        "cactusVersion": getattr(cactus, "__version__", "unknown"),
        "weightsPath": str(cached_weights) if cached_weights else None,
        "weightsBytes": cached_weights.stat().st_size if cached_weights and cached_weights.exists() else None,
        "weightsSha256": compute_sha256(cached_weights) if cached_weights and cached_weights.exists() else None,
    }

    describe_result: Optional[Dict[str, Any]] = None
    describe_ms: Optional[float] = None
    extract_result: Optional[Dict[str, Any]] = None
    extract_ms: Optional[float] = None
    sanitized_c_nan = False

    if mode in ("describe", "both"):
        t_desc_start = time.perf_counter()
        describe_result, did_sanitize = safe_mirage_describe(
            vision, str(image_path), options=options
        )
        sanitized_c_nan = sanitized_c_nan or did_sanitize
        describe_ms = round((time.perf_counter() - t_desc_start) * 1000.0, 2)

    if mode in ("extract", "both"):
        schema = form_schema if form_schema is not None else DEFAULT_DECISIONS_FORM
        t_ext_start = time.perf_counter()
        extract_result, did_sanitize = safe_mirage_extract(
            vision, str(image_path), schema, options={"sections": []}
        )
        sanitized_c_nan = sanitized_c_nan or did_sanitize
        extract_ms = round((time.perf_counter() - t_ext_start) * 1000.0, 2)

    if hasattr(vision, "close") and callable(getattr(vision, "close")):
        vision.close()
    del vision
    rss_after_mb = get_peak_rss_mb()

    return {
        "status": "ok",
        "imagePath": str(image_path),
        "sanitizedCBufferNan": sanitized_c_nan,
        "artifact": artifact_meta,
        "metrics": {
            "initMs": init_ms,
            "describeMs": describe_ms,
            "extractMs": extract_ms,
            "rssBeforeMb": rss_before_mb,
            "peakRssMb": rss_after_mb,
            "rssDeltaMb": round(max(0.0, rss_after_mb - rss_before_mb), 2),
        },
        "describe": describe_result,
        "extract": extract_result,
    }


def validate_smoke_payload(payload: Dict[str, Any]) -> None:
    artifact = payload.get("artifact") or {}
    if artifact.get("weightsBytes") != EXPECTED_MIRAGE_BYTES:
        raise AssertionError(
            f"Unexpected mirage.cact size: {artifact.get('weightsBytes')} != {EXPECTED_MIRAGE_BYTES}"
        )
    if artifact.get("weightsSha256") != EXPECTED_MIRAGE_SHA256:
        raise AssertionError(
            f"Unexpected mirage.cact sha256: {artifact.get('weightsSha256')} != {EXPECTED_MIRAGE_SHA256}"
        )

    desc = payload.get("describe")
    if not isinstance(desc, dict):
        raise AssertionError("vision.describe() did not return a dictionary")

    required_sections = ("scene", "objects", "regions", "text", "relations", "events", "reasoning")
    for section in required_sections:
        if section not in desc:
            raise AssertionError(f"Missing required section '{section}' in vision.describe() output")

    objects = desc.get("objects")
    if not isinstance(objects, list) or len(objects) == 0:
        raise AssertionError("Expected non-empty 'objects' list in vision.describe() output")

    for obj in objects:
        if "name" not in obj or "bbox" not in obj or "score" not in obj:
            raise AssertionError(f"Invalid object entry in vision.describe(): {obj}")
        bbox = obj["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise AssertionError(f"Invalid bbox coordinates in object: {bbox}")

    text_items = desc.get("text")
    if not isinstance(text_items, list) or len(text_items) == 0:
        raise AssertionError("Expected non-empty 'text' (OCR) list in vision.describe() output")

    ext = payload.get("extract")
    if not isinstance(ext, dict) or "form" not in ext:
        raise AssertionError("vision.extract() did not return a 'form' dictionary")

    form_out = ext["form"]
    for key in DEFAULT_DECISIONS_FORM:
        if key not in form_out:
            raise AssertionError(f"Missing Decisions field '{key}' in vision.extract() output")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mirage smoke test and CLI bridge")
    parser.add_argument(
        "--image",
        type=str,
        default=str(DEFAULT_SAMPLE_IMAGE),
        help="Path to input image",
    )
    parser.add_argument(
        "--mode",
        choices=["describe", "extract", "both"],
        default="both",
        help="Inference mode to run",
    )
    parser.add_argument(
        "--form-json",
        type=str,
        default=None,
        help="Optional JSON string specifying Decisions form schema",
    )
    parser.add_argument(
        "--options-json",
        type=str,
        default=None,
        help="Optional JSON string specifying describe() options",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only the raw JSON output to stdout (for programmatic adapter calls)",
    )
    args = parser.parse_args()

    form_schema = json.loads(args.form_json) if args.form_json else None
    options = json.loads(args.options_json) if args.options_json else None

    payload = run_mirage_inference(
        image_path=Path(args.image),
        mode=args.mode,
        form_schema=form_schema,
        options=options,
    )

    if args.json_only:
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    validate_smoke_payload(payload)

    desc = payload["describe"] or {}
    ext_form = (payload["extract"] or {}).get("form", {})
    metrics = payload["metrics"]
    artifact = payload["artifact"]

    summary = {
        "smokeTest": "PASS",
        "cactusVersion": artifact["cactusVersion"],
        "weightsBytes": artifact["weightsBytes"],
        "weightsSha256": artifact["weightsSha256"],
        "imagePath": payload["imagePath"],
        "metrics": metrics,
        "detectedObjectsCount": len(desc.get("objects", [])),
        "detectedObjects": [
            {
                "id": o.get("id"),
                "name": o.get("name"),
                "score": o.get("score"),
                "bbox": o.get("bbox"),
                "maskVertices": len(o.get("mask", [])) // 2,
            }
            for o in desc.get("objects", [])
        ],
        "ocrTextLinesCount": len(desc.get("text", [])),
        "ocrSample": [t.get("string") for t in desc.get("text", [])[:5]],
        "relationsCount": len(desc.get("relations", [])),
        "scene": desc.get("scene"),
        "decisionsForm": ext_form,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
