"""Parsing and validating native GTSDB bounding-box annotations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GTSDBAnnotation:
    """One traffic-sign box in native image-pixel coordinates."""

    filename: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    class_id: int


def read_gtsdb_annotations(annotation_file: str | Path) -> list[GTSDBAnnotation]:
    """Read the official semicolon-delimited six-column GTSDB annotation file.

    Expected columns: filename, left, top, right, bottom, class_id.
    Header rows are allowed. Malformed data rows fail loudly instead of
    silently producing incorrect training labels.
    """
    annotations: list[GTSDBAnnotation] = []
    path = Path(annotation_file)

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=";")
        for line_number, raw_row in enumerate(reader, start=1):
            row = [value.strip() for value in raw_row]
            if not row or all(not value for value in row):
                continue
            if row[0].lower() in {"filename", "image", "image_name"}:
                continue
            if len(row) != 6:
                raise ValueError(
                    f"{path}:{line_number}: expected 6 semicolon-separated "
                    f"fields, received {len(row)}"
                )
            try:
                x_min, y_min, x_max, y_max, class_id = map(int, row[1:])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: coordinates and class_id must be integers"
                ) from exc

            if not row[0]:
                raise ValueError(f"{path}:{line_number}: filename cannot be empty")
            if min(x_min, y_min) < 0 or x_max <= x_min or y_max <= y_min:
                raise ValueError(f"{path}:{line_number}: invalid bounding-box geometry")
            if not 0 <= class_id <= 42:
                raise ValueError(f"{path}:{line_number}: class_id must be in [0, 42]")

            annotations.append(
                GTSDBAnnotation(row[0], x_min, y_min, x_max, y_max, class_id)
            )

    return annotations


def to_yolo_xywh(
    annotation: GTSDBAnnotation, image_width: int, image_height: int
) -> tuple[int, float, float, float, float]:
    """Convert a pixel box to YOLO's normalized center-x, center-y, width, height."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if annotation.x_max > image_width or annotation.y_max > image_height:
        raise ValueError("Bounding box extends beyond the image dimensions")

    x_center = (annotation.x_min + annotation.x_max) / (2 * image_width)
    y_center = (annotation.y_min + annotation.y_max) / (2 * image_height)
    width = (annotation.x_max - annotation.x_min) / image_width
    height = (annotation.y_max - annotation.y_min) / image_height
    return annotation.class_id, x_center, y_center, width, height
