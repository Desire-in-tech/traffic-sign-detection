from pathlib import Path

import pytest

from traffic_sign_detection.annotations import (
    GTSDBAnnotation,
    read_gtsdb_annotations,
    to_yolo_xywh,
)


def test_reads_semicolon_delimited_gtsdb_rows_and_header(tmp_path: Path):
    annotation_file = tmp_path / "gt.txt"
    annotation_file.write_text(
        "Filename;Left;Top;Right;Bottom;ClassID\n"
        "00000.ppm;10;20;30;60;11\n",
        encoding="utf-8",
    )

    result = read_gtsdb_annotations(annotation_file)

    assert result == [GTSDBAnnotation("00000.ppm", 10, 20, 30, 60, 11)]


def test_converts_pixel_box_to_normalized_yolo_xywh():
    annotation = GTSDBAnnotation("scene.ppm", 10, 20, 30, 60, 11)

    result = to_yolo_xywh(annotation, image_width=100, image_height=100)

    assert result == (11, 0.2, 0.4, 0.2, 0.4)


@pytest.mark.parametrize(
    "row",
    [
        "scene.ppm;10;20;10;60;11",
        "scene.ppm;-1;20;30;60;11",
        "scene.ppm;10;20;30;60;43",
        "scene.ppm;10;20;30;60",
    ],
)
def test_rejects_invalid_annotation_rows(tmp_path: Path, row: str):
    annotation_file = tmp_path / "gt.txt"
    annotation_file.write_text(row + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_gtsdb_annotations(annotation_file)


def test_rejects_box_outside_image():
    annotation = GTSDBAnnotation("scene.ppm", 10, 20, 110, 60, 11)

    with pytest.raises(ValueError, match="extends beyond"):
        to_yolo_xywh(annotation, image_width=100, image_height=100)
