# Traffic Sign Detection

YOLO-based traffic sign **object detection** on the German Traffic Sign Detection Benchmark (GTSDB). The project focuses on locating and classifying signs in complete road scenes, including small and visually challenging instances, then measuring the accuracy–latency trade-off for practical inference.

## Project status

**Task 8 — foundation in progress.** The repository structure and initial configuration are being established. Training metrics, trained weights, ONNX results, and video demonstrations will be added only after they have been generated and independently checked.

## Objective

Build an object detector that predicts a bounding box and traffic-sign class for each visible sign in a full-scene image. The project will evaluate both detection quality and inference speed rather than treating accuracy as the only success criterion.

## Dataset

The primary dataset is the [German Traffic Sign Detection Benchmark (GTSDB)](https://benchmark.ini.rub.de/gtsdb_dataset.html).

The official benchmark describes 900 full-scene images (600 training and 300 evaluation images). Images are PPM files; signs can be as small as 16 × 16 pixels. Ground-truth annotations are semicolon-separated and contain the image filename, bounding-box coordinates, and class ID.

We will inspect the exact archive and its ReadMe.txt before converting labels. In particular, we will distinguish the fine-grained class IDs from the benchmark's broader competition categories. We will not infer class names or annotation columns from a third-party Kaggle mirror.

**Data integrity:** Raw archives, extracted images, generated labels, model weights, and large video files will not be committed to Git. The notebook will document dataset provenance, checksums where practical, the official split, and any additional train/validation split.

## Planned workflow

1. Inspect the exact GTSDB archive and verify its annotation format and class-ID mapping.
2. Convert full-scene PPM images and bounding boxes into YOLO format, validating image dimensions, class IDs, and box geometry.
3. Preserve the official evaluation partition as a held-out test set. Split only the official training partition into training and validation subsets; use validation data for model selection and confidence-threshold tuning.
4. Fine-tune a lightweight pretrained YOLOv8 model in Google Colab. Record the model variant, image size, seed, hardware, training configuration, and dependency versions.
5. Evaluate precision, recall, mAP@0.5, and mAP@0.5:0.95, including per-class results and attention to small objects.
6. Benchmark inference FPS with clearly stated hardware, batch size, image size, warm-up, and timing boundaries. Report preprocessing/inference/postprocessing timing separately where possible.
7. Export the selected model to ONNX and validate that ONNX predictions are consistent with the PyTorch model within a documented tolerance.
8. Demonstrate confidence-filtered inference on a recorded road-driving video using OpenCV. This substitutes for a physical webcam while still exercising frame-by-frame video inference. It will be described accurately as prerecorded-video inference, not a webcam test.

## Repository layout

    traffic-sign-detection/
    ├── configs/
    │   └── gtsdb.yaml
    ├── notebooks/
    │   └── task_08_traffic_sign_detection.ipynb
    ├── src/
    │   └── traffic_sign_detection/
    ├── tests/
    ├── reports/
    │   └── figures/
    ├── videos/
    ├── requirements.txt
    ├── pyproject.toml
    └── README.md

The notebook, reusable preprocessing/evaluation/inference code, automated tests, and final report will be added in subsequent reviewed changes.

## Environment

The development and training workflow is designed for Google Colab. A free GPU will be used when available; CPU execution remains possible for lightweight checks. No local webcam or paid compute service is required.

Install dependencies in a compatible Python environment:

    python -m pip install -r requirements.txt

## Evaluation and reporting principles

- Keep the held-out test set out of training, model selection, and threshold tuning.
- Report mAP at both IoU 0.50 and the 0.50:0.95 range.
- Report precision and recall, and inspect per-class performance because the dataset is small and imbalanced.
- Measure FPS empirically; never claim real-time performance without recording the hardware and benchmark protocol.
- Apply a configurable confidence threshold and document how it was selected.
- Distinguish model inference time from end-to-end video processing throughput.
- Do not claim ONNX acceleration unless the exported model is actually benchmarked.

## References

- Houben, S. et al. (2013). [Detection of Traffic Signs in Real-World Images: The German Traffic Sign Detection Benchmark](https://www.ini.rub.de/upload/file/1470692848_e7f65a6c27d4c0ec0c71/HoubenEtAl_GTSDB_IJCNN2013.pdf). IJCNN.
- [Official GTSDB dataset and annotation documentation](https://benchmark.ini.rub.de/gtsdb_dataset.html)
- [Ultralytics documentation](https://docs.ultralytics.com/)

## License

The repository code is licensed under the MIT License. Dataset terms remain those of the original dataset provider or the specific mirror used; the repository's code license does not relicense dataset content.
