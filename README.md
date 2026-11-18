# Traffic Sign Detection

YOLOv8-based traffic sign **object detection** on the German Traffic Sign Detection Benchmark (GTSDB). This project detects and classifies signs in complete road scenes and evaluates detection quality, inference latency, model export, and prerecorded-video inference.

## Project status

**Task 8 implementation and notebook work are complete.** The end-to-end workflow, evaluation, export, video inference, and reproducibility reporting are documented in the Task 8 notebook. The notebook and its saved execution outputs are committed on the `task-08/gtsdb-yolov8-foundation` branch.

Large datasets, trained weights, generated runs, and video files are intentionally kept out of Git; they are generated or stored in the Colab/Drive workspace as described in the notebook. The repository includes the reusable annotation utilities, dataset configuration, tests, and environment metadata.

## Objective and dataset

The project uses the [German Traffic Sign Detection Benchmark (GTSDB)](https://benchmark.ini.rub.de/gtsdb_dataset.html), which provides 900 full-scene images: 600 training images and 300 official evaluation images. The images are PPM files, and the native annotations are semicolon-delimited bounding-box records.

The workflow converts full-scene annotations to YOLO format and uses 43 fine-grained class IDs (0–42). The official 300-image evaluation partition is preserved as the held-out test set. The 600-image training partition is divided into 420 training, 90 validation, and 90 internal test images (70/15/15). Image-level split overlap was checked. Confidence-threshold selection uses validation data, not the held-out test set.

## Completed workflow and results

- Audited the source images and native annotations, checked bounding-box geometry and class IDs, and handled duplicate records.
- Converted PPM images and annotations into YOLO-compatible image/label files, including empty label files for negative scenes.
- Created and validated the dataset configuration and leakage-aware data splits.
- Reused the trained YOLOv8n checkpoint and evaluated the model.
- Generated evaluation visualizations, including training curves, confusion matrix, precision–recall curves, and prediction examples.
- Reviewed confidence thresholds from 0.10 to 0.50 using validation data.
- Exported the selected checkpoint to ONNX (Opset 12) and ran ONNX structural and CPU inference smoke checks. This is an export/compatibility check, not a claim of ONNX speed-up or measured PyTorch-versus-ONNX prediction parity.
- Ran inference on a prerecorded road video using OpenCV. The processed demonstration contains 644 frames at 2560×1440 and 30 FPS (about 21.47 seconds). It is prerecorded-video inference, not a physical webcam test.
- Recorded environment, hardware, split, evaluation, timing, and artifact details in the notebook's reproducibility report.

### Evaluation snapshot

| Metric | Validation split | Held-out internal test |
|---|---:|---:|
| Precision | 0.498 | 0.563 |
| Recall | 0.467 | 0.521 |
| mAP@0.5 | 0.463 | 0.545 |

The reported single-image benchmark latency is 56.89 ms (17.58 FPS). For the prerecorded-video run, measured model inference throughput was approximately 50.43 FPS, while end-to-end video processing throughput was approximately 12.12 FPS. These are different timing measurements and should not be conflated. Results are specific to the recorded Colab T4 environment and the notebook's settings; they are not a general real-time performance guarantee.

## Repository contents

```
traffic-sign-detection/
├── .github/workflows/tests.yml
├── configs/gtsdb.yaml
├── notebooks/task_08_traffic_sign_detection.ipynb
├── src/traffic_sign_detection/
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Reproducing the work

The notebook is designed for Google Colab. It records the setup, data preparation, model evaluation, export, video inference, and reproducibility steps. A free GPU is used when available; no paid compute service or physical webcam is required.

Install the repository dependencies in a compatible Python environment:

```bash
python -m pip install -r requirements.txt
```

Run the lightweight repository tests with:

```bash
pytest -q
```

The GitHub Actions workflow runs these tests on pushes to `main` and `task-08/**`, and on pull requests targeting `main`.

## Evaluation and reporting notes

- The held-out test partition is kept separate from training and confidence-threshold selection.
- Precision, recall, and mAP@0.5 are reported for validation and held-out internal test evaluation.
- Inference FPS and end-to-end video throughput are reported separately.
- ONNX export and CPU smoke testing are documented without claiming acceleration.
- Raw datasets, generated labels, model weights, training runs, and large video files are excluded from Git.

## References

- Houben, S. et al. (2013). [Detection of Traffic Signs in Real-World Images: The German Traffic Sign Detection Benchmark](https://www.ini.rub.de/upload/file/1470692848_e7f65a6c27d4c0ec0c71/HoubenEtAl_GTSDB_IJCNN2013.pdf). IJCNN.
- [Official GTSDB dataset and annotation documentation](https://benchmark.ini.rub.de/gtsdb_dataset.html)
- [Ultralytics documentation](https://docs.ultralytics.com/)

## License

The repository code is licensed under the MIT License. Dataset terms remain those of the original dataset provider or the specific mirror used; the repository's code license does not relicense dataset content.
