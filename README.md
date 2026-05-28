# DTS405 Coursework 2: Basketball Detection and BEV Mini-map

This project implements a basketball broadcast vision pipeline for DTS405 Coursework 2. It trains a YOLO detector on the provided basketball clips, projects player positions to a 2D bird's-eye-view court template using a manually calibrated homography, and renders a demo video with detection boxes and a top-right mini-map overlay.

The implementation is notebook-driven and configuration-driven. There is no command-line argument interface.

## Project Scope

- Object detection with YOLOv8n.
- Four dataset classes are used exactly as provided:
  - `0 player`
  - `1 ball`
  - `2 referee`
  - `3 others`
- The mini-map projects only:
  - players, using bounding-box bottom-centre foot-points
  - the ball, using bounding-box centre as an approximate image point
- `referee` and `others` are drawn on the main video frame but are not projected to the mini-map.
- Camera B, the `clip2` / `clip3` viewpoint, is used for the BEV demo.
- The system produces a 2D BEV mini-map, not a full 3D reconstruction.

## Repository Structure

```text
DTS405_CW2/
├─ basketball_bev_pipeline.ipynb
├─ configs/
│  ├─ project.yaml
│  └─ homography_camera_b.json
├─ src/
│  ├─ config.py
│  ├─ dataset.py
│  ├─ train_yolo.py
│  ├─ evaluate_yolo.py
│  ├─ homography.py
│  ├─ minimap.py
│  ├─ visualization.py
│  └─ video.py
├─ assignment.md
├─ README.md
└─ requirements.txt
```

Generated files are written under `outputs/` by default.

## Expected Raw Dataset Layout

Update `raw_data_root` in `configs/project.yaml` so it points to the folder containing the extracted dataset.

The expected layout is:

```text
raw_data_root/
├─ trian/                 # The provided folder may be named "trian"
│  └─ clip1/
│     ├─ classes.txt
│     ├─ notes.json
│     ├─ images/
│     └─ labels/
└─ test/
   ├─ clip2/
   │  ├─ classes.txt
   │  ├─ notes.json
   │  ├─ images/
   │  └─ labels/
   ├─ clip3/
   └─ clip4/
```

The code also tolerates `train/clip1` if the training folder has been renamed from `trian` to `train`.

## Data Split

The processed YOLO dataset is created from the extracted clips:

- `clip1` first 400 frames: training
- `clip1` last 100 frames: validation
- `clip2`, `clip3`, `clip4`: test

The dataset preparation step writes a standard YOLO layout:

```text
outputs/basketball_yolo/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
├─ labels/
│  ├─ train/
│  ├─ val/
│  └─ test/
├─ data.yaml
└─ split_summary.json
```

## Main Workflow

Open and run `basketball_bev_pipeline.ipynb`.

The notebook is organised as:

1. Setup and Paths
2. Dataset Inspection and Split
3. YOLO Training / Loading Trained Model
4. Detection and Evaluation
5. Manual Homography Calibration
6. Mini-map Template Preview
7. Demo Video Generation
8. Results and Discussion

The notebook contains boolean switches for expensive stages:

```python
PREPARE_DATASET = False
RUN_TRAINING = False
RUN_EVALUATION = False
RUN_DEBUG_DEMO = False
RUN_FULL_DEMO = False
```

Set each switch to `True` only when that stage should run.

## Configuration

### `configs/project.yaml`

This file controls:

- raw dataset path
- processed YOLO dataset path
- output paths
- class names
- YOLO training settings
- confidence threshold
- tracker choice
- mini-map visualisation settings

Before running the notebook, update:

```yaml
paths:
  raw_data_root: "D:/path/to/basketball_dataset"
```

### `configs/homography_camera_b.json`

This file stores the manual Camera B calibration.

Fill:

- `reference_frame`: the selected clear `clip2` frame
- `image_points_px`: manually selected image points from that frame
- `template_points_m`: corresponding standard-court coordinates in metres

Example shape:

```json
{
  "image_points_px": [[100, 200], [300, 220], [420, 380], [80, 360]],
  "template_points_m": [[0.0, 0.0], [14.0, 0.0], [14.0, 15.0], [0.0, 15.0]]
}
```

Use points that are visible, reliable, and not close to collinear. Four points are the minimum; more points can be used if they are clearly identifiable.

## Homography Method

The homography implementation is in `src/homography.py`.

The main method is a manual DLT/SVD implementation:

- image point: `p = [u, v, 1]^T`
- court template point: `P = [x, y, 1]^T`
- homography relation: `P ~ H p`
- point correspondences form `Ah = 0`
- SVD solves for `h`
- `h` is reshaped into the `3 x 3` matrix `H`

`cv2.findHomography` is not used as the primary implementation.

The notebook also computes a reference-point reprojection error as a calibration sanity check.

## Training

Training is implemented in `src/train_yolo.py` and called from the notebook.

Baseline configuration:

- model: `yolov8n.pt`
- epochs: `50`
- image size: `640`
- seed: `42`
- custom/default augmentation is minimised for the main run
- output directory: `outputs/runs/`

The expected best model path is:

```text
outputs/runs/detect/yolov8n_noaug/weights/best.pt
```

## Evaluation

Evaluation is implemented in `src/evaluate_yolo.py`.

The notebook can produce:

- validation metrics on `clip1` last 100 frames
- overall test metrics on `clip2 + clip3 + clip4`
- optional per-clip test metrics for `clip2`, `clip3`, and `clip4`

Reported metrics include:

- Precision
- Recall
- F1, computed as `2PR / (P + R)`
- mAP@0.5
- mAP@0.5:0.95

Metrics JSON files are saved under `outputs/metrics/`.

## Demo Video

Demo rendering is implemented in `src/video.py`.

The main demo uses `clip2` and renders:

- all four detection classes on the original broadcast frame
- player tracking IDs using lightweight tracking
- compressed continuous display IDs for players
- a top-right BEV mini-map overlay
- player short trajectories over roughly the last 20 frames
- ball projection with short missing-frame tolerance

Outputs are saved under:

```text
outputs/demo/
outputs/screenshots/
```

## Notes and Limitations

- The assignment mentions hoop detection, but the provided dataset classes do not include a separate hoop label. This implementation follows the provided label set.
- The BEV mapping assumes a fixed or approximately fixed camera for Camera B.
- Slight camera movement may cause mini-map drift.
- Ball projection is approximate because the ball can be airborne.
- Player projection uses the bounding-box bottom-centre as a foot-point estimate, which can be inaccurate during occlusion or jumping.
- With only four clear homography points, the mini-map is intended as a visually reasonable tactical representation rather than precise physical measurement.
