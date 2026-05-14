# Santos et al. 2024 — side-scan sonar mine-detection dataset

This directory is a placeholder. Download the dataset from Figshare:

- DOI: 10.6084/m9.figshare.24574879
- Reference: Santos et al. 2024

The dataset contains:
- 1170 SSS images
- 304 with bounding-box annotations
- 668 annotated objects across 2 classes: `MILCO`, `NOMBO`

Expected on-disk layout (TODO: confirm; update once data is extracted):

```
data/santos/
├── images/        # .jpg or .png
└── labels/        # YOLO-format .txt, one per image
```
