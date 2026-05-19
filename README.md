# Rice Grain Instance Segmentation

A lightweight computer vision pipeline that converts a raw scanned image of rice grains into a stylized instance-segmentation output, where each detected grain is shown as a distinct colored region on a black background.

The goal is to keep the solution simple, reproducible, and easy to run.

---

## Project Structure

```text
rice-grain-instance-segmentation/
│
├── rice_segmentation.py           # Main Python script for the full pipeline
├── requirements.txt               # Libraries needed to run the project
├── README.md                      # Project guide
│
├── inputs/
│   ├── InputImage.jpg             # Raw rice image
│   └── ExpectedOutput.jpeg        # Reference segmented image
│
└── outputs/
    ├── 01_gray.png                # Grayscale image
    ├── 02_blurred.png             # Blurred grayscale image
    ├── 03_foreground_mask.png     # Rice foreground mask
    ├── 04_distance_map.png        # Distance transform image
    ├── 05_seed_overlay.png        # Detected seed points on input image
    ├── 06_final_segmented_output.png  # Final generated segmentation
    ├── comparison.png             # Input vs generated vs reference
    ├── run_summary.txt            # Run details and metrics
    └── seed_instances.csv         # Seed and ellipse details
```

> If your reference image is named `ExpectedOutput.jpg` instead of `ExpectedOutput.jpeg`, either rename it or update the command path.

---

## Pipeline Overview

```text
Input image
-> Grayscale conversion
-> Gaussian blur
-> Foreground mask creation
-> Distance transform
-> Seed point detection
-> Local grain-shape estimation
-> Colored instance rendering
-> Output saving
```

Briefly, the code separates rice grains from the dark background, finds likely grain centers, estimates a rice-like shape around each center, and renders each detected grain in a different color.

---

## Setup

Clone the repository:

```bash
git clone https://github.com/your-username/rice-grain-segmentation.git
cd rice-grain-segmentation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

Place the input files inside the `inputs/` folder:

```text
inputs/InputImage.jpg
inputs/ExpectedOutput.jpeg
```

Then run:

```bash
python rice_segmentation.py
```

The results will be saved inside:

```text
outputs/
```

---

## Run with Custom Image Paths

If your images are in another folder or have different names:

```bash
python rice_segmentation.py --input "inputs/InputImage.jpg" --expected "inputs/ExpectedOutput.jpeg" --output-dir "outputs"
```

If you do not have a reference image:

```bash
python rice_segmentation.py --input "inputs/InputImage.jpg" --no-expected --output-dir "outputs"
```

---

## Useful Parameters

```bash
python rice_segmentation.py --threshold 60 --min-seed-distance 12 --ellipse-scale 2.0
```

| Parameter | Purpose |
|---|---|
| `--threshold` | Controls foreground extraction |
| `--min-seed-distance` | Controls minimum distance between detected grain centers |
| `--ellipse-scale` | Controls size of rendered grain ellipses |
| `--clip-to-foreground` | Restricts ellipses inside the foreground mask |
| `--output-dir` | Folder where outputs are saved |

---

## Main Output Files

| File | Meaning |
|---|---|
| `06_final_segmented_output.png` | Final generated colored segmentation |
| `comparison.png` | Side-by-side input, generated output, and reference |
| `05_seed_overlay.png` | Shows detected seed points |
| `run_summary.txt` | Short run summary and metrics |
| `seed_instances.csv` | Coordinates and shape details of detected grains |

---

## Notes

- The generated output uses a black background.
- Each detected rice grain is rendered using a distinct color.
- The solution uses a lightweight classical computer vision pipeline.
- Pretrained models such as SAM/SAM2 can be explored later, but this version focuses on clarity, reproducibility, and simple execution.
