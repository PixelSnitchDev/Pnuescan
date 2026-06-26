# PneuScan 🫁

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-red)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-85.25%25-green)

> AI-powered pneumonia detection from chest X-rays using deep learning.

## Overview

PneuScan analyzes chest X-ray images using a trained ResNet18 CNN model, generates Grad-CAM heatmaps to visualize affected lung regions, locates nearby hospitals, and produces downloadable PDF reports — all through a structured web interface.

## Features

- Chest X-ray classification as Normal or Pneumonia using ResNet18
- Nearest hospital finder using Overpass API and Leaflet.js maps
- Automated PDF report generation for each scan
- Patient history dashboard to track previous results
- Login system with structured web interface
- SQLite database for storing patient scan history
## Known Limitations

- Heatmaps are generated using Grad-CAM based on model gradients but do not precisely highlight pneumonia regions due to the absence of a segmentation-annotated dataset. A pixel-level annotated dataset would significantly improve heatmap accuracy.

## Model Performance

| Metric | Score |
|--------|-------|
| Validation Accuracy | 93.75% |
| Test Accuracy | 85.25% |
| Architecture | ResNet18 CNN |
| Framework | PyTorch |
| Training Platform | Google Colab |

## Tech Stack

- **Deep Learning:** PyTorch + ResNet18
- **Backend:** Python 3.11 + Flask
- **Image Processing:** OpenCV + PIL + NumPy
- **Visualization:** Grad-CAM heatmaps
- **Maps:** Leaflet.js + OpenStreetMap + Overpass API
- **Report Generation:** ReportLab
- **Database:** SQLite
- **Frontend:** HTML + CSS + JavaScript

## Setup Instructions

### Step 1 — Clone the repo
```bash
git clone https://github.com/PixelSnitchDev/Pnuescan.git
cd Pnuescan
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install flask torch torchvision pillow reportlab opencv-python requests numpy
```

### Step 4 — Add model file
Place `pneumonia_model_v2.pth` in the root folder.

### Step 5 — Run the app
```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

## Project Journey

This project began as a hackathon prototype to explore the idea and system design. After the hackathon it was fully reworked — including training a deep learning model from scratch on Google Colab, integrating it into a Flask backend, and building a complete user-facing application with meaningful features. The key learning was understanding the full pipeline from model training and preprocessing to building a usable, user-oriented system.

## Note

The model file `pneumonia_model_v2.pth` is not included in this repo due to file size. Contact the author for access.