# Multimodal Chart & Spectra Vision LLM: Automated Scientific Data Extraction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Vision-LLM](https://img.shields.io/badge/Architecture-Qwen2--VL%20%7C%20PaliGemma-purple)]()
[![Pydantic](https://img.shields.io/badge/Output-Strict%20Structured%20JSON-orange)]()

An end-to-end multimodal vision-language pipeline specializing in the parsing, coordinate de-skewing, and digitized tabular extraction of complex scientific figures (UV-Vis spectra, ternary phase diagrams, XRD patterns, and kinetic Arrhenius plots).

---

## 🔬 Core Capabilities

* **Scientific Figure Parsing:** Dissects non-standard scientific axes (log-linear, inverted wave-numbers, dual Y-axes).
* **Spectral Curve Digitization:** Extracts continuous raw $(x, y)$ coordinate points from rasterized plot images using combined Vision-Language Models (VLMs) and OpenCV edge-tracking.
* **Ternary & Phase Diagram Decomposition:** Reconstructs component molar fractions $(x_A, x_B, x_C)$ directly from triangular phase diagrams.
* **Strict Pydantic Validation:** Guarantees machine-readable, schema-verified JSON outputs suitable for automated Materials Informatics databases.

---

## 🏗️ Architecture

```mermaid
graph TD
    A[Scientific PDF / Image] --> B[YOLOv8-Doc Layout Detection]
    B --> C[Chart & Figure Cropping]
    C --> D[VLM Parsing Engine - Qwen2-VL]
    D --> E[Sub-Pixel Contour Extraction - OpenCV]
    E --> F[Coordinate Calibration & Mapping]
    F --> G[Pydantic Structured Output Validation]
    G --> H[Output: JSON / Pandas Parquet]
```

---

## 🚀 Quickstart

```bash
git clone https://github.com/dhrubchatterjee16-lang/multimodal-chart-vision-llm.git
cd multimodal-chart-vision-llm
pip install -r requirements.txt
python extract_chart.py --image_path examples/uv_vis_spectrum.png --output_format json
```
