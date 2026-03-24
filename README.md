<div align="center">

# 🍽️ Plate Waste Detector

### AI-powered food waste analysis through Computer Vision

*Developed at a Hackathon · Computer Vision × Sustainability*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat-square)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 📌 Overview

**Plate Waste Detector** is an intelligent system built with **Computer Vision and Deep Learning** to identify, segment, and estimate food waste on plates or trays after consumption.

Designed for **collective dining environments** — such as university cafeterias and corporate canteens — this solution replaces manual, imprecise food waste tracking with granular, automated data. Using **instance segmentation with YOLOv8-seg**, the system reveals *which foods are most often wasted*, empowering managers to make data-driven decisions and reduce waste at the source.

> 🏆 Built during a hackathon, focused on applying AI to real-world sustainability challenges.

---

## 🎯 Objectives

**General Goal**
Develop a software system capable of **identifying, segmenting, and estimating the volume/weight of food residues** from images or real-time video.

**Specific Goals**
- Collect and prepare a dataset of post-consumption food images (leftover state).
- Train an **Instance Segmentation** model using YOLOv8-seg.
- Estimate residue weight/volume based on the segmented pixel area.
- Deliver actionable waste metrics to support operational decision-making.

---

## 🗂️ Project Structure

```
plate-waste-detector/
├── assets/             # Performance charts and visual demos
├── data/               # Business rules and metadata (dados.json)
├── dataset/            # Structured dataset (Images + Labels)
├── models/             # Trained model weights (best.pt)
├── scripts/            # Automation scripts (dataset download)
├── src/                # Application source code
│   ├── app.py          # Streamlit interface & Dashboard
│   └── detector.py     # Inference engine and data analysis logic
├── .env                # Environment variables (API key security)
├── .gitignore          # Version control filters
└── requirements.txt    # Project dependencies
```

---

## 🔄 Data Flow

```
Upload Image
     │
     ▼
Preprocessing (OpenCV)
Resize + Normalize
     │
     ▼
YOLO Inference (models/best.pt)
Instance Segmentation
     │
     ▼
Business Logic (detector.py × data/dados.json)
Consumable Food vs. Waste Classification
     │
     ▼
Dashboard (Streamlit)
Dynamic Charts + Automated Insights
```

1. **Input** — User uploads a plate image via Streamlit.
2. **Preprocessing** — Image is resized and normalized with OpenCV for model compatibility.
3. **Inference** — YOLO model (loaded from `models/best.pt`) detects objects and confidence scores.
4. **Business Intelligence** — `detector.py` cross-references detections with `data/dados.json` to classify consumable food vs. waste.
5. **Visualization** — Results are processed with Pandas and displayed as dynamic charts in the Dashboard, with automatic insights on the most frequently wasted foods.

---

## 🧠 Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV · Pillow |
| Model | YOLOv8-seg (Ultralytics) |
| Data Annotation | Roboflow / CVAT |
| Data Processing | Pandas · NumPy |
| Interface | Streamlit |
| Backend (optional) | FastAPI |
| Environment | Linux (Ubuntu / WSL) + CUDA |

---

## 🔬 Methodology

### 1. Data Collection & Preprocessing
- Collection of real images of plates/trays post-consumption.
- **Data Augmentation** applied: rotation, brightness, contrast adjustments.

### 2. Model Training
- **Transfer Learning** using YOLOv8-seg pre-trained weights.
- Fine-tuning for specific food classes:
  - Rice · Beans · Meat · Salad

### 3. Waste Estimation
Pixel-area-to-weight conversion formula:

```
Weight (g) = Pixel Area × Density Factor (g/pixel)
```

> The density factor is calibrated experimentally using a precision scale.

### 4. Validation & Interface
- Evaluation metrics: **mAP (Mean Average Precision)** and **IoU (Intersection over Union)**
- Real-time demo via webcam.

---

## 📊 Expected Output

After processing, the system generates reports such as:

```
Detected:  50g of Rice · 30g of Beans
Total Estimated Waste:  80g
```

These outputs enable both quantitative and qualitative analysis of food waste patterns over time.

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/your-username/plate-waste-detector.git
cd plate-waste-detector

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys if needed

# Run the application
streamlit run src/app.py
```

> **GPU recommended:** The model runs on CPU but performs significantly faster with CUDA-enabled hardware. Training is also feasible on [Google Colab](https://colab.research.google.com/).

---

## ✅ Technical Viability

- ✔️ No specialized sensors required — a standard camera is sufficient.
- ✔️ Trainable on local GPU or Google Colab (free tier).
- ✔️ Entirely built on consolidated open-source technologies.
- ✔️ Modular architecture — easy to extend with new food classes or integrate with existing cafeteria systems.

---

## 📚 Academic Context

Project developed in the context of **Software Engineering / Artificial Intelligence**, focused on **Computer Vision applied to Sustainability**. Originally submitted as a hackathon project.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

*Computer Vision applied to sustainability 🌱*

</div>
