# 📦 YOLO Object Detection API (FastAPI + Docker)

## 🎯 Overview

This project demonstrates how to transform a Computer Vision model (YOLO) into a modern web service using FastAPI and Docker.

It is designed as a practical learning project to understand how Artificial Intelligence is integrated into real-world applications.

---

## 🚀 Features

- Object detection using YOLOv8
- REST API with FastAPI
- Image upload and processing
- JSON response with detections
- Annotated image generation
- Docker support for deployment

---

## 🧠 Architecture

```text
Client → FastAPI → YOLO Model → JSON + Annotated Image
```

---

## 🧩 Tech Stack

- Python
- FastAPI
- Ultralytics YOLOv8
- OpenCV
- Docker

---

## 📥 Installation

```bash
git clone https://github.com/Tinny-Robot/Live-Object-Detection-with-Camera.git
cd Live-Object-Detection-with-Camera

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

---

## ▶️ Run the API

```bash
uvicorn main:app --reload
```

---

## 🌐 API Docs

Open in browser:

```
http://127.0.0.1:8000/docs
```

---

## 📡 Endpoints

### GET /
Health check message

### GET /health
Returns API status

### POST /detect
Upload an image and receive detections

### GET /image/{filename}
Retrieve annotated image

---

## 📤 Example Response

```json
{
  "count": 1,
  "detections": [
    {
      "class": "person",
      "confidence": 0.89,
      "bbox": [x1, y1, x2, y2]
    }
  ],
  "image_name": "result_123456.jpg"
}
```

---

## 🐳 Docker

### Build

```bash
docker build -t yolo-api .
```

### Run

Linux / WSL:
```bash
docker run --rm -p 8000:8000 -v $(pwd)/outputs:/app/outputs yolo-api
```

Windows PowerShell:
```powershell
docker run --rm -p 8000:8000 -v ${PWD}/outputs:/app/outputs yolo-api
```

---

## ⚠️ Notes

- CUDA warnings can be ignored (CPU execution is used)
- Images are saved in the `outputs/` directory
- Ensure `yolov8n.pt` is present in the root directory

---

## 🎓 Learning Outcomes

- Build a REST API with FastAPI
- Integrate an AI model into a backend service
- Handle image processing in Python
- Understand containerization with Docker

---

## 📌 Summary

This project shows how to move from:

```
Python Script → Production-ready API
```

---

## 📄 License

For educational purposes.
