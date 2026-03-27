#  TD — Détection d’objets avec IA + FastAPI +Docker

---

##  Objectif

Ce TD a pour objectif de vous faire découvrir comment :

* utiliser un modèle d’intelligence artificielle pour analyser une image ;
* transformer un programme Python en **service web (API)** ;
* manipuler des données au format **JSON** ;
* comprendre les bases d’une application moderne utilisée en entreprise.

 À la fin de ce TD, vous serez capable de créer une API simple intégrant de l’intelligence artificielle et de la déployer dans un conteneur Docker pour garantir sa portabilité et sa reproductibilité.
---

##  À quoi sert ce TD ?

Dans le monde professionnel, les modèles d’IA ne sont **pas utilisés seuls**.
Ils sont intégrés dans des systèmes complets :

* applications web
* services cloud
* logiciels industriels
* systèmes embarqués

 Ce TD vous montre comment passer de :

```text
Script Python → Application utilisable (API)
```

---

##  Exemple concret

Une entreprise peut utiliser ce type de système pour :

* détecter des personnes (sécurité)
* analyser des produits (industrie)
* compter des objets (logistique)
* automatiser des contrôles visuels

---

#  Pourquoi ces technologies ?

---

##  Pourquoi une API ?

Sans API :

```text
Script Python → Résultat local
```

 Limites :

* utilisable uniquement sur votre machine
* non réutilisable
* non accessible

Avec une API :

```text
Client → API → Résultat
```

 Avantages :

* accessible via URL
* utilisable par d’autres applications
* standard du développement moderne

---

##  Pourquoi FastAPI ?

FastAPI permet de créer facilement une API en Python.

 Avantages :

* simple à utiliser
* rapide
* documentation automatique (`/docs`)
* peu de code

 Il transforme un script en **service web**

---

##  Pourquoi YOLO ?

YOLO est un modèle d’intelligence artificielle qui permet de :

* détecter des objets
* localiser leur position
* donner une probabilité

 Modèle rapide et utilisé en industrie

---

##  Pourquoi Docker ?

Docker permet d’exécuter une application dans un conteneur.

###  Sans Docker :

* problèmes d’installation
* versions incompatibles
* “ça marche chez moi”

###  Avec Docker :

```text
Application + dépendances → conteneur
```

 Même environnement partout

---

##  Pipeline du TD

```text
Image → API → YOLO → JSON + image annotée
```

---

#  Phase 1 — Modèle de vision (YOLO)

##  Installation

```bash
git clone https://github.com/Tinny-Robot/Live-Object-Detection-with-Camera.git
cd Live-Object-Detection-with-Camera

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

---

##  Test du modèle

Si vous êtes sous Windows avec caméra :

```bash
python app.py
```

 Objectif :

* tester YOLO
* capturer une image
* comprendre le fonctionnement

---

#  Phase 2 — API avec FastAPI

##  Installation

```bash
pip install fastapi uvicorn python-multipart
```

---

##  Code principal (`main.py`)

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from ultralytics import YOLO
import numpy as np
import cv2
import os
import time

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()
model = YOLO("yolov8n.pt")

@app.get("/")
def home():
    return {"message": "API de détection d'objets"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()

    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Image invalide"}

    results = model(image)

    annotated_image = image
    for result in results:
        annotated_image = result.plot()

    filename = f"result_{int(time.time())}.jpg"
    output_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(output_path, annotated_image)

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        detections.append({
            "class": model.names[cls_id],
            "confidence": round(conf, 3),
            "bbox": [round(x, 2) for x in xyxy]
        })

    return {
        "count": len(detections),
        "detections": detections,
        "image_name": filename
    }

@app.get("/image/{filename}")
def get_image(filename: str):
    return FileResponse(f"outputs/{filename}")
```

---

##  Lancer l’API

```bash
uvicorn main:app --reload
```

---

##  Accès

```text
http://127.0.0.1:8000/docs
```

---

##  Résultat

Les images détectées sont sauvegardées dans :

```text
outputs/
```

---

##  Voir une image

```text
http://127.0.0.1:8000/image/NOM_IMAGE.jpg
```

---

##  Remarques

* Les warnings CUDA peuvent être ignorés
* Le modèle fonctionne en CPU

---

#  Phase 3 — Docker

##  requirements.txt

```txt
fastapi
uvicorn[standard]
python-multipart
opencv-python
ultralytics
supervision
```

---

##  Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

##  .dockerignore

```text
.venv
__pycache__
*.pyc
.git
outputs
```

---

##  Build

```bash
docker build -t yolo-api .
```

---

##  Run

Linux / WSL :

```bash
docker run --rm -p 8000:8000 -v $(pwd)/outputs:/app/outputs yolo-api
```

Windows PowerShell :

```bash
docker run --rm -p 8000:8000 -v ${PWD}/outputs:/app/outputs yolo-api
```

---

##  Test

```text
http://localhost:8000/docs
```

---

##  Résultat attendu

* upload image
* JSON de détection
* image annotée accessible

---

##  Ce que vous apprenez

* transformer un script IA en API
* manipuler des images
* créer un backend
* introduire Docker

---

##  Résumé

 Transformer un modèle d’IA en **service web utilisable**
