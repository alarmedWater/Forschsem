# Strawberry Instance Segmentation - Docker Training Pipeline

## 📋 Projektübersicht

Dieses Projekt trainiert drei verschiedene Instance Segmentation Modelle (Detectron2, YOLACT++, YOLOv8) auf dem StrawDI-Datensatz für Erdbeerernte. Die Modelle können sowohl auf CPU (Schnelltests) als auch GPU (volles Training) laufen.

get the data: https://drive.google.com/file/d/1elFB-q9dgPbfnleA7qIrTb96Qsli8PZl/view

## 🏗️ Projektstruktur
SegemtationTrainig_v1/
├── docker/
│ ├── docker-compose.yml # Alle Docker Services
│ ├── detectron2/
│ │ ├── Dockerfile
│ │ ├── train_detectron2.py
│ │ └── run_detectron2_docker.sh
│ └── yolact/
│ ├── Dockerfile
│ ├── train_yolact.py
│ └── run_yolact_docker.sh
├── models/
│ └── train_yolo8s.py # Original YOLOv8
├── tools/
│ ├── convert_strawdi.py # Datenkonvertierung
│ └── view_yolo_seg.py # Datenvisualisierung
├── converted/ # Generierte Datensätze
│ ├── coco/ # COCO-Format
│ └── yolo/ # YOLO-Format
└── runs/ # Trainingsergebnisse


## 📊 Datensatzvorbereitung

### 1. Datenstruktur vorbereiten

Die Rohdaten müssen folgende Struktur haben:
StrawDI_Db1/
├── train/
│ ├── img/.png|jpg
│ └── label/.png
├── val/
│ ├── img/.png|jpg
│ └── label/.png
└── test/
├── img/.png|jpg
└── label/.png


### 2. Daten konvertieren

```bash
cd SegemtationTrainig_v1

# Daten in COCO und YOLO Format konvertieren
python tools/convert_strawdi.py \
    --src /pfad/zu/StrawDI_Db1 \
    --out_coco converted/coco \
    --out_yolo converted/yolo \
    --link_mode hardlink

Parameter:

    --src: Pfad zum StrawDI_Db1 Ordner

    --link_mode: hardlink (platzsparend), symlink oder copy

### 3. Datenqualität prüfen

# YOLO-Segmentation Labels visualisieren
python tools/view_yolo_seg.py
siehe debug_vis/


🐳 Docker Training
Voraussetzungen

    Docker und Docker Compose installiert

    20+ GB freier Speicherplatz

    Für GPU Training: NVIDIA Docker Support

1. Detectron2 (Mask R-CNN)

CPU Schnelltest (5-15 Minuten):
bash

cd SegemtationTrainig_v1/docker
docker-compose up detectron2-cpu

GPU Volltraining:
bash

docker-compose up detectron2-gpu

Manuell mit Script:
bash

cd docker/detectron2
./run_detectron2_docker.sh cpu    # oder gpu

2. YOLACT++

Extrem Schnelltest (1-2 Minuten):
bash

docker-compose up yolact-quicktest

CPU Schnelltest (5-15 Minuten):
bash

docker-compose up yolact-cpu

GPU Volltraining:
bash

docker-compose up yolact-gpu

Manuell mit Script:
bash

cd docker/yolact
./run_yolact_docker.sh test    # quicktest
./run_yolact_docker.sh cpu     # sanity test  
./run_yolact_docker.sh gpu     # full training

3. YOLOv8 (ohne Docker)

CPU Schnelltest:
bash

conda activate yolo8-seg
python models/train_yolo8s.py  # USE_CUDA = False in Script

GPU Training:
bash

conda activate yolo8-seg  
python models/train_yolo8s.py  # USE_CUDA = True in Script

⚙️ Konfiguration
Environment Variablen

Detectron2:

    DETECTRON2_USE_CUDA=1 für GPU

    DETECTRON2_USE_CUDA=0 für CPU

YOLACT:

    YOLACT_USE_CUDA=1 für GPU

    YOLACT_USE_CUDA=0 für CPU

    YOLACT_QUICK_TEST=1 für extrem schnellen Test

Trainingsparameter
Modus	Bilder	Iterationen	Geschätzte Zeit	Zweck
Quick Test	10	100	1-2 min	Basics check
CPU Sanity	50	500	5-15 min	Daten/Config validieren
GPU Full	Alle	80,000	Mehrere Stunden	Finales Training
📁 Output Verzeichnisse

    Detectron2: runs/maskrcnn_r50fpn/

    YOLACT: runs/yolactpp_r101/

    YOLOv8: runs/detect/train_yolov8s/

Enthalten:

    Model Checkpoints

    Trainingslogs

    Evaluationsmetriken

    Visualisierungen

🔧 Troubleshooting
Disk Space Issues
bash

# Docker Cache leeren
docker system prune -a -f

# Conda Cache leeren
conda clean --all -y

GPU Probleme
bash

# GPU Verfügbarkeit prüfen
docker run --gpus all nvidia/cuda:11.8-base nvidia-smi

# Fallback zu CPU
docker-compose up detectron2-cpu

Datenprobleme
bash

# Datenstruktur validieren
python tools/view_yolo_seg.py

# Konvertierung prüfen
ls -la converted/coco/images/train/
ls -la converted/yolo/train/labels/

🚀 Empfohlener Workflow

    Daten vorbereiten mit convert_strawdi.py

    Quick Test mit yolact-quicktest (1-2 Minuten)

    CPU Sanity mit beiden Modellen (15-30 Minuten)

    GPU Training nach erfolgreichen Tests

📝 Notizen

    CPU Tests sind für schnelle Entwicklung und Fehlerbehebung

    GPU Training für finale Modelle

    Alle Outputs sind persistent in runs/ gespeichert

    Docker Images werden automatisch gebaut beim ersten Start
