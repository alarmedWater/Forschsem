# README – meca_d405_capture

Ziel:  
- **Roboter sicher an definierte Posen fahren** (3 Strawberry-Views)  
- **D405 RGB + aligned Depth + optional PLY** aufnehmen  
- **Checks**, damit du schnell siehst, ob **Kamera/Depth/Roboterpose** plausibel sind  
- Optional: **Kamera-Offset (TRF→Camera optical center)** per Pivot/Ellipse schätzen  

## 0) Ordnerstruktur

```text
meca_d405_capture/
  config.yaml
  requirements.txt
  src/
    camera_d405.py
    robot_meca.py
    capture_dataset.py
    transforms.py
    validate_robot_pose.py
    estimate_cam_offset_pivot.py
```

---

## 1) Environment aktivieren & Packages installieren

### 1.1 Conda aktivieren
```bash
conda activate forschsem
```

### 1.2 Python Packages
```bash
python -m pip install -r requirements.txt
```

**requirements.txt (Beispiel)**
```text
numpy
PyYAML
opencv-python
mecademicpy
pyrealsense2
# optional
open3d
```

### 1.3 Quick sanity check (Imports)
```bash
python -c "import numpy, yaml, cv2; import pyrealsense2 as rs; print('OK')"
python -c "import importlib.metadata as m; print('pyrealsense2', m.version('pyrealsense2'))"
```

---

## 2) Kamera prüfen (ohne Roboter)

### 2.1 Kamera wird erkannt?
```bash
python - <<'PY'
import pyrealsense2 as rs
ctx = rs.context()
devs = ctx.query_devices()
print("devices:", len(devs))
for i, d in enumerate(devs):
    print(i, d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
PY
```

**Erwartung:** `devices: 1` (oder mehr) und Name/Serial.

### 2.2 Self-Check (depth_scale + intrinsics + z_center)
```bash
python src/camera_d405.py --self-check --patch 20 --debug-out runs/selfcheck.png
```

**Erwartung (Terminal):**
- `depth_scale_m_per_unit` (z.B. `0.0001`)
- intrinsics color/depth (fx/fy/cx/cy)
- `z_center_m` (Median-Depth am Zentrum; **sollte realistisch** sein: z.B. 0.25–1.2 m, je nach Abstand)

**Erwartung (Datei):**
- `runs/selfcheck.png` mit markiertem Patch & Textoverlay

Wenn `z_center_m = nan` oder `valid_n sehr klein`: dann sieht Depth am Zentrum nichts (zu nah/zu weit, IR-Probleme, falsche Beleuchtung, Objektfläche zu schlecht).

---

## 3) Roboter prüfen (ohne Kamera)

### 3.1 Roboter-Connect + Home + Pose lesen
```bash
python - <<'PY'
from src.robot_meca import Meca500Controller
r = Meca500Controller.from_config_yaml("config.yaml", verbose=True)
r.connect()
r.activate_and_home()
r.set_wrf_trf_from_config("config.yaml")
print(r.get_pose_mm_deg())
print("stability:", r.pose_stability_check(samples=10, dt_s=0.05))
r.disconnect()
PY
```

**Erwartung:**
- Roboter homed
- Pose wird ausgegeben
- `stability` sollte klein sein (typisch: < 0.1–0.5 mm, < 0.05–0.2 deg; je nach System)

Wenn `stability` groß ist: mechanische Vibration, nicht idle, falsches Timing → `settle_s` erhöhen.

---

## 4) Dataset aufnehmen (Roboter + Kamera)

### 4.1 config.yaml prüfen (Minimal)
Du brauchst mindestens:
- `robot.ip`
- `robot.positions` (z.B. `l/m/r` Joint-Posen)
- `dataset.view_to_pose_key` (view_id → pose_key)
- `camera.width/height/fps`

Dann starten:

```bash
python src/capture_dataset.py --config config.yaml
```

**Erwartete Outputs:**  
Ein Ordner in `runs/<run_name>/`, z.B.:

```text
runs/dataset_20260122_123456/
  camera_meta.yaml
  poses.csv
  plant_000/
    color_0.png
    depth_0.png
    depth_raw_0.png          (wenn aktiv)
    cloud_aligned_0.ply      (wenn aktiv)
    ...
  plant_001/
    ...
```

**Wichtig:**
- `depth_{view}.png` ist **aligned** → passt pixelgenau zu `color_{view}.png`
- `poses.csv` enthält die Roboterpose pro Aufnahme (mm/deg)

---

## 5) Validieren: Stimmen Koordinaten & Alignment?

### 5.1 Schneller Praxis-Check (ohne Mathe)
- Lade `color_*.png` und `depth_*.png`
- Prüfe, ob Objektkanten im RGB auch im Depth-Bild (als Colormap) an gleicher Stelle liegen  
  → sonst stimmt alignment / stream setting nicht.

### 5.2 Numerischer Check (Kamera intern)
- Vergleiche **RS-Deprojection** vs. **eigene Backprojection** (falls du beide nutzt)
- Erwartung: Differenzen im Bereich **wenige Millimeter** (bei ruhiger Depth)

### 5.3 Robot Pose Konsistenz
- Fahre **gleiches Pose-Key** 3–5× an → `GetPose` sollte nahezu identisch sein
- Wenn nicht: settle_s erhöhen, check Idle/Wait

---

## 6) Kamera-Offset schätzen (Optional, aber sehr hilfreich)

Wenn du einen festen Marker (Ellipse/Target) im Bild hast:

### 6.1 Capture für Pivot/Ellipse
```bash
python src/estimate_cam_offset_pivot.py --config config.yaml capture --run_dir runs/pivot_001 --debug
```

**Erwartung:**
- `runs/pivot_001/measurements.csv`
- `runs/pivot_001/debug/dbg_*.png` mit Ellipse + Zentrum + Depth

### 6.2 Solve Offset
```bash
python src/estimate_cam_offset_pivot.py --config config.yaml solve --run_dir runs/pivot_001
```

**Erwartung:**
- Terminal: `t_trf_cam` in m/mm + RMS error in mm
- Datei: `runs/pivot_001/estimated_cam_offset.yaml`

Wenn RMS > ~5–10 mm:
- Ellipse detection unsauber
- Depth noisy
- falsche Euler-Konvention / falsches `R_trf_cam`
- Marker bewegt sich

---

## Typische Probleme & was du dann siehst

**A) Depth zeigt überall 0 / valid_n=0**
- `z_center_m` = nan, `valid_n` klein  
- Fix: Abstand ändern, IR/Reflexion, Exposure, Target matt machen

**B) pyrealsense2 importiert, aber keine Devices**
- `devices: 0`  
- Fix: USB, Rechte (udev rules), VM-Passthrough, D405 Strom/Hub

**C) Roboterpose “wackelt”**
- `pose_stability_check` große Werte  
- Fix: `settle_s` hoch, `WaitIdle` prüfen, Vibration minimieren
