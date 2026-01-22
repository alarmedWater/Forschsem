# meca_d405_capture

Mini-Tooling für Meca500 + Intel RealSense D405:
- Robot-Posen zuverlässig abfahren und prüfen
- RGB + aligned depth + raw depth speichern
- Punktwolken exportieren (PLY)
- Kamera-Offset (TRF->CAM Translation) per Pivot/Ellipse-fit schätzen
- Dataset mit 3 Views aufnehmen

## 0) Installation (Ubuntu)

### RealSense (empfohlen über librealsense)
- Installiere librealsense2 + python bindings (je nach Setup/Repo).
- Prüfe danach:
  python3 -c "import pyrealsense2 as rs; print(rs.__version__)"

### Python venv
cd meca_d405_capture
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## 1) Projekt starten / Imports
Wir führen Skripte direkt aus dem Projekt-Root aus:
python src/validate_robot_pose.py --config config.yaml

(Die Skripte fügen src automatisch in sys.path ein.)

## 2) Empfohlener Workflow

### Schritt A: Kamera testen
python src/camera_d405.py

=> Gibt Meta/Intrinsics/Depth-Scale aus und macht einen Frame-Test.

### Schritt B: Roboter-Posen validieren (Pose stabil? Wiederholbar?)
python src/validate_robot_pose.py --config config.yaml

=> Fährt sequence ab (l,m,r) und prüft Pose-Stabilität / Wiederholbarkeit.

### Schritt C: Kamera-Offset schätzen (Pivot/Ellipse)
1) Lege einen gut sichtbaren elliptischen Marker ins Sichtfeld (z.B. gedruckter Kreis/ellipse, stark kontrastreich).
2) Capture Messungen:
python src/estimate_cam_offset_pivot.py --config config.yaml capture --run_dir runs/pivot_001 --debug
3) Solve:
python src/estimate_cam_offset_pivot.py --config config.yaml solve --run_dir runs/pivot_001

=> Ergebnis:
- runs/pivot_001/estimated_cam_offset.yaml
Übernimm den Wert in config.yaml -> robot.camera_in_trf_translation_mm

Ziel: RMS typischerweise wenige mm (je nach Marker/Depth).

### Schritt D: Dataset aufnehmen (RGB + aligned depth + PLY + robot pose)
python src/capture_dataset.py --config config.yaml

=> Output:
runs/<run_name>/
  camera_meta.yaml
  poses.csv
  plant_000/ color_0.png depth_0.png cloud_aligned_0.ply ...
  ...

## 3) Wenn Views verdreht sind
- Prüfe euler_convention (RzRyRx_deg ist häufig korrekt für Mecademic "mobile XYZ")
- Prüfe R_trf_cam (cam_axes_correction_R_trf_cam_row_major_3x3)
- Schätze t_trf_cam neu (Pivot) und übernimm camera_in_trf_translation_mm
- Prüfe depth aligned: depth_{view}.png ist aligned zu color -> Backprojection nutzt COLOR intrinsics
