#!/usr/bin/env python3
"""
Sortiere Bilder in plant_view Ordnerstruktur
Von: /media/psf/Home/Downloads/Basisdatensatz/
Nach: /home/parallels/Forschsemrep/strawberry_ws/data/plant_views/
"""

import os
import shutil
from pathlib import Path

def organize_images():
    # Quell- und Zielpfade
    src_dir = Path("/media/psf/Home/Downloads/Basisdatensatz")
    dest_base = Path("/home/parallels/Forschsemrep/strawberry_ws/data/plant_views")
    
    # Sicherstellen, dass Zielpfad existiert
    dest_base.mkdir(parents=True, exist_ok=True)
    
    # Alle color und depth Bilder finden und sortieren
    color_images = sorted(src_dir.glob("color_*.png"))
    depth_images = sorted(src_dir.glob("depth_*.png"))
    
    # Prüfen, ob Bilder gefunden wurden
    if not color_images:
        print(f"❌ Keine Color-Bilder gefunden in: {src_dir}")
        return
    if not depth_images:
        print(f"❌ Keine Depth-Bilder gefunden in: {src_dir}")
        return
    
    print(f"🔍 Gefunden: {len(color_images)} Color-Bilder")
    print(f"🔍 Gefunden: {len(depth_images)} Depth-Bilder")
    
    # Paare bilden (Color und Depth mit gleicher Nummer)
    # Dictionary für schnellen Zugriff auf depth Bilder
    depth_dict = {img.stem: img for img in depth_images}
    
    # Bilder gruppieren (je 3 Bilder pro plant)
    grouped_images = []
    current_group = []
    
    for color_img in sorted(color_images, key=lambda x: int(x.stem.split('_')[1])):
        img_num = int(color_img.stem.split('_')[1])
        depth_img = depth_dict.get(f"depth_{img_num}")
        
        if depth_img:
            current_group.append((color_img, depth_img))
            
            # Wenn wir 3 Paare haben, zur Gruppe hinzufügen
            if len(current_group) == 3:
                grouped_images.append(current_group)
                current_group = []
    
    # Falls eine unvollständige Gruppe übrig bleibt
    if current_group:
        print(f"⚠️  Letzte Gruppe hat nur {len(current_group)} Bilder (ignoriert)")
    
    print(f"📦 Bild-Gruppen: {len(grouped_images)}")
    
    # Bilder kopieren und Ordner erstellen
    for group_index, image_group in enumerate(grouped_images):
        plant_name = f"plant_{group_index:03d}"  # plant_000, plant_001, etc.
        plant_dir = dest_base / plant_name
        plant_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Erstelle: {plant_name}")
        
        for view_index, (color_img, depth_img) in enumerate(image_group):
            # Neue Dateinamen
            new_color_name = f"color_{view_index}.png"
            new_depth_name = f"depth_{view_index}.png"
            
            # Zielpfade
            color_dest = plant_dir / new_color_name
            depth_dest = plant_dir / new_depth_name
            
            # Dateien kopieren
            shutil.copy2(color_img, color_dest)
            shutil.copy2(depth_img, depth_dest)
            
            print(f"   → {new_color_name} (von {color_img.name})")
            print(f"   → {new_depth_name} (von {depth_img.name})")
    
    print(f"\n✅ Fertig! {len(grouped_images)} Pflanzen-Ordner erstellt in:")
    print(f"   {dest_base}")

def main():
    print("=" * 60)
    print("🌱 STRAWBERRY WS - BILDER ORGANISIEREN")
    print("=" * 60)
    
    organize_images()
    
    print("=" * 60)

if __name__ == "__main__":
    main()