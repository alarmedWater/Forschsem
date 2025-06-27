import os
import random
import shutil

'''
Dieses Skript kombiniert und verarbeitet zwei Bilddatensätze (z. B. Riseholme und Hydro),
um ein gemischtes Trainings-/Test-Setup für ein Klassifikationsproblem zu erstellen.

----------------------------------------
Funktionen:
----------------------------------------

1. mix_image_datasets():
   - Mischt Bilder aus zwei Quellordnern mit einem festgelegten Verhältnis.
   - Speichert die gemischten Bilder im Ordner: mixed_datasets/RHX_HydroY/Data_master

2. split_dataset():
   - Teilt die gemischten Normal-Bilder in Training und Test auf.
   - Fügt eine gleich große Menge an Anomalous-Bildern zum Testset hinzu.

----------------------------------------
Anwendungsschritte:
----------------------------------------

1. Quellpfade in main festlegen:
    - RH_source_normal:      Normal-Bilder aus Datensatz 1
    - Hydro_source_normal:   Normal-Bilder aus Datensatz 2
    - RH_source_anomalous:   Anomalous-Bilder aus Datensatz 1
    - Hydro_source_anomalous:Anomalous-Bilder aus Datensatz 2

2. Mischverhältnis setzen:
    - Beispiel: mix_percentage_from_dataset1 = 0.9 (→ 90% RH, 10% Hydro)

3. NOCH IN BEARBEITUNG - (Optional) Datensatzgröße reduzieren:
    - downsize_dataset_length = 0.5 z.B. für 50% der Bilder

4. Skript ausführen:
    - Erstellt automatisch folgende Ordner:
        mixed_datasets/RH<X>_Hydro<X>/Data_master
        mixed_datasets/RH<X>_Hydro<X>/Data_train_test/Train/Normal
        mixed_datasets/RH<X>_Hydro<X>/Data_train_test/Test/Normal
        mixed_datasets/RH<X>_Hydro<X>/Data_train_test/Test/Anomalous

Benennung des Ordners erfolgt automatisch auf dem gewählten Mischverhältnis!
'''

def mix_image_datasets(
    base_dir_rh: str,
    base_dir_hydro: str,
    mix_percentage: float,
    subset_name: str,
    new_base_dir: str = "mixed_datasets",
    downsize_dataset_length: float = 0.0,
):
    """
    Mischt Bilddateien aus zwei Quellordnern basierend auf einem Prozentsatz
    und speichert die gemischten Bilder in einem neuen Ordner.

    """

    if not (0 <= mix_percentage <= 1):
        raise ValueError("Der Mischprozentsatz muss zwischen 0 und 1 liegen.")

    # Sammeln aller PNG-Dateien aus den Quellordnern
    def get_image_files(folder_path):
        image_files = []
        for f in os.listdir(folder_path):
            lower_f = f.lower()
            if lower_f.endswith('.png') or lower_f.endswith('.jpg') or lower_f.endswith('.jpeg'):
                image_files.append(os.path.join(folder_path, f))
        return image_files
    
    target_dir_rh = os.path.join(base_dir_rh, subset_name)
    target_dir_hydro = os.path.join(base_dir_hydro, subset_name)

    images_rh = get_image_files(target_dir_rh)
    images_hydro = get_image_files(target_dir_hydro)

    if not images_rh:
        print(f"Keine PNG-Bilder im Ordner '{base_dir_rh}' gefunden.")
        return
    if not images_hydro:
        print(f"Keine PNG-Bilder im Ordner '{base_dir_hydro}' gefunden.")
        return
    
    random.shuffle(images_rh)
    random.shuffle(images_hydro)

    num_images_rh = len(images_rh)
    num_images_hydro = len(images_hydro)
    new_dataset_size = min(num_images_rh, num_images_hydro)  # Größe des neu erstellten Datensatzes wird gecappt auf die maximale Bildanzahl einer der beiden Datensätze

    num_images_from_rh_needed = int(new_dataset_size * mix_percentage)
    num_images_from_hydro_needed = new_dataset_size - num_images_from_rh_needed 

    print(f"RH-Datensatz hat {num_images_rh} Bilder.")
    print(f"Davon werden {num_images_from_rh_needed} Bilder verwendet.")
    print(f"\nHydro-Datensatz hat {num_images_hydro} Bilder.")
    print(f"Davon werden {num_images_from_hydro_needed} Bilder verwendet.")

    # Prüfen, ob die Anzahl der Bilder in den Quellordnern ausreicht
    if num_images_from_hydro_needed > len(images_hydro):
        print(f"Warnung: nicht genügend Bilder im zweiten Quellordner '{base_dir_hydro}' vorhanden um neuen Datensatz mit '{1-mix_percentage}'% zu füllen.")

        """ HIER MUSS NOCH DOWNSIZE LOGIK HINZUGEFÜGT WERDEN"""


    # Auswahl der Bilder basierend auf dem Mischprozentsatz
    selected_images_rh = images_rh[:num_images_from_rh_needed]
    selected_images_hydro = images_hydro[:num_images_from_hydro_needed]

    # Generieren des Ausgabepfadnamens
    percentage_rh_str = f"{int(mix_percentage * 100)}"
    percentage_hydro_str = f"{int((1 - mix_percentage) * 100)}"

    # Namensgebung neuer Ordner
    new_dataset_folder_name = f"RH{percentage_rh_str}_Hydro{percentage_hydro_str}"
    output_path_base_dir = os.path.join(new_base_dir, new_dataset_folder_name, "Data_master", subset_name)
            
    os.makedirs(output_path_base_dir, exist_ok=True)
    print(f"Erstelle Ausgabeordner: {output_path_base_dir}")

    output_path_hydro = os.path.join(output_path_base_dir, "Hydro")
    output_path_rh = os.path.join(output_path_base_dir, "RH")

    os.makedirs(output_path_hydro, exist_ok=True)
    print(f"Erstelle Ausgabeordner: {output_path_hydro}")
    os.makedirs(output_path_rh, exist_ok=True)
    print(f"Erstelle Ausgabeordner: {output_path_rh}")

    def copy_images(selected_images, output_path, source_folder):
        """
        Kopiert die ausgewählten Bilder in den angegebenen Ausgabepfad.
        """
        copied_count = 0
        for original_path in selected_images:
            filename = os.path.basename(original_path)
            destination_path = os.path.join(output_path, filename)
            try:
                shutil.copy2(original_path, destination_path)
                copied_count += 1
            except Exception as e:
                print(f"Fehler beim Kopieren von '{original_path}' nach '{destination_path}': {e}")
    
        print(f"\nMischvorgang abgeschlossen. {copied_count} Bilder wurden in '{output_path}' kopiert.")
        print(f"  Davon {len(selected_images)} aus '{source_folder}'.")

    copy_images(selected_images_rh, output_path_rh, base_dir_rh)
    copy_images(selected_images_hydro, output_path_hydro, base_dir_hydro)

    '''
    # Kopieren der ausgewählten Bilder
    copied_count = 0
    for original_path in all_selected_images:
        filename = os.path.basename(original_path)
        # Um Duplikate zu vermeiden, wenn Dateinamen in beiden Source-Ordnern gleich sind,
        # könnte man hier einen Präfix hinzufügen (z.B. "ds1_" oder "ds2_")
        # For this version, we assume unique filenames or handle potential overwrites by latest copy.
        # If filenames are not unique, a simple solution is to prepend the source folder name:
        # new_filename = f"{os.path.basename(os.path.dirname(original_path))}_{filename}"
        # destination_path = os.path.join(output_path, new_filename)
        
        destination_path = os.path.join(output_path, filename)
        
        try:
            shutil.copy2(original_path, destination_path)
            copied_count += 1
        except Exception as e:
            print(f"Fehler beim Kopieren von '{original_path}' nach '{destination_path}': {e}")


    print(f"\nMischvorgang abgeschlossen. {copied_count} Bilder wurden in '{output_path}' kopiert.")
    print(f"  Davon {len(selected_images_ds1)} aus '{source_folder1}' und {len(selected_images_ds2)} aus '{source_folder2}'.")
    '''

    return os.path.join(new_base_dir, new_dataset_folder_name)

def split_dataset(base_dir_mix_dataset: str, source_hydro: str, subset_type_normal: str, subset_type_anomalous: str, split_percentage: float = 0.8, test_size_each: int = 300):
    """
    Teilt 'Normal'-Bilder auf in Train/Test nach dem gegebenen Prozentsatz.
    Fügt zusätzlich eine gleiche Anzahl von 'Anomalous'-Bildern in den Test-Ordner hinzu.

    Testdaten sind in dieser Logik nur Hydrodaten.
    
    """
    # Quellpfade
    source_folder__rh_normal = os.path.join(base_dir_mix_dataset, "Data_master", subset_type_normal, "RH")
    source_folder_hydro_normal = os.path.join(base_dir_mix_dataset, "Data_master", subset_type_normal, "Hydro")

    source_folder_hydro_anomalous_test = os.path.join(source_hydro, subset_type_anomalous)  # Hier ist path zu den Hydro-Daten für Anomalous Testdaten
    source_folder_hydro_normal_test_data = os.path.join(source_hydro, subset_type_normal)    # Hier ist path zu den not-seen Hydro-Daten nur zum Testen notwendig

    # Zielpfade
    target_train_normal = os.path.join(base_dir_mix_dataset, "Data_train_test", "Train", "Normal")
    target_test_normal = os.path.join(base_dir_mix_dataset, "Data_train_test", "Test", "Normal")
    target_test_anomalous = os.path.join(base_dir_mix_dataset, "Data_train_test", "Test", "Anomalous")

    os.makedirs(target_train_normal, exist_ok=True)
    os.makedirs(target_test_normal, exist_ok=True)
    os.makedirs(target_test_anomalous, exist_ok=True)

    # Alle Dateien aus den beiden Normal-Verzeichnissen sammeln
    normal_files_all_train = [
    os.path.join(source_folder__rh_normal, f)
    for f in os.listdir(source_folder__rh_normal)
    if os.path.isfile(os.path.join(source_folder__rh_normal, f))
    ]

    normal_files_all_train += [
    os.path.join(source_folder_hydro_normal, f)
    for f in os.listdir(source_folder_hydro_normal)
    if os.path.isfile(os.path.join(source_folder_hydro_normal, f))
    ]

    random.shuffle(normal_files_all_train)  # Mischen der Dateien für zufällige Auswahl

    # Testdaten festlegen 
    normal_files_hydro_test = [f for f in os.listdir(source_folder_hydro_normal_test_data) if os.path.isfile(os.path.join(source_folder_hydro_normal_test_data, f))]
    anomalous_files_hydro_test = [f for f in os.listdir(source_folder_hydro_anomalous_test) if os.path.isfile(os.path.join(source_folder_hydro_anomalous_test, f))]

    # Train/Normal kopieren
    for file_path in normal_files_all_train:
        shutil.copy2(file_path, target_train_normal)
    print(f"→ 'Normal': {len(normal_files_all_train)} Bilder in Train.")    

    # Test/Normal kopieren
    for file_path in normal_files_hydro_test:
        shutil.copy2(file_path, target_test_normal)
    print(f"→ 'Normal': {len(normal_files_hydro_test)} Bilder in Test.")

    # Test/Anomalous kopieren
    for file_path in anomalous_files_hydro_test:
        shutil.copy2(file_path, target_test_anomalous)
    print(f"→ 'Anomalous': {len(anomalous_files_hydro_test)} Bilder in Test.")


if __name__ == "__main__":
    # Pfade zu den Quellordnern
    # Alle Bilder für Normal bzw. Anomalous Daten müssen vorher in den angegebenen Ordnern liegen
    # Daten werden anschließend gesplitet in train und test 
    RH_source = "C:/Users/Lenovo/Documents/02_repos/Forschungsseminar/yolo11_riseholme/Riseholme-2021/Data/Normal/Ripe"
    Hydro_source = "C:/Users/Lenovo/Downloads/erdbeeren/erdbeeren/train"

    # Angeben ob Normal oder Anomalous
    subset_type_normal = "Normal"
    subset_type_anomalous = "Anomalous"

    # Der gewünschte Mix-Prozentsatz (z.B. 0.6 für 60% von RH_source_folder und 40% von Hydro_source_folder)
    mix_percentage_from_dataset1 = 0.9

    # NOCH IN ARBEIT bzw. wird gemacht, sofern notwendig - Optional: Datasetsize downsizen 
    #downsize_dataset_length = 0.5 # z.B. 0.5 für 50% der Bilder aus dem ersten Ordner

    generated_dataset_normal_path = mix_image_datasets(
        base_dir_rh=RH_source,
        base_dir_hydro=Hydro_source,
        subset_name=subset_type_normal,
        mix_percentage=mix_percentage_from_dataset1,
        #downsize_dataset_length=downsize_dataset_length,  # Optional
    )


