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
    source_folder1: str,
    source_folder2: str,
    mix_percentage_from_dataset1: float,
    subset_name: str,
    base_dir: str = "mixed_datasets",
    downsize_dataset_length: float = 0.0,
):
    """
    Mischt Bilddateien aus zwei Quellordnern basierend auf einem Prozentsatz
    und speichert die gemischten Bilder in einem neuen Ordner.

    """

    if not (0 <= mix_percentage_from_dataset1 <= 1):
        raise ValueError("Der Mischprozentsatz muss zwischen 0 und 1 liegen.")

    # Sammeln aller PNG-Dateien aus den Quellordnern
    def get_image_files(folder_path):
        image_files = []
        for f in os.listdir(folder_path):
            lower_f = f.lower()
            if lower_f.endswith('.png') or lower_f.endswith('.jpg') or lower_f.endswith('.jpeg'):
                image_files.append(os.path.join(folder_path, f))
        return image_files
    
    images1 = get_image_files(source_folder1)
    images2 = get_image_files(source_folder2)

    if not images1:
        print(f"Keine PNG-Bilder im Ordner '{source_folder1}' gefunden.")
        return
    if not images2:
        print(f"Keine PNG-Bilder im Ordner '{source_folder2}' gefunden.")
        return
    
    random.shuffle(images1)
    random.shuffle(images2)

    num_images1 = len(images1)
    num_images2 = len(images2)
    new_dataset_size = int(num_images1/mix_percentage_from_dataset1)  # Größe des neu erstellten Datensatzes

    num_from_images2_needed = new_dataset_size-num_images1


    print(f"RH-Datensatz hat {num_images1} Bilder.")
    print(f"Hydro-Datensatz hat {num_images2} Bilder.")

    # Prüfen, ob die Anzahl der Bilder in den Quellordnern ausreicht
    if num_from_images2_needed > len(images2):
        print(f"Warnung: nicht genügend Bilder im zweiten Quellordner '{source_folder2}' vorhanden um neuen Datensatz mit '{1-mix_percentage_from_dataset1}'% zu füllen.")

        """ HIER MUSS NOCH DOWNSIZE LOGIK HINZUGEFÜGT WERDEN"""


    # Auswahl der Bilder basierend auf dem Mischprozentsatz
    selected_images_ds1 = images1
    selected_images_ds2 = images2[:num_from_images2_needed]

    all_selected_images = selected_images_ds1 + selected_images_ds2

    # Optional: Mischen der ausgewählten Bilder
    # random.shuffle(all_selected_images) 

    # Generieren des Ausgabepfadnamens
    percentage1_str = f"{int(mix_percentage_from_dataset1 * 100)}"
    percentage2_str = f"{int((1 - mix_percentage_from_dataset1) * 100)}"
    
    # Namensgebung neuer Ordner
    new_dataset_folder_name = f"RH{percentage1_str}_Hydro{percentage2_str}"
    output_path = os.path.join(base_dir, new_dataset_folder_name, "Data_master", subset_name)

    # Erstellen des Ausgabeordners
    os.makedirs(output_path, exist_ok=True)
    print(f"Erstelle Ausgabeordner: {output_path}")

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

    return os.path.join(base_dir, new_dataset_folder_name)

def split_dataset(base_dir_mix_dataset: str, subset_type1: str, subset_type2: str, split_percentage: float = 0.8):
    """
    Teilt 'Normal'-Bilder auf in Train/Test nach dem gegebenen Prozentsatz.
    Fügt zusätzlich eine gleiche Anzahl von 'Anomalous'-Bildern in den Test-Ordner hinzu.
    
    Args:
        base_dir_mix_dataset (str): Basisordner des gemischten Datensatzes.
        subset_type1 (str): Name des 'Normal'-Ordners.
        subset_type2 (str): Name des 'Anomalous'-Ordners.
        split_percentage (float): Anteil der 'Normal'-Bilder, die für das Training verwendet werden (zwischen 0 und 1).
    """
    # Quellpfade
    source_folder_normal = os.path.join(base_dir_mix_dataset, "Data_master", subset_type1)
    source_folder_anomalous = os.path.join(base_dir_mix_dataset, "Data_master", subset_type2)

    # Zielpfade
    target_train_normal = os.path.join(base_dir_mix_dataset, "Data_train_test", "Train", "Normal")
    target_test_normal = os.path.join(base_dir_mix_dataset, "Data_train_test", "Test", "Normal")
    target_test_anomalous = os.path.join(base_dir_mix_dataset, "Data_train_test", "Test", "Anomalous")

    os.makedirs(target_train_normal, exist_ok=True)
    os.makedirs(target_test_normal, exist_ok=True)
    os.makedirs(target_test_anomalous, exist_ok=True)

    # Dateien in Normal-Verzeichnis
    normal_files = [f for f in os.listdir(source_folder_normal) if os.path.isfile(os.path.join(source_folder_normal, f))]
    random.shuffle(normal_files)

    split_index = int(len(normal_files) * split_percentage)
    train_normal_files = normal_files[:split_index]
    test_normal_files = normal_files[split_index:]

    # Train/Normal kopieren
    for f in train_normal_files:
        shutil.copy2(os.path.join(source_folder_normal, f), os.path.join(target_train_normal, f))

    # Test/Normal kopieren
    for f in test_normal_files:
        shutil.copy2(os.path.join(source_folder_normal, f), os.path.join(target_test_normal, f))

    print(f"→ 'Normal': {len(train_normal_files)} Bilder in Train, {len(test_normal_files)} in Test.")

    # Jetzt gleiche Anzahl an Anomalous-Bildern wie train_normal_files in Test/Anomalous kopieren
    anomalous_files = [f for f in os.listdir(source_folder_anomalous) if os.path.isfile(os.path.join(source_folder_anomalous, f))]
    random.shuffle(anomalous_files)

    num_anomalous_needed = len(train_normal_files)
    selected_anomalous_files = anomalous_files[:num_anomalous_needed]

    for f in selected_anomalous_files:
        shutil.copy2(os.path.join(source_folder_anomalous, f), os.path.join(target_test_anomalous, f))

    print(f"→ 'Anomalous': {len(selected_anomalous_files)} Bilder in Test (entspricht Anzahl Train/Normal).")

    print("\nSplit abgeschlossen.")
    print(f"File: Train/Normal: {len(train_normal_files)}")
    print(f"File: Test/Normal: {len(test_normal_files)}")
    print(f"File: Test/Anomalous: {len(selected_anomalous_files)}")


if __name__ == "__main__":
    # Pfade zu den Quellordnern
    # Alle Bilder für Normal bzw. Anomalous Daten müssen vorher in den angegebenen Ordnern liegen
    # Daten werden anschließend gesplitet in train und test 
    RH_source_normal = "C:/Users/Lenovo/Documents/02_repos/Forschungsseminar/yolo11_riseholme/Riseholme-2021/Data/Normal/Ripe"
    Hydro_source_normal = "C:/Users/Lenovo/Downloads/erdbeeren/erdbeeren/train"

    RH_source_anomalous ="C:/Users/Lenovo/Documents/02_repos/Forschungsseminar/yolo11_riseholme/Riseholme-2021/Data/Anomalous"
    Hydro_source_anomalous = "C:/Users/Lenovo/Downloads/erdbeeren/erdbeeren/test"

    # Angeben ob Normal oder Anomalous
    subset_type_normal = "Normal"
    subset_type_anomalous = "Anomalous"

    # Der gewünschte Mix-Prozentsatz (z.B. 0.6 für 60% von RH_source_folder und 40% von Hydro_source_folder)
    mix_percentage_from_dataset1 = 0.9

    # NOCH IN ARBEIT bzw. wird gemacht, sofern notwendig - Optional: Datasetsize downsizen 
    #downsize_dataset_length = 0.5 # z.B. 0.5 für 50% der Bilder aus dem ersten Ordner

    ''' 
    ### Hier wird der Mix der beiden Datensätze erstellt

    Der Mix wird in einem neuen Ordner im aktuellen Verzeichnis gespeichert
    Für den neuen Datenmix wird automatisch ein neuer Ordner mit entsprechendem Namen generiert

    '''
    generated_dataset_normal_path = mix_image_datasets(
        source_folder1=RH_source_normal,
        source_folder2=Hydro_source_normal,
        subset_name=subset_type_normal,
        #downsize_dataset_length=downsize_dataset_length,
        mix_percentage_from_dataset1=mix_percentage_from_dataset1,
    )

    generated_dataset_anomalous_path = mix_image_datasets(
        source_folder1=RH_source_anomalous,
        source_folder2=Hydro_source_anomalous,
        subset_name=subset_type_anomalous,
        #downsize_dataset_length=downsize_dataset_length,
        mix_percentage_from_dataset1=mix_percentage_from_dataset1,
    )
    split_dataset(
        base_dir_mix_dataset= generated_dataset_normal_path,
        subset_type1=subset_type_normal,
        subset_type2=subset_type_anomalous,
        split_percentage=0.8
    )

