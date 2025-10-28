Datensatz StrawDI_Db1: https://datasetninja.com/strawdi
Hier gesehen: https://www.sciencedirect.com/science/article/abs/pii/S0168169920300624

quelle zu Dataset
@article{PEREZBORRERO2020105736,
title = "A fast and accurate deep learning method for strawberry instance segmentation",
journal = "Computers and Electronics in Agriculture",
volume = "178",
pages = "105736",
year = "2020",
issn = "0168-1699",
doi = "https://doi.org/10.1016/j.compag.2020.105736",
url = "http://www.sciencedirect.com/science/article/pii/S0168169920300624",
author = "Isaac Pérez-Borrero and Diego Marín-Santos and Manuel E. Gegúndez-Arias and Estefanía Cortés-Ancos"
}





Setup: 
Get the data from:
https://strawdi.github.io/
https://drive.google.com/file/d/1elFB-q9dgPbfnleA7qIrTb96Qsli8PZl/view 

-> Put it into data: StrawDI_Db1/test and so on 
-> run tools converter_strawdi.py to convert the labels for yolo anc coco
-> run view_yolo_seg.py to check if it works (check folder debug_vis)



RUn yolo 
start with python SegemtationTrainig_v1/models/train_yolo.py
set USE_CUDA to true for compleete training with gpu
run it with USE_CUDA false to run it with subset und cpu 

USE_CUDA = True
GPU_DEVICE = "0"   # falls mehrere GPUs, ggf. anpassen
EPOCHS = 100       # oder mehr
BATCH = 16         # je nach VRAM erhöhen
WORKERS = 8        # >0 für schnellere IO auf GPU