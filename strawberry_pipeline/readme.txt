perer data 

perp data format 
conda run -n forschsem python tools/convert_dataset.py \
  data/nichtmehrkrisebilder \
  data/prepared_nichtmehrkrisebilder \
  --overwrite


  medium redboost:
  conda run -n forschsem python tools/redboost_dataset.py \
  --src data/prepared_nichtmehrkrisebilder \
  --dst data/prepared_nichtmehrkrisebilder_medium_redboost \
  --sat-scale 1.45 \
  --red-gain 1.30 \
  --overwrite



  run it on medium redboost:
  conda run -n forschsem python -m strawberry_py.main \
  --config configs/meca_d405.yaml \
  --dataset_root data/prepared_nichtmehrkrisebilder_medium_redboost \
  --out_root outputs/test_nichtmehrkrisebilder_medium_redboost