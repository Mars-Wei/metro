export SEQ_GRAPH_RETRO=/path/to/dir/
python setup.py develop

# Data Process
python json2csv.py

python data_process/canonicalize_prod.py --filename train.csv
python data_process/canonicalize_prod.py --filename eval.csv
python data_process/canonicalize_prod.py --filename test.csv

python data_process/core_edits/bond_edits.py

python data_process/lg_edits/lg_classifier.py
python data_process/lg_edits/lg_tensors.py

# Train
python scripts/benchmarks/run_model.py --config_file configs/single_edit/defaults.yaml
python scripts/benchmarks/run_model.py --config_file configs/lg_ind/defaults.yaml

# Test
bash eval.sh
python scripts/eval/top5_inference.py --edits_exp SingleEdit_20220823_044246 --lg_exp LGIndEmbed_20220823_04432 --edits_step best_model --lg_step best_model --exp_dir models
