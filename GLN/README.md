cd gln
pip install -e

# Data Process
python process_data.py -save_dir data -num_cores 12 -num_parts 1 -fp_degree 2 -f_atoms data/atom_list.txt -retro_during_train False $@

# Train
bash run_mf.sh schneider

# Test
python3 top5_inference.py -save_dir data -f_atoms data/atom_list.txt -gpu 0