# Metro: Memory-Enhanced Transformer for Retrosynthetic Planning via Reaction Tree

This repository contains an implementation of ["Metro: Memory-Enhanced Transformer for Retrosynthetic Planning via Reaction Tree"](https://arxiv.org/pdf/2109.03856.pdf).

Note that we provide the model files for all models, you can skip the training and use our provided files for testing directly.



## Dropbox
We provide the starting material file in dropbox, you can download this file via:
https://www.dropbox.com/s/nwh2ijrjzbyia73/zinc_stock_17_04_20.hdf5?dl=0

Please move this file into the root folder.




## Metro

cp train_dataset.json valid_data.json test_data.json zinc_stock_17_04_20.hdf5 Metro/
cd Metro

**Data Process**

python to_canolize.py

**Train**

python train.py --batch_size 64 --epochs 4000

**Test**

python top5_inference.py




## Transformer

cp train_dataset.json valid_data.json test_data.json zinc_stock_17_04_20.hdf5 Transformer/
cd Transformer

**Data Process**

python to_canolize.py

**Train**
python train.py --batch_size 64 --epochs 2000

**Test**

python top5_inference.py




## Retrosim

cp train_dataset.json valid_data.json test_data.json zinc_stock_17_04_20.hdf5 Retrosim/
cd Retrosim

**Test**

python top5_inference.py




## Neuralsym

cp train_dataset.json valid_data.json test_data.json zinc_stock_17_04_20.hdf5 Neuralsym/

**Model download**

Download the model folder via:
https://www.dropbox.com/sh/lw8e6epk74vmsh0/AADr0VhLiA6LwC34_Qj0Naufa?dl=0
mv checkpoint Neuralsym/
cd Neuralsym

**Data Process**

python prepare_data.py

**Train**

bash train.sh

**Test**

python top5_inference.py




## GLN

cp train_dataset.json valid_data.json test_data.json zinc_stock_17_04_20.hdf5 GLN/gln/
cd GLN
pip install -e
cd gln

**Data Process**

python process_data.py -save_dir data -num_cores 12 -num_parts 1 -fp_degree 2 -f_atoms data/atom_list.txt -retro_during_train False $@

**Train**

bash run_mf.sh schneider

**Test**

python3 top5_inference.py -save_dir data -f_atoms data/atom_list.txt -gpu 0




## Megan

cp train_dataset.json valid_data.json test_data.json zinc_stock_17_04_20.hdf5 Megan/data/

**Model download**

Download the model folder via:
https://www.dropbox.com/sh/aq9h5jyautjpgrh/AAAv_cC2CiQtYTbP7IKFqQcMa?dl=0
mv models Megan/
cd Megan
source env.sh

**Data Process**

python json2csv.py
python acquire.py uspto_50k
python featurize.py uspto_50k megan_16_bfs_randat

**Train**

python bin/train.py uspto_50k models/uspto_50k

**Test**

python bin/top5_inference.py models/uspto_50k --beam-size 5




## GraphRetro

cp train_dataset.json valid_data.json test_data.json zinc_stock_17_04_20.hdf5 GraphRetro/datasets/uspto-50k
cd GraphRetro
export SEQ_GRAPH_RETRO=/path/to/dir/
python setup.py develop

**Data Process**

python json2csv.py

python data_process/canonicalize_prod.py --filename train.csv
python data_process/canonicalize_prod.py --filename eval.csv
python data_process/canonicalize_prod.py --filename test.csv

python data_process/core_edits/bond_edits.py

python data_process/lg_edits/lg_classifier.py
python data_process/lg_edits/lg_tensors.py

**Train**

python scripts/benchmarks/run_model.py --config_file configs/single_edit/defaults.yaml
python scripts/benchmarks/run_model.py --config_file configs/lg_ind/defaults.yaml

**Test**

bash eval.sh
python scripts/eval/top5_inference.py --edits_exp SingleEdit_20220823_044246 --lg_exp LGIndEmbed_20220823_04432 --edits_step best_model --lg_step best_model --exp_dir models




## G2Gs

cp train_dataset.json valid_data.json test_data.json zinc_stock_17_04_20.hdf5 G2Gs/datasets/

**Model download**

Download the model files via:
https://www.dropbox.com/s/bvrsun8yap4sal9/g2gs_reaction_model.pth?dl=0
https://www.dropbox.com/s/0n2ff4zbnbml6xq/g2gs_synthon_model.pth?dl=0

mv g2gs_synthon_model.pth g2gs_reaction_model.pth G2Gs/
cd G2Gs

**Train**

**Test**

python script/top5_inference.py -g [0] -k 5 -b 1




## Reference
Retrosim: https://github.com/connorcoley/retrosim
Neuralsym: https://github.com/linminhtoo/neuralsym
GLN: https://github.com/Hanjun-Dai/GLN
G2Gs: https://torchdrug.ai/docs/tutorials/retrosynthesis
GraphRetro: https://github.com/vsomnath/graphretro
Transformer: https://github.com/bigchem/synthesis
Megan: https://github.com/molecule-one/megan
