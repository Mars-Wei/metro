source env.sh

# Data Process
python json2csv.py
python acquire.py uspto_50k
python featurize.py uspto_50k megan_16_bfs_randat

# Train
python bin/train.py uspto_50k models/uspto_50k

# Test
python bin/top5_inference.py models/uspto_50k --beam-size 5