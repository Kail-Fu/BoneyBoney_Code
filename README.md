This is a re-implementatino of https://github.com/hszhao/SAN/blob/master/model/san.py SAN model. We translated this model to tensorflow and used it to classify bone x-ray photos, achieving a 78.9% accuracy.

# Run on sample_dataset
Command "python model/train.py" to run on sample_dataset of 2000 images (training + testing combined)

# Run on whole_dataset
1. Ask permission from https://stanfordmlgroup.github.io/competitions/mura/
2. !curl https://cs.stanford.edu/group/mlgroup/MURA-v1.1.zip --output m.zip
3. !unzip m.zip
4. !mkdir whole_dataset
5. !mkdir whole_dataset/train
6. !mkdir whole_dataset/test
7. !python mura_dataset_reorganize.py
8. Finally, change directory in model/train.py from sample_dataset to whole_dataset

# How to tune
1. Tune hyper parameters in config.py
2. You can tune kernal size and layer size in model/train.py
3. You can tune channels in model/san.py

