#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from pathlib import Path
import shutil
target_dir = Path("/home/kailiang_fu/BoneyBoney/whole_dataset/train")
for body_part in ["ELBOW", "FINGER", "FOREARM", "HAND", "HUMERUS"]:
    body_part_dir = target_dir.joinpath(Path(body_part))
    os.mkdir(body_part_dir)
    train_dir = Path('/home/kailiang_fu/MURA-v1.1/train/XR_' + body_part)
    counter = 0
    for sub_dir in os.listdir(train_dir):
        sub_dir = train_dir.joinpath(sub_dir)
        for study_dir in os.listdir(sub_dir):
            study_dir = Path(sub_dir).joinpath(study_dir)
            try:
                for file_name in os.listdir(study_dir):
                    assert file_name.endswith('png')

                    src_file_path = study_dir.joinpath(file_name)
                    new_file_name = f"{counter}.png"

                    image_dst = body_part_dir.joinpath(Path(f"{counter}.png"))
                    shutil.copy(src_file_path, image_dst)

                    counter += 1

            except NotADirectoryError:
                pass


# In[3]:


target_dir = Path("/home/kailiang_fu/BoneyBoney/whole_dataset/test")
for body_part in ["ELBOW", "FINGER", "FOREARM", "HAND", "HUMERUS"]:
    body_part_dir = target_dir.joinpath(Path(body_part))
    os.mkdir(body_part_dir)
    train_dir = Path('/home/kailiang_fu/MURA-v1.1/valid/XR_'+ body_part)
    counter = 0
    for sub_dir in os.listdir(train_dir):
        sub_dir = train_dir.joinpath(sub_dir)
        for study_dir in os.listdir(sub_dir):
            study_dir = Path(sub_dir).joinpath(study_dir)
            try:
                for file_name in os.listdir(study_dir):
                    assert file_name.endswith('png')

                    src_file_path = study_dir.joinpath(file_name)
                    new_file_name = f"{counter}.png"

                    image_dst = body_part_dir.joinpath(Path(f"{counter}.png"))
                    shutil.copy(src_file_path, image_dst)

                    counter += 1

            except NotADirectoryError:
                pass
