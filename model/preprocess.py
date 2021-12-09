import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import config

# def preprocess_w_valid(directory):
#     train_image, train_label, test_image, test_label = [], [], [], []
#     for file in tqdm(glob(directory + "/train/*/*.png")): # windows sys: tqdm(glob(directory + "/train\*\*.png"))
#         train_label = file.split("/")[-2] # windows sys: file.split("\\")[-2] 
#         image = cv2.resize(cv2.imread(file), (config.image_height, config.image_width)) # resize
#         train_image.append(image/255.0) # normalize
#         train_label.append(config.lbl2id[train_label])

#     for file in tqdm(glob(directory + "/test/*/*.png")): # windows sys: tqdm(glob(directory + "/test\*\*.png"))
#         test_label = file.split("/")[-2] # windows sys: file.split("\\")[-2] 
#         image = cv2.resize(cv2.imread(file), (config.image_height, config.image_width)) # resize
#         test_image.append(image/255.0) # normalize
#         test_label.append(config.lbl2id[test_label])

#     train_image = np.array(train_image)
#     train_label = np.array(train_label)
#     train_label = to_categorical(train_label, num_classes=config.NUM_CLASSES, dtype='uint8') # one-hot
#     train_image, train_label, valid_image, valid_label = train_test_split(train_image, train_label, test_size=0.33, random_state=42)
    
#     test_image = np.array(test_image)
#     test_label = np.array(test_label)
#     test_label = to_categorical(test_label, num_classes=config.NUM_CLASSES, dtype='uint8') # one-hot
    
#     return train_image, train_label, valid_image, valid_label, test_image, test_label


def preprocess_wo_valid(directory):
    train_image, train_label, test_image, test_label = [], [], [], []
    for file in tqdm(glob(directory + "/train/*/*.png")): # windows sys: tqdm(glob(directory + "/train\*\*.png"))
        label = file.split("/")[-2] # windows sys: file.split("\\")[-2] 
        image = cv2.resize(cv2.imread(file), (config.image_height, config.image_width)) # resize
        train_image.append(image/255.0) # normalize
        train_label.append(config.lbl2id[label])

    for file in tqdm(glob(directory + "/test/*/*.png")): # windows sys: tqdm(glob(directory + "/test\*\*.png"))
        label = file.split("/")[-2] # windows sys: file.split("\\")[-2] 
        image = cv2.resize(cv2.imread(file), (config.image_height, config.image_width)) # resize
        test_image.append(image/255.0) # normalize
        test_label.append(config.lbl2id[label])

    train_image = np.array(train_image)
    train_label = np.array(train_label)
    train_label = to_categorical(train_label, num_classes=config.NUM_CLASSES, dtype='uint8') # one-hot
    
    test_image = np.array(test_image)
    test_label = np.array(test_label)
    test_label = to_categorical(test_label, num_classes=config.NUM_CLASSES, dtype='uint8') # one-hot
    
    return train_image, train_label, test_image, test_label

def to_batch(images, labels):
    DataGen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True)
    examples_in_batch = DataGen.flow(images, labels, batch_size=config.BATCH_SIZE) # train_lab is categorical 
    return examples_in_batch