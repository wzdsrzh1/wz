import os
import json

class args:

    train_modality1_dir = './data/train/PET' #(eg. PET)
    train_modality2_dir = './data/train/MRI' #(eg. MRI)
    batch_size = 6
    save_dir = './results'
    lr = 1e-4
    epochs = 100