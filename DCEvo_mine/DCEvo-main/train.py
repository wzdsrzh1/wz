import os
import warnings

# 禁用TensorFlow日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 过滤特定警告
warnings.filterwarnings('ignore',
                       message=".*tf.losses.sparse_softmax_cross_entropy.*")
warnings.filterwarnings('ignore',
                       category=FutureWarning,
                       module='tensorflow')

# 然后导入TensorFlow
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from sleepnet import (DE_Encoder, DE_Decoder, LowFreqExtractor, HighFreqExtractor,
                        CrossAttentionBlock,prominent)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss_fusion import Fusionloss, cc
import kornia
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
import warnings
import logging
from models.common import DetectMultiBackend
from utils.img_read_save import img_save,image_read_cv2
import pygad

import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import PVT_Classification

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_img_size,
                           check_suffix, check_yaml, colorstr, get_latest_run, increment_path, init_seeds,
                           intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, one_flat_cycle, print_args, print_mutation, strip_optimizer, yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss_tal import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP,
                               smart_optimizer, smart_resume, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = None

class trainer():
    def __init__(self, args):

        self.args = args
        self.model_classification = PVT_Classification.pvt_v2_b2_li



