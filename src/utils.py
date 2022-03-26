from parso import split_lines
from torch import ge
from transformers import KerasMetricCallback
import yaml
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import random

from model import DeepmojiNet
from dataset import get_data_loader_

def load_config(args):

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def set_seeds(cfg):
    seed = cfg['seed']
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    return

def set_device(cfg):
    if cfg['device'] == 'cpu':
        tf.config.set_visible_devices([], 'GPU')
    
    return

def get_data_loader(cfg, split):

    return get_data_loader_(cfg,split)


def get_corpus(cfg):
    data_dir = cfg['data_dir']
    dataset = cfg['dataset']
    
    df = pd.read_pickle(os.path.join(data_dir,dataset,'cleaned.pickle'))
    cleaned_texts = tf.ragged.constant(df.cleaned)
    
    return cleaned_texts

def get_model(cfg,lookup_l,embedding_l):
    if cfg["model"] == "DeepmojiNet":
        # if get_model only takes cfg, then define set() in DeepmojiNet and set lookup and embedding in train() 
        model = DeepmojiNet(cfg=cfg,lookup_layer=lookup_l,embedding_layer=embedding_l)
    else:
        raise NotImplementedError('Model not implemented')

    return model


def get_optimizer(cfg):

    if cfg["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(lr=cfg["lr"])
    else:
        raise NotImplementedError('optimizer not implemented')

    return optimizer

