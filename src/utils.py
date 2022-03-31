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

def get_data_loader(cfg, lookup_l, split):

    return get_data_loader_(cfg,lookup_l, split)


def get_corpus(cfg):
    data_dir = cfg['data_dir']
    dataset = cfg['dataset']
    
    df = pd.read_pickle(os.path.join(data_dir,dataset,'cleaned.pickle'))
    cleaned_texts = tf.ragged.constant(df.cleaned)
    
    return cleaned_texts

def get_model(cfg,embedding_l):
    if cfg["model"] == "DeepmojiNet":
        # if get_model only takes cfg, then define set() in DeepmojiNet and set lookup and embedding in train() 
        model = DeepmojiNet(cfg=cfg,embedding_layer=embedding_l)
    else:
        raise NotImplementedError('Model not implemented')

    return model


def get_optimizer(cfg):

    if cfg["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"])
    elif cfg["optimizer"] == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=cfg["learning_rate"])
    else:
        raise NotImplementedError('optimizer not implemented')

    return optimizer

