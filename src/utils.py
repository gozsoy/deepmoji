import yaml
import pickle
import spacy
import os
import pandas as pd
import tensorflow as tf
import tensorflow.keras.metrics as tfm
from tensorflow.keras.layers import Embedding,StringLookup
from tensorflow.keras.initializers import Constant
import numpy as np
import random

from model import DeepmojiNet,DeepmojiBasedClassifier,BertBasedClassifier
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

def get_data_loader(cfg, lookup_l,embedding_l, split):

    return get_data_loader_(cfg,lookup_l,embedding_l, split)


def get_corpus(cfg):
    data_dir = cfg['data_dir']
    dataset = cfg['dataset']
    preprocess_type = cfg['preprocess_type']
    
    df = pd.read_pickle(os.path.join(data_dir,dataset,'cleaned.pickle'))
    cleaned_texts = tf.ragged.constant(df[preprocess_type])

    return cleaned_texts

def get_model(cfg):
    if cfg["model"] == "DeepmojiNet":
        model = DeepmojiNet(cfg=cfg)
    elif cfg["model"] == "DeepmojiBasedClassifier":
        model = DeepmojiBasedClassifier(cfg=cfg)
    elif cfg["model"] == "BertBasedClassifier":
        model = BertBasedClassifier(cfg=cfg)
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


def prepare_embeddings(cfg):

    data_dir = cfg['data_dir']
    dataset = cfg['dataset']
    preprocess_type = cfg['preprocess_type']

    # create lookup layer
    if cfg['load_vocabulary']:
        with open(os.path.join(data_dir,dataset,'processed_voc_'+preprocess_type), "rb") as fp:
            processed_voc = pickle.load(fp)
        lookup_layer = StringLookup(vocabulary=processed_voc,mask_token='[MASK]')
        print(f'Loaded string lookup layer')
    else:
        corpus = get_corpus(cfg)
        lookup_layer = StringLookup(mask_token='[MASK]')
        lookup_layer.adapt(corpus)
        print(f'Created string lookup layer from scratch')
    
    voc = lookup_layer.get_vocabulary()
    print(f'vocabulary size: {len(voc)}')

    # create embedding matrix
    if cfg['load_embeddings']:
        embedding_matrix = np.load(os.path.join(data_dir,dataset,'embedding_matrix_'+preprocess_type+'.npy'))  
        print(f'Loaded embedding matrix')
    else:
        if cfg['embedding_type']== 'spacy':
            nlp = spacy.load('en_core_web_md')
        else:
            raise NotImplementedError('unknown embedding type')
        # {word : word vectors}
        embeddings_index = {}

        for key,vector in list(nlp.vocab.vectors.items()):
            
            embeddings_index[nlp.vocab.strings[key]] = vector


        emb_dim = cfg['embedding_size']
        voc = lookup_layer.get_vocabulary()

        embedding_matrix = np.zeros((len(voc),emb_dim))
        
        hits = 0
        misses = 0
        missed_words = []
        for i,word in enumerate(voc):
            temp_vec = embeddings_index.get(word)
            if temp_vec is not None:
                embedding_matrix[i] = temp_vec
                hits += 1
            else:
                misses += 1
                missed_words.append(word)
        print(f'hits: {hits}, misses: {misses}')
        print(f'Created embedding matrix from scratch')

    input_dim, output_dim = embedding_matrix.shape
    embedding_layer = Embedding(input_dim=input_dim,output_dim=output_dim,embeddings_initializer=Constant(tf.constant(embedding_matrix,dtype=tf.float32)),mask_zero=True,trainable=cfg['train_embeddings'])

    return lookup_layer,embedding_layer




# below are borrowed from: https://github.com/tensorflow/tensorflow/issues/42182
# the reason is most of the tf metrics does not accept from_logit argument
class AUC(tfm.AUC):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(AUC, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(AUC, self).update_state(y_true, y_pred, sample_weight)


class BinaryAccuracy(tfm.BinaryAccuracy):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(BinaryAccuracy, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(BinaryAccuracy, self).update_state(y_true, y_pred, sample_weight)


class TruePositives(tfm.TruePositives):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(TruePositives, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(TruePositives, self).update_state(y_true, y_pred, sample_weight)


class FalsePositives(tfm.FalsePositives):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(FalsePositives, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(FalsePositives, self).update_state(y_true, y_pred, sample_weight)


class TrueNegatives(tfm.TrueNegatives):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(TrueNegatives, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(TrueNegatives, self).update_state(y_true, y_pred, sample_weight)


class FalseNegatives(tfm.FalseNegatives):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(FalseNegatives, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(FalseNegatives, self).update_state(y_true, y_pred, sample_weight)


class Precision(tfm.Precision):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(Precision, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(Precision, self).update_state(y_true, y_pred, sample_weight)


class Recall(tfm.Recall):
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self._from_logits:
            super(Recall, self).update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight)
        else:
            super(Recall, self).update_state(y_true, y_pred, sample_weight)

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1", from_logits=False, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision(from_logits)
        self.recall = Recall(from_logits)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return (2 * p * r) / (p + r + tf.keras.backend.epsilon())

    def reset_state(self):
        self.precision.reset_states()
        self.recall.reset_states()