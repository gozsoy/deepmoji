import argparse
import spacy
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Embedding,StringLookup
from tensorflow.keras.initializers import Constant

from utils import load_config,get_corpus, set_device, set_seeds


def prepare_embeddings(cfg):

    corpus = get_corpus(cfg)

    # {int idx: word}
    lookup_layer = StringLookup()
    lookup_layer.adapt(corpus)

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

    embedding_layer = Embedding(input_dim=len(voc),output_dim=emb_dim,embeddings_initializer=Constant(tf.constant(embedding_matrix,dtype=tf.float32)),trainable=cfg['train_embeddings'])

    return lookup_layer,embedding_layer,(hits,misses,missed_words)

def train(cfg):
    print(info)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a config file.")

    _args = parser.parse_args()

    cfg = load_config(_args)

    set_seeds(cfg)
    set_device(cfg)

    lookup_layer,embedding_layer,info = prepare_embeddings(cfg)

    train(cfg)



