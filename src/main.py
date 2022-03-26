import argparse
import spacy
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Embedding,StringLookup
from tensorflow.keras.initializers import Constant
from tensorflow.keras.losses import SparseCategoricalCrossentropy, \
                                     BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, \
                                        ModelCheckpoint, TensorBoard

from utils import get_data_loader, load_config
from utils import get_model, get_corpus, get_optimizer
from utils import set_device, set_seeds


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
    
    train_ds = get_data_loader(cfg,split=0)
    #valid_ds = get_data_loader(cfg,split=1)

    model = get_model(cfg,lookup_layer,embedding_layer)
    optimizer = get_optimizer(cfg)
    loss = BinaryCrossentropy(from_logits=True)


    #early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2)
    #lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
    #checkpointer = ModelCheckpoint(filepath='../checkpoints/'+dataset+'/'+experiment_name, monitor='val_loss', verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir='../logs/'+cfg['dataset']+'/'+cfg['experiment_name'], write_images=True)

    model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')

    model.fit(x = train_ds, epochs=cfg['n_epochs'], verbose=2, shuffle=True,
              callbacks=[tensorboard], validation_split=0.0)

    #model.load_weights('../checkpoints/'+dataset+'/'+experiment_name)  # load the best model
    model.evaluate(x=train_ds) # needs to be fixed

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to a config file.")

    _args = parser.parse_args()

    cfg = load_config(_args)

    set_seeds(cfg)
    set_device(cfg)

    lookup_layer,embedding_layer,info = prepare_embeddings(cfg)
    print(f'hits: {info[0]}, misses: {info[1]}')

    train(cfg)



