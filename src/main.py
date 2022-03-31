import argparse
import spacy
import pandas as pd
import numpy as np
import os

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
from utils import F1Score,Precision,Recall

def prepare_embeddings(cfg):

    corpus = get_corpus(cfg)

    # {int idx: word}
    lookup_layer = StringLookup(mask_token='[MASK]')
    #lookup_layer = StringLookup()
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

    embedding_layer = Embedding(input_dim=len(voc),output_dim=emb_dim,embeddings_initializer=Constant(tf.constant(embedding_matrix,dtype=tf.float32)),mask_zero=True,trainable=cfg['train_embeddings'])

    return lookup_layer,embedding_layer,(hits,misses,missed_words)

def train(cfg,lookup_layer,embedding_layer):
    
    train_ds = get_data_loader(cfg,lookup_layer,split=0)
    valid_ds = get_data_loader(cfg,lookup_layer,split=1)
    test_ds = get_data_loader(cfg,lookup_layer,split=2)

    model = get_model(cfg,embedding_layer)
    optimizer = get_optimizer(cfg)
    loss = BinaryCrossentropy(from_logits=True)

    main_dir,_ = os.path.split(cfg['data_dir'])
    checkpointer = ModelCheckpoint(filepath=os.path.join(main_dir,'checkpoints',cfg['dataset'],cfg['experiment_name']), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    tensorboard = TensorBoard(log_dir=os.path.join(main_dir,'logs',cfg['dataset'],cfg['experiment_name']), write_images=True)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
    
    precision = Precision(from_logits=True,name='precision')
    recall = Recall(from_logits=True,name='recall')
    f1_score = F1Score(from_logits=True,name='f1_score')
    metrics = [precision,recall,f1_score]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.fit(x = train_ds, epochs=cfg['n_epochs'], verbose=2, shuffle=True,
              callbacks=[early_stopper,lr_scheduler,checkpointer,tensorboard], validation_data=valid_ds)

    model.load_weights(os.path.join(main_dir,'checkpoints',cfg['dataset'],cfg['experiment_name']))  # load the best model
    model.evaluate(x=test_ds)

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

    train(cfg,lookup_layer,embedding_layer)



