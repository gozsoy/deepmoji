import argparse
import spacy
import pandas as pd
import numpy as np
import os
import pickle

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
from utils import F1Score

def prepare_embeddings(cfg):

    data_dir = cfg['data_dir']
    dataset = cfg['dataset']

    # create lookup layer
    if cfg['load_vocabulary']:
        with open(os.path.join(data_dir,dataset,'processed_voc'), "rb") as fp:
            processed_voc = pickle.load(fp)
        lookup_layer = StringLookup(vocabulary=processed_voc,mask_token='[MASK]')
        print(f'Created string lookup layer from processed vocabulary')
    else:
        corpus = get_corpus(cfg)
        lookup_layer = StringLookup(mask_token='[MASK]')
        lookup_layer.adapt(corpus)
        print(f'Created string lookup layer from scratch using corpus')
    
    voc = lookup_layer.get_vocabulary()
    print(f'vocabulary size: {len(voc)}')

    # create embedding matrix
    if cfg['load_embeddings']:
        embedding_matrix = np.load(os.path.join(data_dir,dataset,'embedding_matrix.npy'))  
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

def train(cfg,lookup_layer,embedding_layer):
    
    train_ds = get_data_loader(cfg,lookup_layer,embedding_layer,split=0)
    print(f'train_ds ready')
    valid_ds = get_data_loader(cfg,lookup_layer,embedding_layer,split=1)
    print(f'valid_ds ready')
    test_ds = get_data_loader(cfg,lookup_layer,embedding_layer,split=2)
    print(f'test_ds ready')
    

    model = get_model(cfg)
    optimizer = get_optimizer(cfg)
    loss = BinaryCrossentropy(from_logits=True)

    main_dir,_ = os.path.split(cfg['data_dir'])
    checkpointer = ModelCheckpoint(filepath=os.path.join(main_dir,'checkpoints',cfg['dataset'],cfg['experiment_name']), monitor='val_loss', verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=os.path.join(main_dir,'logs',cfg['dataset'],cfg['experiment_name']), write_images=True)
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=2)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)
    

    f1_score = F1Score(from_logits=True,name='f1_score')
    metrics = ['accuracy',f1_score]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.fit(x = train_ds, epochs=cfg['n_epochs'], verbose=1, shuffle=True,
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

    lookup_layer,embedding_layer = prepare_embeddings(cfg)

    train(cfg,lookup_layer,embedding_layer)



