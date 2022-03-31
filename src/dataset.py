import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_data_loader_(cfg,lookup_l,split):

    data_dir = cfg['data_dir']
    dataset = cfg['dataset']
    
    df = pd.read_pickle(os.path.join(data_dir,dataset,'cleaned.pickle'))

    x = tf.ragged.constant(df[df['split']==split]['cleaned'])
    idx = lookup_l(x)
    padded_x = tf.convert_to_tensor(pad_sequences(sequences=list(idx),maxlen=cfg["max_sentence_length"]))

    y = df[df['split']==split]['label']

    ds = tf.data.Dataset.from_tensor_slices((padded_x,y))
    ds = ds.shuffle(1000).batch(cfg['batch_size'])
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds