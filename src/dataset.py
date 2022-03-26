import tensorflow as tf
import pandas as pd
import os

def get_data_loader_(cfg,split):

    data_dir = cfg['data_dir']
    dataset = cfg['dataset']
    
    df = pd.read_pickle(os.path.join(data_dir,dataset,'cleaned.pickle'))

    x = tf.ragged.constant(df[df['split']==split]['cleaned'])
    y = df[df['split']==split]['info']

    ds = tf.data.Dataset.from_tensor_slices((x,y))
    ds = ds.shuffle(1000).batch(cfg['batch_size'])
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds