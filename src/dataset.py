import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import DistilBertTokenizerFast

def get_data_loader_(cfg,lookup_l,embedding_l,split):

    data_dir = cfg['data_dir']
    dataset = cfg['dataset']
    preprocess_type = cfg['preprocess_type']
    
    df = pd.read_pickle(os.path.join(data_dir,dataset,'cleaned.pickle'))

    y = df[df['split']==split]['label']

    
    if cfg['model'] == 'BertBasedClassifier':
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        x = list(df[df['split']==split][preprocess_type].apply(lambda row: " ".join(row)))

        inputs = tokenizer(text=x,max_length=cfg['max_sentence_length'],padding=True,truncation=True, return_tensors='tf')

        ds = tf.data.Dataset.from_tensor_slices(((inputs['input_ids'],inputs['attention_mask']),y))
        ds = ds.shuffle(1000).batch(cfg['batch_size'])

    else: # deepmoji expects each word's embedding vector as input
        x = tf.ragged.constant(df[df['split']==split][preprocess_type])
        idx = lookup_l(x)
        padded_x = tf.convert_to_tensor(pad_sequences(sequences=list(idx),maxlen=cfg["max_sentence_length"]))

        ds = tf.data.Dataset.from_tensor_slices((padded_x,y))
        ds = ds.shuffle(1000).batch(cfg['batch_size']).map(lambda x,y:(embedding_l(x),y))


    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds