import spacy
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import StringLookup
from gensim.utils import simple_preprocess



def preprocess_kaggle_insult():
    # preprocessing kaggle-insults

    nlp = spacy.load('en_core_web_md')

    data = pd.read_pickle('../data/kaggle-insults/raw.pickle')
    df = pd.DataFrame.from_dict({'info':data['info'],'texts':data['texts']})

    df['split'] = -1
    df.loc[data['train_ind'],'split'] = 0
    df.loc[data['val_ind'],'split'] = 1
    df.loc[data['test_ind'],'split'] = 2
    df['info'] = df['info'].apply(lambda row: row['label'])
    df.rename(columns={'info':'label'},inplace=True)

    # only tokenization
    def custom_preprocess_1(row):
        return [t.text.lower() for t in nlp(row)]

    # spacy full tokenization -> tokenize and stop word, punct removal, lowercase, lemmatize
    def custom_preprocess_2(row):
        return [t.lemma_.lower() for t in nlp(row) if not t.is_stop and not t.like_num and not t.is_punct and not t.is_space]

    # gensim simple_preprocess -> tokenize and eleminate token with len <2 or >15
    def custom_preprocess_3(row):
        return simple_preprocess(row)


    df['only_tokenized'] = df['texts'].apply(lambda row: custom_preprocess_1(row))
    df['spacy_processed'] = df['texts'].apply(lambda row: custom_preprocess_2(row))
    df['gensim_processed'] = df['texts'].apply(lambda row: custom_preprocess_3(row))
    df.to_pickle('../data/kaggle-insults/cleaned.pickle',protocol=4)

    return


def preprocess_amazon_polarity():

    # preprocess amazon-polarity

    amazon_df = pd.read_csv('../data/amazon-polarity/test.csv',header=None)

    amazon_df.rename(columns={0:'label',2:'texts'},inplace=True)
    amazon_df['label'] = amazon_df['label']-1
    amazon_df['split'] = 0
    amazon_df.loc[amazon_df[amazon_df['label']==1].sample(frac=.2).index,'split'] = 1
    amazon_df.loc[amazon_df[amazon_df['label']==0].sample(frac=.2).index,'split'] = 1
    amazon_df['gensim_processed'] = amazon_df['texts'].apply(lambda row: simple_preprocess(row))
    amazon_df = amazon_df[['label','gensim_processed','split']]
    amazon_df.to_pickle('../data/amazon-polarity/cleaned.pickle',protocol=4)


    # prepare and save amazon-polarity vocabulary and embedding matrix beforehand
    # it is expensive to this everytime

    cleaned_texts = tf.ragged.constant(amazon_df.cleaned)

    # {int idx: word}
    lookup_layer_base = StringLookup(mask_token='[MASK]')
    lookup_layer_base.adapt(cleaned_texts)

    nlp = spacy.load('en_core_web_md')

    processed_voc = list(set(lookup_layer_base.get_vocabulary()) & set(list(nlp.vocab.strings)))

    lookup_layer = StringLookup(vocabulary=processed_voc,mask_token='[MASK]')

    # {word : word vectors}
    embeddings_index = {}

    for key,vector in list(nlp.vocab.vectors.items()):
        
        embeddings_index[nlp.vocab.strings[key]] = vector


    emb_dim = 300
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


    with open("../data/amazon-polarity/processed_voc", "wb") as fp:
        pickle.dump(processed_voc, fp,protocol=4)

    np.save("../data/amazon-polarity/embedding_matrix.npy",embedding_matrix)

    return