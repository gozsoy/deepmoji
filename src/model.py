import tensorflow as tf
from tensorflow.keras.layers import Bidirectional,LSTM,Embedding,StringLookup,Dense
from tensorflow.keras.initializers import Constant
from tensorflow.keras import Model
from tensorflow.keras.activations import softmax
from tensorflow.math import reduce_sum
from tensorflow.keras.preprocessing.sequence import pad_sequences


class DeepmojiNet(Model):
    def __init__(self,cfg,embedding_layer):
        super().__init__()
        self.cfg = cfg
        self.embedding_layer = embedding_layer
        self.lstm1 = Bidirectional(LSTM(units=cfg['lsmt_hidden_size'],return_sequences=True))
        self.lstm2 = Bidirectional(LSTM(units=cfg['lsmt_hidden_size'],return_sequences=True))
        self.dense1 = Dense(units=1, activation='relu')
        self.dense2 = Dense(units=128, activation='relu')
        self.dense3 = Dense(units=cfg['out_dim'], activation=None)

    def call(self,x):
        embs = self.embedding_layer(x)
        # put F.tanh here?

        hiddens1 = self.lstm1(embs)
        hiddens2 = self.lstm2(hiddens1)

        word_repr = tf.concat([embs,hiddens1,hiddens2],axis=-1)
        o = self.dense1(word_repr)
        att_weights = softmax(o,axis=1)
        sent_repr = reduce_sum(word_repr * att_weights,axis=1)

        output = self.dense2(sent_repr)
        output = self.dense3(output)
        
        return output