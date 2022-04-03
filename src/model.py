import os
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional,LSTM,Embedding,StringLookup,Dense
from tensorflow.keras.initializers import Constant
from tensorflow.keras import Model
from tensorflow.keras.activations import softmax
from tensorflow.math import reduce_sum
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import DistilBertTokenizerFast,TFDistilBertModel

# base model for pretraining
class DeepmojiNet(Model):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.lstm1 = Bidirectional(LSTM(units=cfg['lsmt_hidden_size'],return_sequences=True))
        self.lstm2 = Bidirectional(LSTM(units=cfg['lsmt_hidden_size'],return_sequences=True))
        self.dense1 = Dense(units=1, activation='relu')
        self.dense2 = Dense(units=128, activation='relu')
        self.dense3 = Dense(units=cfg['out_dim'], activation=None)

    def call(self,embs):

        hiddens1 = self.lstm1(embs)
        hiddens2 = self.lstm2(hiddens1)

        word_repr = tf.concat([embs,hiddens1,hiddens2],axis=-1)
        o = self.dense1(word_repr)
        att_weights = softmax(o,axis=1)
        sent_repr = reduce_sum(word_repr * att_weights,axis=1)

        output = self.dense2(sent_repr)
        output = self.dense3(output)
        
        return output


# classifier to be fine-tuned on base DeepmojiNet
class DeepmojiBasedClassifier(Model):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg

        self.base = DeepmojiNet(cfg)
        self.base.trainable = False
        self.base(tf.random.normal(shape=(1,cfg['max_sentence_length'],cfg['embedding_size']))) #initialize base model weights
        main_dir,_ = os.path.split(cfg['data_dir'])
        self.base.load_weights(os.path.join(main_dir,'checkpoints','amazon-polarity',cfg['pretraining_experiment_name']))
 
        self.dense2 = Dense(units=128, activation='relu')
        self.dense2_2 = Dense(units=128, activation='relu')
        self.dense3 = Dense(units=cfg['out_dim'], activation=None)


    def call(self,embs):
        # not the best practice! one should output intermediate tensors from base model instead.

        hiddens1 = self.base.layers[0](embs)
        hiddens2 = self.base.layers[1](hiddens1)

        word_repr = tf.concat([embs,hiddens1,hiddens2],axis=-1)
        o = self.base.layers[2](word_repr)
        att_weights = softmax(o,axis=1)
        sent_repr = reduce_sum(word_repr * att_weights,axis=1)

        output = self.dense2(sent_repr)
        output = self.dense2_2(output)
        output = self.dense3(output)
        
        return output


# classifier to be fine-tuned on base distilbert
class BertBasedClassifier(Model):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        #self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.distilbert_base = TFDistilBertModel.from_pretrained("distilbert-base-uncased")

        for layer in self.distilbert_base.layers:
            layer.trainable = False

        self.dense1 = Dense(units=128, activation='relu')
        self.dense2 = Dense(units=1, activation=None)

    def call(self,x):
        #print(x)
        #encoded_input = self.tokenizer(text=x,max_length=self.cfg['max_sentence_length'],padding=True,truncation=True, return_tensors='tf')
        output = self.distilbert_base(input_ids=x[0],attention_mask=x[1])

        last_hidden_state = output[0]
        cls = last_hidden_state[:,0,:]

        output = self.dense1(cls)
        output = self.dense2(output)
        
        return output