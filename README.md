# deepmoji
Tensorflow implementation of the paper [Using millions of emoji occurrences to learn any-domain representations
for detecting sentiment, emotion and sarcasm](https://arxiv.org/pdf/1708.00524.pdf).

give dataset links
explain your findings: one problem is validation set is only 500 samples which is not very informative. + samples are noisy and some are meaningless.
put deepmoji graph here
i selected amazon-polarity to reflect sentiment like nature of its data
also report best validation errors of each!

transfer learning is regualarization as we can see from validation loss convergence
why spacy processed is the lowest one ?
freeze true vs false

### dataset: kaggle-insults
Deepmoji base (trained directly on kaggle-insults)

test f1 score, best val f1 score
deepmoji_base_lstm512_cls128_1_only_tokenized : 0.75, 0.76
deepmoji_base_lstm512_cls128_1_gensim_processed : 0.76, 0.75
deepmoji_base_lstm512_cls128_1_spacy_processed : 0.72, 0.75

Deepmoji fine-tuning (pretrained on amazon-polarity, fine-tuned on kaggle-insults)

test f1 score, best val f1 score
deepmoji_transfer_freezeTrue_lstm512_cls128_128_1_only_tokenized : 0.71, 0.74
deepmoji_transfer_freezeTrue_lstm512_cls128_128_1_gensim_processed : 0.70, 0.74
deepmoji_transfer_freezeTrue_lstm512_cls128_128_1_spacy_processed: 0.68, 0.73

test f1 score, best val f1 score
deepmoji_transfer_freezeFalse_lstm512_cls128_128_1_only_tokenized: 0.74, 0.76
deepmoji_transfer_freezeFalse_lstm512_cls128_128_1_gensim_processed: 0.72, 0.75
deepmoji_transfer_freezeFalse_lstm512_cls128_128_1_spacy_processed: 0.73, 0.72

Distilbert fine-tuning (fine tuned on kaggle-insults)

test f1 score, best val f1 score
bert_freezeTrue_cls128_1_only_tokenized : 0.70, 0.72
bert_freezeTrue_cls128_1_gensim_processed: 0.69, 0.71
bert_freezeTrue_cls128_1_spacy_processed: 0.63, 0.68

test f1 score, best val f1 score
bert_freezeFalse_cls128_1_only_tokenized : 0.75, 0.79
bert_freezeFalse_cls128_1_gensim_processed : 0.76, 0.76
bert_freezeFalse_cls128_1_spacy_processed : 0.72, 0.74

