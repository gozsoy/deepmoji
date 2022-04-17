# deepmoji
Tensorflow implementation of the paper [Using millions of emoji occurrences to learn any-domain representations
for detecting sentiment, emotion and sarcasm](https://arxiv.org/pdf/1708.00524.pdf).

### Usage
```
cd src
python main.py --config ../config_{bert|deepmoji}.yaml
```

### Explanation

Deepmoji model is a pre-trained base which is able to generalize well on text classification tasks. The model is as on fig xx. The authors use emoji classification for model's pre-training and fine-tune it more on downstream tasks' small scale datasets. It is a nice example of representation learning for text which is nowadays best performed by transformer based language models such as BERT.

As the 1.2 billion tweet emoji dataset is not publicly available, and I see this project as a proof-of-concept, I used [amazon-polarity](https://huggingface.co/datasets/amazon_polarity)'s test set as pre-training dataset. Since it has positive-negative sentiment classification objective, it is similar to chosen final task, [insult-detection](https://github.com/bfelbo/DeepMoji/tree/master/data).

Furthermore, for comparing model performance with state-of-the-art language models, I also used distilBERT.

### Evaluation

Learning rate decay and early stopping are both set on validation loss. Plain tokenization, gensim simple_preprocess() and full preprocessing (stop word, punctuation, numerics removal, lemmatization) are implemented and only the best among these per model is reported below. Spacy embeddings are used for initialization in deepmoji model.

| Model      | Test F1 score | Best Validation F1 score     |
| :---        |    :----:   |          :---: |
| Deepmoji base | 0.76       | 0.75  |
| Deepmoji fine-tuning (freeze base) | 0.71        | 0.74      |
| Deepmoji fine-tuning (train base) | 0.74        | 0.76      |
| DistilBERT fine-tuning (freeze base) | 0.70        | 0.72      |
| DistilBERT fine-tuning (train base) | 0.76        | 0.76      |


Deepmoji base: trained directly on insult-detection
Deepmoji fine-tuning: pretrained on amazon-polarity, fine-tuned on insult-detection
Distilbert fine-tuning: fine-tuned on insult-detection

### Insight

In most of the experiments, plain tokenization performed best, while full preprocessing being the worst. It is sensible as the model also cares functional words for understanding. \
Validation set for insult-detection has only 495 samples and I observed label-sentence mismatch for some samples. This makes the validation loss not informative, and training unstable. \
Freezing pre-trained base model degrades the performance on down-stream task as only the classification head is updated. \
DistilBERT fine-tuning leads to best performance. Compared to it, Deepmoji model is low capacity because of 10x less parameter size and lack of attention mechanism, and its pre-training is very limited. Thus I admits its lower performance.

