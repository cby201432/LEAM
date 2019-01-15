## Introduction
This is the implementation of Guoyin Wang's [Joint Embedding of Words and Labels for Text Classification (ACL 2018)] paper in PyTorch.

Code refers Guoyin Wang(https://github.com/guoyinwang/LEAM)

## Dependencies
* python 3.6
* pytorch 0.4.0
* jieba 0.39

Additional dependencies for running experiments are: numpy, pickle, gensim.

## Result
I tried online shopping review dataset.

The best test accuracy is 89%

## Prepare datasets

Data are prepared in pickle format. Each `data.p` file has the same fields in same order: train text, val text, test text, train label, val label, test label and vocab.p contains: dictionary and reverse dictionary.

Datasets can be downloaded here (https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md). Put the download data in data1 directory. Each dataset has three files: tokenized data, vocab data and corresponding pretrained fastText embedding.

Follow the code in preprocess.py to tokenize and split train/dev/test dataset. To build pretrained word embeddings, first download [fastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) Chinese pre-trained word embeddings and then follow generate_emb.py.

## Training

```
python main.py -h
```

```
optional arguments:
  -h, --help            show this help message and exit
  --filename FILENAME
  --n_epochs N_EPOCHS   number of epochs for train [default: 200]
  --valid_freq VALID_FREQ
  --hidden_size HIDDEN_SIZE
  --lr LR               initial learning rate [default: 0.001]
  --batch_size BATCH_SIZE
                        batch size [default: 100
  --max_len MAX_LEN     PAD sentence to same length [default: 55]
  --embed_size EMBED_SIZE
                        word embedding dim [default: 300]
  --dropout DROPOUT     the probability for dropout [default: 0.5]
  --optimizer OPTIMIZER
  --class_penalty CLASS_PENALTY
  --ngram NGRAM
  --save_path SAVE_PATH
                        path to save trained model
  --restore RESTORE     restore trained model
  --predict PREDICT     predict the sentence given [option: 'None', 'pos',
                        'key_words']
  --part_data PART_DATA
  --portion PORTION


```

## Predict

```
bash predict_key_words < input.txt > output.txt
```

```
bash predict_pos < input.txt > output.txt
```