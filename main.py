import argparse
import pickle
import torch
import numpy as np
from utils import load_class_embedding
from train import training, predict

# parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--filename', type=str, default='./data1/data.p')
argparser.add_argument('--n_epochs', type=int, default=15, help='number of epochs for train [default: 200]')
argparser.add_argument('--valid_freq', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=300)
argparser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
argparser.add_argument('--batch_size', type=int, default=100, help='batch size [default: 100')
argparser.add_argument('--max_len', type=int, default=55, help='PAD sentence to same length [default: 55]')
argparser.add_argument('--embed_size', type=int, default=300, help='word embedding dim [default: 300]')
argparser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
argparser.add_argument('--optimizer', type=str, default='Adam')
argparser.add_argument('--class_penalty', type=float, default=1.0)
argparser.add_argument('--ngram', type=int, default=11)
argparser.add_argument('--save_path', type=str, default='./save/train.pt', help='path to save trained model')
argparser.add_argument('--restore', type=bool, default=False, help='restore trained model')
argparser.add_argument('--predict', type=str, default='None', help="predict the sentence given [option: 'None', 'pos', 'key_words']")
argparser.add_argument('--part_data', type=bool, default=False)
argparser.add_argument('--portion', type=float, default=0.2)

args = argparser.parse_args()

vocabpath = './data1/vocab.p'
word2idx, idx2word = pickle.load(open(vocabpath, 'rb'), encoding='latin1')
args.word2idx = word2idx
args.idx2word = idx2word

args.vocab_size = len(idx2word)
print('Total words: %d' % args.vocab_size)
print('batch size: %d' % args.batch_size)
print('learning rate: %.4f' % args.lr)

class_name = ['好',
              '不好']
args.num_class = len(class_name)
embpath = './data1/data_fast.p'
try:
    args.W_emb = np.array(pickle.load(open(embpath, 'rb'), encoding='latin1')[0], dtype='float32')
    args.W_class_emb = load_class_embedding(word2idx, args.W_emb, class_name)
    args.W_emb = torch.FloatTensor(args.W_emb)
    args.W_class_emb = torch.FloatTensor(args.W_class_emb)
except IOError:
    print('No embedding file found.')

if args.predict == 'None':
    training(args)
else:
    predict(args)

