# coding: utf-8
import re
import jieba
import numpy as np
import pickle


def load_data(pos_data_file, neg_data_file):
    """
    Loads raw data from file
    split data into words using jieba
    label all the data
    """
    x = []
    label = []
    vocab = {}
    for line in open(pos_data_file, 'r', encoding='utf-8'):
        temp = [item for item in jieba.cut(clean_str(line), cut_all=False)]
        if temp:
            x.append(temp)
            label.append([1, 0])
            for w in set(temp):
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1

    for line in open(neg_data_file, 'r', encoding='utf-8'):
        temp = [item for item in jieba.cut(clean_str(line), cut_all=False)]
        if temp:
            x.append(temp)
            label.append([0, 1])
            for w in set(temp):
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1

    vocab = [x for x, y in vocab.items() if y >= 5]
    word2idx = {}
    word2idx['END'] = 0
    word2idx['UNK'] = 1
    idx2word = {}
    idx2word[0] = 'END'
    idx2word[1] = 'UNK'
    idx = 2
    for w in vocab:
        word2idx[w] = idx
        idx2word[idx] = w
        idx += 1
    print("vocab size is %d" % len(word2idx))
    x = convert_word2idx(x, word2idx)
    x = split2train_val_test(x, label)
    return(x, [word2idx, idx2word])


def split2train_val_test(data, label):
    """
    split the whole data set to training set(80%) and validation(10%) and test set(10%)
    """
    assert len(data) == len(label)
    data = np.array(data)
    label = np.array(label, dtype='float32')
    data_size = len(data)
    test_size = val_size = data_size // 10
    shuffle_indice = np.random.permutation(np.arange(data_size))
    data = data[shuffle_indice]
    label = label[shuffle_indice]
    test_x = data[:test_size]
    test_label = label[:test_size]
    val_x = data[test_size:(test_size + val_size)]
    val_label = label[test_size:(test_size + val_size)]
    train_x = data[(test_size + val_size):]
    train_label = label[(test_size + val_size):]
    print('train data size: ', len(train_x))
    print('val data size: ', len(val_x))
    print('test data size: ', len(test_x))
    return([train_x, val_x, test_x, train_label, val_label, test_label])


def convert_word2idx(data, vocab):
    res = []
    for sent in data:
        temp = []
        for w in sent:
            if w in vocab:
                temp.append(vocab[w])
            else:
                temp.append(1)
        temp.append(0)
        res.append(temp)
    return(res)


def clean_str(string):
    """
    clean every review
    """
    string = re.sub(r"[^\u4e00-\u9fff]", "", string)
    string = re.sub(r"没有描述", "", string)
    string = re.sub(r"暂时还没有发现缺点哦", "", string)
    # string = re.sub(r"(暂时|暂)(还)?(没|没有)(发现)?(什么)?(缺点|不足)?(哦|呢|呀|吧)?", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return(string.strip())


def main():
    pos = './data1/pos.txt'
    neg = './data1/neg.txt'
    data, vocab = load_data(pos, neg)
    pickle.dump(data, open('./data1/data.p', 'wb'))
    pickle.dump(vocab, open('./data1/vocab.p', 'wb'))

if __name__ == "__main__":
    main()
