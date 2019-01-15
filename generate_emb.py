import gensim
import pickle
import numpy as np


def load_embedding_vectors(vocab, filename):
    print("loading embedding")
    model = gensim.models.KeyedVectors.load_word2vec_format(filename)
    vector_size = model.vector_size
    embedding_vectors = np.random.uniform(-0.01, 0.01, (len(vocab), vector_size))
    fast_vocab = list(model.vocab.keys())
    count = 0
    mis_count = 0
    for word in vocab.keys():
        idx = vocab.get(word)
        if word in fast_vocab:
            embedding_vectors[idx] = model.wv[word]
            count += 1
        else:
            mis_count += 1
    print("num of vocab in : {}".format(count))
    print("num of vocab not in fasttext: {}".format(mis_count))
    return embedding_vectors

if __name__ == '__main__':

    loadpath = "./data1/vocab.p"
    embpath = "./data1/data_fast.p"
    x = pickle.load(open(loadpath, "rb"), encoding='latin1')
    word2idx, idx2word = x[0], x[1]

    print("load data finished")

    y = load_embedding_vectors(word2idx, './data1/wiki.zh.vec' )
    pickle.dump([y.astype(np.float32)], open(embpath, 'wb'))
