import numpy as np
import matplotlib.pyplot as plt
import torch


def load_class_embedding(wordtoidx, W_emb, class_name):
    print("load class embedding")
    name_list = [k.lower().split(' ') for k in class_name]
    id_list = [[wordtoidx[i] for i in l] for l in name_list]
    value_list = [[W_emb[i] for i in l] for l in id_list]
    value_mean = [np.mean(l, 0) for l in value_list]
    return np.asarray(value_mean)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size
    return zip(range(len(minibatches)), minibatches)


def prepare_data_for_emb(seqs_x, args):
    """
    make the sentence same length
    """
    maxlen = args.max_len
    lengths_x = [len(s) for s in seqs_x]
    # maxlen = np.max(lengths_x)
    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
            else:
                new_seqs_x.append(s_x[:maxlen])
                new_lengths_x.append(maxlen)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None

    n_samples = len(seqs_x)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
        x_mask[idx, :lengths_x[idx]] = 1. # change to remove the real END token
    x = torch.LongTensor(x)
    x_mask = torch.FloatTensor(x_mask)
    return x, x_mask


def visualize(args, loss, train_accuracy, val_accuracy):
    plt.figure(1)
    plt.title("Train Loss vs Iteration")
    plt.xlabel("Iteration number(batch: %d, learning rate: %f)" % (args.batch_size, args.lr))
    plt.ylabel("Loss")

    n = len(loss)
    iters = list(range(args.valid_freq, args.valid_freq * (n + 1), args.valid_freq))
    plt.plot(iters, loss)
    fname = 'Loss_lr_' + str(np.log10(args.lr).round(3)) + '_batch_' + str(args.batch_size) + '.png'
    plt.savefig('./save/' + fname)
    plt.close()

    plt.figure(2)
    plt.title("Accuracy vs Iteration")
    plt.xlabel("Iteration number(batch: %d, learning rate: %f)" % (args.batch_size, args.lr))
    plt.ylabel("Accuracy")

    train_plot, = plt.plot(iters, train_accuracy, label='train set')
    valid_plot, = plt.plot(iters, val_accuracy, label='valid set')

    plt.legend([train_plot, valid_plot], ['train set', 'valid set'])
    fname = 'Acc_lr_' + str(np.log10(args.lr).round(3)) + '_batch_' + str(args.batch_size) + '.png'

    plt.savefig('./save/' + fname)
    plt.close()
