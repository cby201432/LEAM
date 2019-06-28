import torch
import sys
import pickle
import jieba
import numpy as np
from torch.nn import functional as F
from utils import get_minibatches_idx, prepare_data_for_emb, visualize
from model import LEAM
from preprocess import convert_word2idx
import numpy as np


def training(args):
    x = pickle.load(open(args.filename, 'rb'), encoding='latin1')
    train, val, test = x[0], x[1], x[2]
    train_lab, val_lab, test_lab = x[3], x[4], x[5]
    del x
    print("load data finished")

    train_lab = np.array(train_lab, dtype='float32')
    val_lab = np.array(val_lab, dtype='float32')
    test_lab = np.array(test_lab, dtype='float32')
    model = LEAM(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.part_data:
        # np.random.seed(123)
        train_ind = np.random.choice(len(train), int(len(train) * args.portion), replace=False)
        train = [train[t] for t in train_ind]
        train_lab = [train_lab[t] for t in train_ind]

    if args.restore:
        try:
            checkpoint = torch.load(args.save_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch_start = checkpoint['epoch']
            losses = checkpoint['losses']
            max_val_accuracy = checkpoint['max_val_accuracy']
            max_test_accuracy = checkpoint['max_test_accuracy']
            val_accuracies = checkpoint['val_accuracies']
            train_accuracies = checkpoint['train_accuracies']
            model.train()
        except:
            print("No saving session")
    else:
        epoch_start = 1
        max_val_accuracy = 0.
        max_test_accuracy = 0.
        losses = []
        val_accuracies = []
        train_accuracies = []
    steps = 0
    for epoch in range(epoch_start, args.n_epochs + 1):
        print("Starting epoch %d" % epoch)
        kf = get_minibatches_idx(len(train), args.batch_size, shuffle=True)
        for _, train_index in kf:
            steps += 1
            sents = [train[t] for t in train_index]
            x_labels = [train_lab[t] for t in train_index]
            x_labels = torch.FloatTensor(x_labels)
            x_labels = x_labels.reshape((len(x_labels), args.num_class))
            x_batch, x_batch_mask = prepare_data_for_emb(sents, args)
            optimizer.zero_grad()
            logits, logits_class, _ = model(x_batch, x_batch_mask)

            class_y = torch.LongTensor(np.arange(args.num_class))
            class_x = torch.max(x_labels, 1)[1]
            loss = torch.mean(F.cross_entropy(logits, class_x)) + \
                   args.class_penalty * torch.mean(F.cross_entropy(logits_class, class_y))
            loss.backward()
            optimizer.step()

            if steps % args.valid_freq == 0:
                train_correct = 0
                # sample evaluate accuaccy on 500 sample data
                kf_train = get_minibatches_idx(1000, args.batch_size, shuffle=True)
                for _, train_index in kf_train:
                    train_sents = [train[t] for t in train_index]
                    train_labels = [train_lab[t] for t in train_index]
                    train_labels = torch.FloatTensor(train_labels)
                    train_labels = train_labels.reshape((len(train_labels), args.num_class))
                    x_train_batch, x_train_batch_mask = prepare_data_for_emb(train_sents, args)
                    train_logits, _, _ = model(x_train_batch, x_train_batch_mask)
                    train_correct += (torch.max(train_logits, 1)[1].data == torch.max(train_labels, 1)[1].data).sum().item()

                train_accuracy = train_correct / 1000
                print("Iteration %d: Training loss %f " % (steps, loss))
                print("Train accuracy %f " % train_accuracy)

                losses.append(loss)
                train_accuracies.append(train_accuracy)


                val_correct = 0.0
                kf_val = get_minibatches_idx(len(val), args.batch_size, shuffle=True)
                for _, val_index in kf_val:
                    val_sents = [val[t] for t in val_index]
                    val_labels = [val_lab[t] for t in val_index]
                    val_labels = torch.FloatTensor(val_labels)
                    val_labels = val_labels.reshape((len(val_labels), args.num_class))
                    x_val_batch, x_val_batch_mask = prepare_data_for_emb(val_sents, args)
                    val_logits, _, _ = model(x_val_batch, x_val_batch_mask)
                    val_correct += (torch.max(val_logits, 1)[1].data == torch.max(val_labels, 1)[1].data).sum().item()

                val_accuracy = val_correct / len(val)
                print("Validation accuracy %f " % val_accuracy)
                val_accuracies.append(val_accuracy)

                if val_accuracy > max_val_accuracy:
                    max_val_accuracy = val_accuracy

                    test_correct = 0.0

                    kf_test = get_minibatches_idx(len(test), args.batch_size, shuffle=True)
                    for _, test_index in kf_test:
                        test_sents = [test[t] for t in test_index]
                        test_labels = [test_lab[t] for t in test_index]
                        test_labels = torch.FloatTensor(test_labels)
                        test_labels = test_labels.reshape((len(test_labels), args.num_class))
                        x_test_batch, x_test_batch_mask = prepare_data_for_emb(test_sents, args)
                        test_logits, _, _ = model(x_test_batch, x_test_batch_mask)
                        test_correct += (torch.max(test_logits, 1)[1].data == torch.max(test_labels, 1)[1].data).sum().item()

                    test_accuracy = test_correct / len(test)
                    print("Test accuracy %f " % test_accuracy)
                    max_test_accuracy = test_accuracy
        visualize(args, losses, train_accuracies, val_accuracies)
        print("Epoch %d: Max Test accuracy %f" % (epoch, max_test_accuracy))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'max_val_accuracy': max_val_accuracy,
            'max_test_accuracy': max_test_accuracy,
            'val_accuracies': val_accuracies,
            'train_accuracies': train_accuracies,
            }, args.save_path)
    print("Max Test accuracy %f " % max_test_accuracy)


def predict(args):
    print("loading model")
    model = LEAM(args)
    model.eval()
    checkpoint = torch.load(args.save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    predicted_sents = sys.stdin.readlines()
    class_name = ['pos', 'neg']
    for sent in predicted_sents:
        temp = list(jieba.cut(sent.strip('\n'), cut_all=False))
        if not temp:
            sys.stdout.write('empty review cannot be predicted\n')
        x = [temp]
        x = convert_word2idx(x, args.word2idx)
        x, x_mask = prepare_data_for_emb(x, args)
        logits, _, beta = model(x, x_mask)
        if args.predict == 'key_words':
            n = len(temp)
            beta = np.array(beta.squeeze().data)
            key_words = []
            idx = beta.argsort()
            num_key = min(n // 5, 5)
            num_key = max(num_key, 1)
            count = 0
            j = 1
            while(count < num_key and j <= len(beta)):
                if idx[-j] < n:
                    key_words.append(temp[idx[-j]])
                    count += 1
                j += 1

            sys.stdout.write(' '.join(key_words) + '\n')
        elif args.predict == 'pos':
            class_x = torch.max(logits, 1)[1]
            sys.stdout.write(class_name[class_x[0]] + '\n')
        else:
            print("oops, mistake!")

