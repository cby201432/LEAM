import torch
import torch.nn as nn
from torch.nn import functional as F


class LEAM(nn.Module):
    def __init__(self, args):
        super(LEAM, self).__init__()
        """
        Arguments
        ---------
        batch_size : Size of the batch
        num_class : 2 (pos, neg)
        hidden_size : Size of the hidden_state of MLP
        vocab_size : Size of the vocabulary containing unique words
        embed_size : Embeddding dimension
        W_emb : Pre-trained word_embeddings for X with dim(vocab_size, embed_size)
        W_class_emb: embeddings for label with dim (num_class, embed_size)
        ngram: length of phrase (2*r + 1)
        --------

        """

        self.batch_size = args.batch_size
        self.num_class = args.num_class
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        self.embed_size = args.embed_size
        self.W_emb = args.W_emb
        self.W_class_emb = args.W_class_emb
        self.dropout1 = nn.Dropout(args.dropout)
        self.dropout2 = nn.Dropout(args.dropout)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.embedding_class = nn.Embedding(self.num_class, self.embed_size)
        self.embedding.weight = nn.Parameter(self.W_emb, requires_grad=True)
        self.embedding_class.weight = nn.Parameter(self.W_class_emb, requires_grad=True)

        self.conv = nn.Conv2d(2, 2, (args.ngram, 1), padding=((args.ngram - 1) // 2, 0))

        self.fc_layer1 = nn.Linear(self.embed_size, self.hidden_size)
        self.label = nn.Linear(self.hidden_size, self.num_class)

    def MLP(self, H):

        """
        classifier
        """
        H = self.dropout1(H)
        H_dis = F.relu(self.fc_layer1(H))
        H_dis = self.dropout2(H_dis)
        labels = self.label(H_dis)
        return(labels)

    def forward(self, x, x_mask):

        """
        x: input sentence with shape (b, s) torch Variable
        x_mask: input sentence mask with shape (b, s) torch Variable
        y: label with shape(b, c)

        Returns
        -------
        Output the train data logits and label logits for pos & neg class

        """
        # b: batch size, s: sequence length, e: embedding dim, c : num of class
        x_emb = self.embedding(x) # b * s * e
        # y_pos = torch.argmax(y, -1) # b * 1
        # y_emb = self.embedding_class(y_pos) # b * e

        H_enc, Att_v_max = self.att_emb_ngram_encoder_maxout(x_emb, x_mask) # b * e
        logits = self.MLP(H_enc)
        logits_class = self.MLP(self.embedding_class.weight)

        return logits, logits_class, Att_v_max

    def att_emb_ngram_encoder_maxout(self, x_emb, x_mask):
        """
        output
        """
        x_mask = x_mask.unsqueeze(-1)  # b * s * 1
        x_emb_1 = torch.mul(x_emb, x_mask)  # b * s * e

        x_emb_norm = F.normalize(x_emb_1, p=2, dim=2)  # b * s * e
        W_class_norm = F.normalize(self.embedding_class.weight, p=2, dim=1)  # c * e
        G = torch.einsum("abc,cd->abd", (x_emb_norm, W_class_norm.t()))  # b * s * c
        G = G.permute(0, 2, 1) # b * c * s
        G = G.unsqueeze(-1) # b * c * s * 1
        Att_v = F.relu(self.conv(G))  # b * c * s *  1
        Att_v = Att_v.squeeze(-1) # b * c * s
        Att_v = Att_v.permute(0, 2, 1) # b * s * c

        Att_v = torch.max(Att_v, -1, True)[0]  # b * s * 1
        Att_v_max = self.partial_softmax(Att_v, x_mask, 1)  # b * s * 1

        x_att = torch.mul(x_emb, Att_v_max)  # b * s * e
        H_enc = torch.sum(x_att, 1)  # b * e
        return H_enc, Att_v_max

    def partial_softmax(self, logits, weights, dim):
        exp_logits = torch.exp(logits)
        exp_logits_weighted = torch.mul(exp_logits, weights)
        exp_logits_sum = torch.sum(exp_logits_weighted, dim, True)
        partial_softmax_score = torch.div(exp_logits_weighted, exp_logits_sum)
        return partial_softmax_score