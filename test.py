# import pickle
# x = pickle.load(open('./data1/data.p', 'rb'), encoding='latin1')
#
# word2idx = x[6]
#
#
# count = 0
# total = 0
# with open('pos_pred.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         if line.strip('\n') == 'neg':
#             count += 1
#         total += 1
# print(count / total)
#
# import re
# def clean_str(string):
#     """
#     clean every review
#     """
#     string = re.sub(r"[^\u4e00-\u9fff]", "", string)
#     string = re.sub(r"没有描述", "", string)
#     string = re.sub(r"暂时还没有发现缺点哦", "", string)
#     # string = re.sub(r"还没(有)?发现(缺点)?", "", string)
#     # string = re.sub(r"")
#     # string = re.sub(r"(暂时|暂)(还)?(没|没有)(发现)?(什么)?(缺点|不足)?(哦|呢|呀|吧)?", "", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return(string.strip())
# from collections import Counter
# res = []
# with open('./data1/neg.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         res.append(clean_str(line))
# c = Counter(res)
# write_f = open('./data1/neg_cleaned.txt', 'w', encoding='utf-8')
# with open('./data1/neg.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         temp = clean_str(line)
#         if c[temp] > 100 and temp not in ['价格有点贵', '声音有点大', '价格贵了点', '声音有点小', '价格有点高',
#                                           '噪音有点大', '金子相当不错', '有点小贵', '价格太贵了', '一分钱一分货', '充电时间长']:
#             continue
#         write_f.write(line)

#
# import pickle
# x = pickle.load(open('./data1/data.p', 'rb'), encoding='latin1')
# train, val, test = x[0], x[1], x[2]
# print(len(train))
# print(len(val))
# print(len(test))


# #
# import torch
# import torch.nn as nn
# word_to_ix = {"hello": 0, "world": 1}
# embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
# lookup_tensor = torch.tensor([[0, 1, 0, 1],[0, 1, 1, 1]], dtype=torch.long)
# hello_embed = embeds(lookup_tensor)

import numpy as np
import torch
import torch.nn as nn
X = torch.rand(50, 1, 100, 300)
print( "X's size", X.shape)
m = nn.Conv2d(1, 2, (5, 300)) # creates a test convolution
Z = m(X)
print("Z's ", Z.shape)


