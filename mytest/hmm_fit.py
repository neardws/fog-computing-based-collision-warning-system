#coding=utf-8
'''
Created on 2018-1-22

@author: 10205025
'''
import numpy as np
from hmmlearn import hmm

# 这里假设隐藏层数量为5个
model = hmm.MultinomialHMM(n_components=3, verbose=True, n_iter=1000, tol=0.001)
# model = hmm.GaussianHMM(n_components=3, n_iter=1000, tol=0.1,covariance_type="full", verbose=True)

X1 = np.array([[2], [1],[0]])
X2 = np.array([[2], [1],[0],[2]])
X3 = np.array([[2], [1],[1]])
X4 = np.array([[2], [1],[0]])
X5 = np.array([[1], [2],[0]])

X = np.vstack((X1,X2,X3,X4,X5))
print(X)
# [[2]
#  [1]
#  [0]
#  [2]
#  [1]
#  [0]
#  [2]
#  [2]
#  [1]
#  [1]
#  [2]
#  [1]
#  [0]
#  [1]
#  [2]
#  [0]]

# 这里分别为X1,X2,X3,X4,X5的长度
X_lens = [3,4,3,3,3]
model.fit(X,X_lens)

# 转换矩阵
print(model.transmat_)
# [[  4.90994062e-267   8.00000000e-001   1.00000000e-001   1.00000000e-001
#     4.90994062e-267]
#  [  1.00000000e-001   2.00000000e-001   3.00000000e-001   3.00000000e-001
#     1.00000000e-001]
#  [  5.00000000e-001   3.59090699e-133   2.80458184e-133   2.80458184e-133
#     5.00000000e-001]
#  [  5.00000000e-001   3.59090699e-133   2.80458184e-133   2.80458184e-133
#     5.00000000e-001]
#  [  4.90994062e-267   8.00000000e-001   1.00000000e-001   1.00000000e-001
#     4.90994062e-267]]

# 正常的序列
test1 = np.array([[2, 1,0]]).T
print(test1)
# [[2]
#  [1]
#  [0]
#  [2]
#  [1]
#  [0]]
# score = model.score(test1)
# print(score)

squence = model.predict(test1)
print(squence)
# 10.1943163957

# 不正常的序列
test2 = np.array([[2, 1,0,2,1,0,3]]).T
print(test2)
# [[2]
#  [1]
#  [0]
#  [2]
#  [1]
#  [0]
#  [3]]
# score = model.score(test2)
# print(score)
# -137.8727309
