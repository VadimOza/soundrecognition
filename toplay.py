import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt

# oneArray = np.array([[1, 2], [2, 1], [5, 4]])
#
# print(oneArray)
#
# print(oneArray.sum(axis=1))
# some = oneArray.sum(axis=1)
#
# print('Some ', np.pad(some, ((0, 2)), mode='wrap'))
# oneArray = np.array([[1, 2, 3],
#                      [2, 1, 4],
#                      [5, 4, 5]])
#
# sns.heatmap(oneArray)
# print(oneArray.reshape(-1,1))
logits = [2.0, 1.0, 0.1]
import numpy as np

exps = [np.exp(i) for i in logits]

print(exps)

sums = sum(exps)
print('sums', sums)

softmax = [j / sums for j in exps]
print('softmax ', softmax)
