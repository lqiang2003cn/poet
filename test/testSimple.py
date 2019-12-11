import numpy as np

size = 3

x = np.array([1,2,3,4,5,6])
y=x.reshape(3,2)
print(x)
print(y)


# x = np.array([[1,2,3,4,5,6],
#               [1,2,3,4,5,6],
#               [13,14,15,16,17,18],
#               [3,8,4,5,5,6],
#               [10,6,6,21,5,7],
#               [10,8,7,5,5,5]])
# m=x
#m = np.zeros((6,6),np.float32)
# for i in range(6):
#     m[i] = x
# print(m)
#
# result = np.sum(m, axis=0)
# print(result)
# max = np.amax(result, axis=0)
# max_ind=np.argmax(result,axis=0)
# print(max)
# print(max_ind)
# #
# row_sum = np.sum(m,axis=1)
# print('row sum is:',row_sum)
# o = np.argsort(row_sum)
# print('order is ',o)
# for i in range(len(o)):
#     if o[i] == 2:
#         md1 = i
#
# print('middle index are:',md1)
#
# rs_seeds = np.random.randint(np.int32(2 ** 31 - 1))
# print(rs_seeds)
#import uuid
# print(uuid.uuid1())
# p='x'+str(uuid.uuid1())
# print(p)


# for i in range(0,16):
#     print(i)
#
# for i in range(0,100):
#     seed = np.random.randint(100000000)
#     print(seed)


#weight_init = np.random.uniform(0,1,shape)

#test dot
# Y=[1,2]
# X = np.array([[10, 1,2,3,4], [2, 3,4,2,4]])
# print(X.shape)
# print(X[:, 0])
# print(X.sum())
# print(np.dot(Y,X))

# x=np.random.uniform(-1,1)
# print((x+1)*5)
# print()
#
#
# print(np.tanh(-0.1))
# print(np.tanh(np.exp(0.1)))

# rs = np.random.RandomState(np.random.seed())
# print(rs.uniform(7, 8))
# sumTest=np.array([[1,2,3],[30,4,5],[50,6,7]])
# print('the first items :',sumTest[:, 1])

# print(sumTest.sum())
# x=sumTest.ravel()
# print(x)
# #ranks = np.empty(len(x), dtype=int)
# ranks = np.array([100,200,300,400,500,600,700,800,900], dtype=int)
# print(ranks)
# print(x.argsort())
# print('np arange:',np.arange(9))
# print('ranks:',ranks[x.argsort()])
# ranks[x.argsort()] = np.arange(len(x))
# print(ranks)