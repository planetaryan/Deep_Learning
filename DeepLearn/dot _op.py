import numpy as np

# a=np.random.rand(1000000)
# b=np.random.rand(1000000)

#matrix=np.zeroes((rows,cols))
#matrix=np.dot(w.T,X)+b

#w.T is w transposed
#X is xi vectors placed side hy side(nx X m)
#b is expanded to give row vector

A=np.array([[56.0,0.0,4.4,68.0],[1.2,104.0,52.0,8.0],[1.8,135.0,99.0,0.9]])
# print(A)

cal=A.sum(axis=0)
percentage=100*A/cal.reshape(1,4)

print(percentage)


# c=np.dot(a,b)
# print(c)