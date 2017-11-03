import numpy as np



def costGrad(A, w, x, y):
	z = np.matmul(A, w)
	m, n = A.shape
    
	return J, grad

A = np.column_stack(np.ones(m), np.random.randn(2,3))


costGrad(A, w, x, y)