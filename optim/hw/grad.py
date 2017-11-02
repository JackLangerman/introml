import numpy as np

def costGrad(A, y, w):
	def g(z, y):
		return (y - 1/z) ** 2

	z = np.matmul(A, w)
	J = np.sum( g(z, y) )	

	def gprime(z, y):
		return (2*((1/z) - y)) / (z**2)

	grad = np.matmul(A.T, gprime(z, y))

	return J, grad


def costGrad(A, y, w):
	def g(z, y):
		return (y - 1/z) ** 2

	z = np.matmul(A, w)
	J = np.sum( g(z, y) )	

	def gprime(z, y):
		return (2*((1/z) - y)) / (z**2)

	grad = np.zeros_like(np.matmul(A.T, gprime(z, y)))

	for j in range(len(grad)):
		for i in range(A.shape[0]):
			grad[j] += gprime(z[i], y[i]) * A[i][j] 

	return J, grad


# test code
A = np.column_stack((np.ones(5), np.random.randn(5, 3)))
w = 0.1 * np.random.randn(4)
y = 1/np.matmul(A, w)
w = 1e-3 * np.random.randn(4)

print("A:")
print(A)
print("w:")
print(w)

print("\n Aw:")
print(np.matmul(A, w))

print("\n\n")

J, grad = costGrad(A, y, w)

print("J: {}".format(J))
print("grad: ")
print(grad)

# grad checking
print("\n"*4, "Grad Checking")

def checkGrad(analytic, A, y, w):
	Ja, grada = analytic(A, y, w)

	gradn = np.zeros_like(grada)
	
	h = 1e-4
	for i, _ in enumerate(w):

		wp = list(w)
		wp[i] += h

		print(w - wp)

		Jn, _ = analytic(A, y, wp)

		gradn[i] = (Ja - Jn)

	gradn /= h

	print("analytic \t\t numerical \t\t diff")
	for u in zip(grada, gradn, grada-gradn):
		print(("{:.3f}\t\t"*3).format(*u))

checkGrad(costGrad, A, y, w)


# grad decent
print("\n"*4, "Grad Decent")
print("cost \t deltaCost \t\t\t gradient")
alpha = 1e-3

J = 0
for i in range(30):
	J0 = J
	J, grad = costGrad(A, y, w)
	print("{:.4f}  {:.4f}".format(J, J-J0), "\t|  ",("{:.3f}   "*4).format(*grad))
	w = w - alpha * grad







