import numpy
import math
import matplotlib.pyplot as plt

def normalize(X):
	m = X.mean(axis=0)
	sd = X.std(axis=0)
	Z = numpy.empty(X.shape)
	for i,rec in enumerate(X):
		Z[i] = (rec - m)/sd
	return Z

def sigmoid(x, theta):
	return 1/(1+math.exp(-numpy.dot(x,theta)))

def hThetaX(X, theta):
	z = numpy.dot(X,theta)
	h = 1/(1 + numpy.exp(-z))
	return h

def getGradient(X, Y, theta):
	h = hThetaX(X, theta)
	grad = numpy.dot(X.T, (Y-h))
	return grad

def getHessian(X, theta):
	predicted = hThetaX(X,theta)
	m = predicted.shape[0]
	D = numpy.eye(m)
	for i,val in enumerate(predicted):
		D[i][i] = -val*(1-val)
	return numpy.dot(X.T, numpy.dot(D, X))

def getTheta(X, Y, theta):
	hessian = numpy.mat(getHessian(X, theta))
	gradient = numpy.mat(getGradient(X, Y, theta))
	theta = theta - hessian.I*gradient
	return theta

tempX = numpy.loadtxt('logisticX.csv', delimiter=',')
Y = numpy.loadtxt('logisticY.csv', delimiter=',')

tempX = normalize(tempX)
X = [ numpy.insert(x, 0, 1) for x in tempX]
X = numpy.array(X)
#print(X)
Y = Y.reshape((Y.shape[0],1))
#print(Y)
theta = numpy.zeros((X.shape[1],1))
grad = getGradient(X,Y, theta)
m = max(grad)
while m > 0.0000001:
	t = theta
	theta = getTheta(X, Y, theta)
	m = max(abs(theta-t))
#print(theta)
#theta = [[ 0.40125316],[2.5885477] ,[-2.72558849]]

print('Theta')
print(theta)
plt.figure()
ax = plt.subplot('111')
for i,y in enumerate(Y):
	if y == 0: 
		sc0 = plt.scatter(X[i,1], X[i,2], 30, color='red', marker='+')
	else:
		sc1 = plt.scatter(X[i,1], X[i,2], 30, color='blue', marker='*')

sc0.set_label('0')
sc1.set_label('1')
mins = tempX.min(axis = 0)
maxm = tempX.max(axis = 0)
x1 = numpy.linspace(mins[0]-1,maxm[0]+1,100)
x2 = numpy.linspace(mins[1]-1,maxm[1]+1,100)
x1,x2 = numpy.meshgrid(x1,x2)
z = numpy.zeros(x1.shape)
for i in range(x1.shape[0]):
	for j in range(x1.shape[1]):
		z[i][j] = sigmoid(numpy.array([1,x1[i][j], x2[i][j]]), theta)
plt.contour(x1, x2, z, levels=[0.5])
#prob = hThetaX(X,theta)
#print(prob)
#plt.plot()
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.legend()
plt.show()
