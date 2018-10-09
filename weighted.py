import math
import numpy
import matplotlib.pyplot as plt


def normalize(X):
	m = X.mean(axis=0)
	sd = X.std(axis=0)
	Z = numpy.empty(X.shape)
	for i,rec in enumerate(X):
		Z[i] = (rec - m)/sd
	return Z

def getWeightMatrix(X, featureX, t=1):
	n = X.shape[0]
	i = 0
	W = numpy.zeros((n,n))
	den = 2*t*t
	for xi in X:
		num = numpy.dot((xi-featureX), (xi-featureX))
		W[i][i] = math.exp(-num/den)
		i += 1
	#print('Weight Matrix')
	#print('-'*50)
	#print(W)
	#print()
	return W

def getParameters(X, W, y):
	theta = numpy.mat(numpy.dot(numpy.dot(X.T, W), X)).I
	theta = numpy.dot(numpy.dot(numpy.dot(theta, X.T),W),y)
	#print('Parameters')
	#print('-'*50)
	#print(theta)
	#print()
	return theta

# X : matrix of feature vectors
# x : vector for which y is to be predicted
def predict(X, Y, x, t):
	W = getWeightMatrix(X, x, t)
	theta = getParameters(X, W, Y)
	predicted = numpy.dot(x, theta)
	return predicted
	
xFile = open('weightedX.csv', 'r')
yFile = open('weightedY.csv', 'r')
tempX = numpy.loadtxt(xFile, delimiter=',')
tempX = normalize(tempX)
Y = numpy.loadtxt(yFile, delimiter=',')
Y = Y.reshape((Y.shape[0],1))
X = [[1.0, x] for x in tempX]
X = numpy.array(X)
predicted = numpy.array([])
predictOn = numpy.linspace(X.min(axis=0)[1]-0.5, X.max(axis=0)[1]+0.5, 20)
t = 0.8
for x in predictOn:
		predicted = numpy.append(predicted, predict(X, Y, numpy.array([1,x]), t))

plt.figure()
plt.scatter(tempX, Y, 40, color='red')
plt.plot(predictOn, predicted, 'b-')
plt.title('t = 0.8')
plt.show()

plt.figure()

del predicted

predicted = numpy.array([])
t = 0.1
for x in predictOn:
		predicted = numpy.append(predicted, predict(X, Y, numpy.array([1,x]), t))

plt.subplot('221')
plt.scatter(tempX, Y, 40, color='red')
plt.plot(predictOn, predicted, 'b-')
plt.title('t = 0.1')
del predicted

predicted = numpy.array([])
t = 0.3
for x in predictOn:
		predicted = numpy.append(predicted, predict(X, Y, numpy.array([1,x]), t))

plt.subplot('222')
plt.scatter(tempX, Y, 40, color='red')
plt.plot(predictOn, predicted, 'b-')
plt.title('t = 0.3')
del predicted

predicted = numpy.array([])
t = 2
for x in predictOn:
		predicted = numpy.append(predicted, predict(X, Y, numpy.array([1,x]), t))

plt.subplot('223')
plt.scatter(tempX, Y, 40, color='red')
plt.title('t = 2')
plt.plot(predictOn, predicted, 'b-')

del predicted

predicted = numpy.array([])
t = 10
for x in predictOn:
		predicted = numpy.append(predicted, predict(X, Y, numpy.array([1,x]), t))

plt.subplot('224')
plt.scatter(tempX, Y, 40, color='red')
plt.plot(predictOn, predicted, 'b-')
plt.title('t = 10')
plt.show()