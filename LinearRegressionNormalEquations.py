import numpy
import matplotlib.pyplot as plt

def normalize(X):
	m = X.mean()
	sd = X.std()
	Z = numpy.empty(X.shape)
	for i,rec in enumerate(X):
		Z[i] = (rec - m)/sd
	return Z

def analyticalMethod(X, y):
	theta = X.T*X
	theta = theta.I
	theta = theta*X.T
	theta = theta * y
	return theta

weightedXFileName = 'weightedX.csv'
weightedYFileName = 'weightedY.csv'

weightedXFile = open(weightedXFileName, 'rt')
weightedYFile = open(weightedYFileName, 'rt')

tempX = numpy.loadtxt(weightedXFile, delimiter=',')
y = numpy.loadtxt(weightedYFile, delimiter=',')

tempX = normalize(tempX)
X = [[1.0, x] for x in tempX]
X = numpy.mat(X)
y = y.reshape((len(y),1))

theta = analyticalMethod(X,y)
hx = X*theta

plt.figure()
plt.title('Linear Regression (Unweighted) using Normal Equations')
plt.scatter(tempX, y, 30, color='green', label='data')
plt.plot(tempX, hx, 'b-', label='hypothesis')
plt.legend()
plt.show()