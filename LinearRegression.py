import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from matplotlib import cm

def cost(X, y, theta):
	hx = numpy.dot(X, theta)
	sqDiff = numpy.dot((hx - y).T , (hx - y)) * (0.5)
	return sqDiff

def modifyTheta(X, y, theta, learningRate):
	hxMinusY = numpy.dot(X, theta) - y
	grad = X.T.dot(hxMinusY)
	theta = theta - (learningRate) * grad
	return theta,grad


def normalize(X):
	m = X.mean(axis=0)
	sd = X.std(axis=0)
	Z = numpy.empty(X.shape)
	for i,rec in enumerate(X):
		Z[i] = (rec - m)/sd
	return Z

def learn(X, y, learningRate):
	theta0 = numpy.linspace(-2.5,3.5,100)
	theta1 = numpy.linspace(-3,3,100)
	theta0,theta1 = numpy.meshgrid(theta0,theta1)
	costs = numpy.zeros(theta0.shape)
	(r,c) = theta0.shape
	plt.ion()
	for i in range(r):
		for j in range(c):
			theta = numpy.array([[theta0[i][j]],[theta1[i][j]]])
			costs[i][j] = cost(X, y, theta)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	c = ax.contour(theta0, theta1, costs)
	ax.clabel(c, inline=1, fontsize=10)
	sc = ax.scatter([], [], 10, 'r')
	plt.title('Learning Rate, eta {}'.format(learningRate))
	theta = numpy.zeros((2,1))
	thetas = numpy.array([[0,0]])
	cost1 = [cost(X,y,theta)[0][0]]
	grad = numpy.ones((2,1))
	eps = 0.00000001
	i = 0
	while abs(max(grad)) > eps:
		theta,grad = modifyTheta(X,y,theta, learningRate)
		cost1.append(cost(X,y,theta)[0][0])
		sc.set_offsets(thetas[:,:])
		thetas = numpy.concatenate((thetas,theta.T), axis=0)
		fig.canvas.draw_idle()
		plt.pause(0.2)
		i+=1
	return thetas,cost1

def plot3D(X,y,learningRate):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	plt.title('3D Mesh')
	ax.set_xlabel('theta0')
	ax.set_ylabel('theta1')
	ax.set_zlabel('Error')
	theta0 = numpy.linspace(-5,5,100)
	theta1 = numpy.linspace(-5,5,100)
	theta0,theta1 = numpy.meshgrid(theta0,theta1)
	costs = numpy.zeros(theta0.shape)
	(r,c) = theta0.shape
	for i in range(r):
		for j in range(c):
			theta = numpy.array([[theta0[i][j]],[theta1[i][j]]])
			costs[i][j] = cost(X, y, theta)
	ax.plot_surface(X=theta0, Y=theta1, Z=costs, cmap=cm.coolwarm)
	theta = numpy.zeros((2,1))	
	thetas = numpy.array([[0,0]])
	cost1 = [cost(X,y,theta)[0][0]]
	sc = ax.scatter(thetas[:,0], thetas[:,1], cost1, s=20, c='r')
	grad = numpy.ones((2,1))
	eps = 0.00000001
	i = 0
	while abs(max(grad)) > eps:
		plt.pause(0.2)
		theta,grad = modifyTheta(X,y,theta, learningRate)
		cost1.append(cost(X,y,theta)[0][0])
		thetas = numpy.concatenate((thetas,theta.T), axis=0)
		sc._offsets3d = (thetas[:,0], thetas[:,1], numpy.array(cost1))
		plt.draw()
		i+=1
	

learningRate = 0.007
linearXFile = 'linearX.csv'
linearYFile = 'linearY.csv'

xfile = open(linearXFile, 'rt')
tempX = numpy.loadtxt(xfile, delimiter=',')
tempX = normalize(tempX)

yfile = open(linearYFile, 'rt')
y = numpy.loadtxt(yfile, delimiter=',')

y = y.reshape(len(y),1)

X = [ [1.0, x] for x in tempX]
X = numpy.array(X)
thetas, costs = learn(X, y, learningRate)
theta = thetas[-1]
plot3D(X,y,learningRate)

prediction = numpy.dot(X,theta)
print("Theta")
print(theta)
print('-'*50)
plt.figure()
plt.subplot('111')
plt.scatter(tempX, y, 20, color='green', label='data')
#plt.title('Scatter Plot of Dataset')
plt.plot(tempX, prediction, 'r-', label='hypothesis')
plt.legend()
plt.show()

learn(X,y, 0.001)
learn(X,y, 0.005)
learn(X,y, 0.009)
learn(X,y, 0.013)
learn(X,y, 0.017)
learn(X,y, 0.021)
learn(X,y, 0.025)