import numpy 
import matplotlib.pyplot as plt
import math

#alaska = 1, canada = 0
def getMeans(X, y):
	m1 = numpy.zeros((1,X.shape[1]))
	m0 = numpy.zeros((1,X.shape[1]))
	c1 = 0
	c0 = 0
	for i in range(len(y)) :
		if str.lower(y[i]) == 'alaska':
			m1 += X[i]
			c1 += 1
		else:
			m0 += X[i]
			c0 += 1
	m1 = m1/c1
	m0 = m0/c0
	return m0,m1

def getPhi(y):
	size = len(y)
	c1 = 0
	for i in range(size):
		if str.lower(y[i]) == 'alaska':
			c1 += 1
	return c1/size

def getSigma(X, y, m0, m1):
	Z = numpy.zeros(X.shape)
	for i in range(X.shape[0]):
		if str.lower(y[i]) == 'alaska':
			Z[i] = X[i] - m1
		else:
			Z[i] = X[i] - m0
	Z = numpy.dot(Z.T, Z)/X.shape[0]
	return Z

# m0 is 1 x n vector
# phi a scalar
# sigma n x n matrix
# x (1 x n) vector

def probY(phi, m, sigma, x):
	sigma = numpy.mat(sigma)
	det = numpy.linalg.det(sigma)
	prob = (phi)/math.sqrt(det)
	z = (x-m).dot(sigma.I).dot((x-m).T)
	prob = prob * math.exp(-z/2)
	#print(1/math.exp(z/2))
	return prob	

def getSigmaI(X, Y, m, yi = 'alaska'):
	z = numpy.empty(X.shape)
	c = 0
	for i,y in enumerate(Y):
		if str.lower(y) == yi:
			z[c] = (X[i]-m)
			c += 1
	z = z[:c,:]
	return (numpy.dot(z.T, z)/c)	
	
def normalize(X):
	m = X.mean(axis=0)
	sd = X.std(axis=0)
	Z = numpy.empty(X.shape)
	for i,rec in enumerate(X):
		Z[i] = (rec - m)/sd
	return Z

xFile = open('q4x.dat', 'r')
yFile = open('q4y.dat', 'r')
X = []
for x in xFile:
	val = x.split()
	val = list(map(lambda x: int(x), val))
	X.append(val)
X = numpy.array(X)
Y = []
for y in yFile:
	Y.append(y.strip())

X = normalize(X)
#print(X)
m0, m1 = getMeans(X,Y)
phi = getPhi(Y)
sigma = getSigma(X,Y,m0,m1)
sigma0 = getSigmaI(X, Y, m0, 'canada')
sigma1 = getSigmaI(X, Y, m1, 'alaska')

print('Phi \n', phi)
print('Mean of distribution corresponding to Canada class \n', m0)
print('Mean of distribution corresponding to Alaska class \n', m1)
print('Covariance of distribution \n', sigma)
print('Covariance of distribution corresponding to Canada class \n', sigma0)
print('Covariance of distribution corresponding to Alaska class \n', sigma1)

mins = X.min(axis=0)
maxm = X.max(axis=0)

xx = numpy.linspace(mins[0]-0.5, maxm[0]+0.5, 100)
yy = numpy.linspace(mins[1]-0.5, maxm[0]+0.5, 100)
xx,yy = numpy.meshgrid(xx,yy)

zz = numpy.zeros(xx.shape)
for i in range(xx.shape[0]):
	for j in range(xx.shape[1]):
		zz[i][j] = (probY(phi, m1, sigma, numpy.array([xx[i,j], yy[i,j]])))/(probY(1-phi, m0, sigma, numpy.array([xx[i,j], yy[i,j]])))

plt.figure()
sub = plt.subplot(111)
for i,place in enumerate(Y):
	if str.lower(place) == 'alaska':
		sca = sub.scatter(X[i][0], X[i][1], 25, 'r', marker='+')
	else:
		scc = sub.scatter(X[i][0], X[i][1], 25, 'g', marker='*')
sca.set_label('Alaska')
scc.set_label('Canada')
sub.contour(xx, yy, zz, levels=[1])
sub.legend()
plt.show()

zz1 = numpy.zeros(xx.shape)
for i in range(xx.shape[0]):
	for j in range(xx.shape[1]):
		zz1[i][j] = (probY(phi, m1, sigma1, numpy.array([xx[i,j], yy[i,j]])))/(probY(1-phi, m0, sigma0, numpy.array([xx[i,j], yy[i,j]])))

plt.figure()
sub = plt.subplot('111')
for i,place in enumerate(Y):
	if str.lower(place) == 'alaska':
		sca = sub.scatter(X[i][0], X[i][1], 25, 'r', marker='+')
	else:
		scc = sub.scatter(X[i][0], X[i][1], 25, 'g', marker='*')
sca.set_label('Alaska')
scc.set_label('Canada')
sub.contour(xx, yy, zz, levels=[1])
sub.contour(xx, yy, zz1, levels=[1], colors='r')
sub.legend()
plt.show()

