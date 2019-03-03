import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

rawData = np.genfromtxt('housing.data')
print(rawData.shape)
N, pp1 = rawData.shape

# Last column is target
X = np.matrix(rawData[:, 0:pp1-1])
y = np.matrix(rawData[:, pp1-1]).T
print(X.shape, y.shape)
# Solve linear regression, plot target and prediction
w = (np.linalg.inv(X.T * X)) * X.T * y  # pseudo-inverse
yh_lin = X * w

# plt.plot(y, yh_lin, '.', Color='magenta')
# plt.show()
# J = 20basis functions obtained by k-means clustering
# sigma set to standard deviation of entire data
J = 15
print('clusters:', J)
kmeans = KMeans(n_clusters=J, random_state=0).fit(X)
print('kmeans.cluster_centers shape', kmeans.cluster_centers_.shape)
sigma = np.std(X)
# sigma = np.std(kmeans.cluster_centers_, axis=1)
# print('sigma shape:', sigma.shape)

# Construct design matrix
U = np.zeros((N,J))
for i in range(N):
    for j in range(J):
       # U[i][j] = np.linalg.norm(X[i] - kmeans.cluster_centers_[j])
       u = np.linalg.norm(X[i] - kmeans.cluster_centers_[j])
       U[i][j] = np.exp(- np.square(u / sigma))

# Solve RBF model, predict and plot
def gradient(w, X, y):
    m = len(X)
    return (- 1 / m) * (X.T.dot(y - X.dot(w)))

# mean square error
def cost_func(w, X, y):
    return np.square(y - X.dot(w)).mean() / 2


# train, sgd
np.random.seed(0)
w0 = np.random.rand(J, 1)   # initialize w0

alpha = 0.005
max_iter = 50
cost_hist = []
for _ in range(max_iter):
    for i, x in enumerate(U):
        x = np.expand_dims(x, axis=0)
        gd = gradient(w0, x, y[i])
        w0 = w0 - alpha * gd
        cost = cost_func(w0, U, y)
        cost_hist.append(cost)

plt.plot(cost_hist)
plt.ylabel('mean square error')
plt.xlabel('iterations')
plt.show()
yh_rbf = np.dot(U, w0)
# print(np.square(y - yh_rbf).mean())
plt.plot(y, yh_rbf, '.', Color='cyan')
plt.axis([0,50,0,50])
plt.show()
# root square error
print(np.linalg.norm(y-yh_lin), np.linalg.norm(y-yh_rbf))