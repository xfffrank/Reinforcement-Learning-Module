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

plt.plot(y, yh_lin, '.', Color='magenta')
plt.show()
# J = 20basis functions obtained by k-means clustering
# sigma set to standard deviation of entire data
J = 14
kmeans = KMeans(n_clusters=J, random_state=0).fit(X)
sigma = np.std(X)

# Construct design matrix
U = np.zeros((N,J))
for i in range(N):
    for j in range(J):
       # U[i][j] = np.linalg.norm(X[i] - kmeans.cluster_centers_[j])
       u = np.linalg.norm(X[i] - kmeans.cluster_centers_[j])
       U[i][j] = np.exp(- np.square(u / sigma))
# Solve RBF model, predict and plot
w = np.dot((np.linalg.inv(np.dot(U.T,U))), U.T) * y
# w = np.dot((np.linalg.inv(np.dot(U.T,U))), U.T).dot(y)

yh_rbf = np.dot(U,w)
plt.plot(y, yh_rbf, '.', Color='cyan')
plt.axis([0,50,0,50])
print(np.square(y - yh_rbf).mean())
plt.show()
print(np.linalg.norm(y-yh_lin), np.linalg.norm(y-yh_rbf))