import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.axes import Axes


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
linear_loss = np.square(y - yh_lin).mean()
print(linear_loss)

def validate_mse():
    mse_hist = []
    for J in range(1, 100):
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
        loss = np.square(y - yh_rbf).mean()
        mse_hist.append(loss)
        if J % 20 == 0:
            print(J, loss)
    plt.axhline(y=linear_loss, label='linear', color='orange')
    plt.plot(mse_hist, label='RBF')
    plt.xlabel('number of basis function')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    

def test():
    J = 15
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
    plt.xlabel('true values')
    plt.ylabel('predicted values')
    plt.plot(y, yh_lin, '.', Color='magenta')
    plt.show()
    plt.xlabel('true values')
    plt.ylabel('predicted values')
    plt.plot(y, yh_rbf, '.', Color='cyan')
    print(np.square(y - yh_rbf).mean())
    plt.show()
    print(np.linalg.norm(y-yh_lin), np.linalg.norm(y-yh_rbf))    



# validate_mse()  
# test()                       