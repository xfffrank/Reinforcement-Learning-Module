import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(0)

class MountainCar:

    def __init__(self, env):
        self.env = env
        self.num_of_states = 40
        self.episodes = 400
        # self.episodes = 50
        self.initial_lr = 1.0
        self.min_lr = 0.005
        self.gamma = 0.99
        self.max_iter = 8000
        self.env = env.unwrapped
        self.env.seed(0)
        np.random.seed(0)
        print('max_iter: %s, episodes: %s' % (self.max_iter, self.episodes))

    def discretization(self, env, obs):
        env_low = env.observation_space.low
        env_high = env.observation_space.high
        env_den = (env_high - env_low) / self.num_of_states
        pos_den = env_den[0]
        vel_den = env_den[1]
        # pos_high = env_high[0]
        pos_low = env_low[0]
        # vel_high = env_high[1]
        vel_low = env_low[1]
        pos_scaled = int((obs[0] - pos_low) / pos_den)
        vel_scaled = int((obs[1] - vel_low) / vel_den)
        return pos_scaled, vel_scaled

    def train(self, epsilon):
        print('epsilon:', epsilon)
        q_table = np.zeros((self.num_of_states, self.num_of_states, self.env.action_space.n))
        for episode in range(self.episodes):
            # sum of the action values for the current episode
            Q_curr = 0
            obs = self.env.reset()  # returns an initial observation
            total_reward = 0
            alpha = max(self.min_lr, self.initial_lr * (self.gamma ** (episode // 100)))
            #iter_count = 0
            for i in range(self.max_iter):
                pos, vel = self.discretization(self.env, obs)
                if np.random.uniform(low=0, high=1) < epsilon:
                    # exploration
                    a = np.random.choice(self.env.action_space.n)
                else:
                    # exploitation
                    a = np.argmax(q_table[pos][vel])
                obs, reward, terminate, _ = self.env.step(a)
                total_reward += abs(obs[0]+0.5)
                pos_, vel_ = self.discretization(self.env, obs)
                q_table[pos][vel][a] = (1 - alpha) * q_table[pos][vel][a] + alpha * (reward + self.gamma * np.max(q_table[pos_][vel_]))
                # update the sum of the action values
                Q_curr += q_table[pos][vel][a]
                if terminate: 
                    # print(iter_count)
                    break
            if episode % 50 == 0:
                print('Episode: %s, Total reward: %s' % (episode, total_reward))
        self.q_table =q_table

    def test(self, policy):
        obs = self.env.reset()
        while True:
            self.env.render()
            pos, vel = self.discretization(self.env, obs)
            a = policy[pos][vel] # ensure the action is an integer
            obs, reward, terminate,_ = self.env.step(a)
            if terminate: 
                break


class RBFModel(MountainCar):
    
    def __init__(self, env):
        super().__init__(env)

    def construct_dataset(self):
        # flatten = self.q_table.flatten()
        # non_zero = 0
        # for q in flatten:
        #     if q != 0:
        #         non_zero += 1
        data = np.zeros([self.num_of_states * self.num_of_states * self.env.action_space.n, 4])
        print('data shape:' ,data.shape)
        count = 0
        for i in range(self.q_table.shape[0]):
            for j in range(self.q_table.shape[1]):
                for k in range(self.env.action_space.n):
                    data[count][0] = i
                    data[count][1] = j
                    data[count][2] = k
                    data[count][3] = self.q_table[i][j][k]
                    count += 1
        # print('data shape:', data.shape)
        # print('data first 50:', data[0:50])
        X = data[:, :data.shape[1] - 1]
        y = np.expand_dims(data[:, data.shape[1] - 1], axis=1)
        # print(self.solution_policy[25][3])
        print('X shape:', X.shape)
        print('y shape:', y.shape)
        return X, y

    def approximate(self):
        print('train...')
        self.train(epsilon=0.05)
        policy = np.argmax(self.q_table, axis=2)
        self.test(policy)
        print('construct dataset from the learned policy...')
        X, y = self.construct_dataset()
        features = X[:, :2]
        actions = X[:, 2]
        print('construct design matrix...')
        J = 4  # number of clusters
        print('clusters:', J)
        U = self.gen_design_matrix(features, J)
        print('sgd training...')
        w0 = np.random.rand(J, self.env.action_space.n)
        alpha = 0.001
        max_iter = 10
        cost_hist = []
        for _ in range(max_iter):
            for i, u in enumerate(U):
                u = np.expand_dims(u, axis=0)
                gd = self.gradient(w0[:, int(actions[i])], u, y[i])
                w0[:, int(actions[i])] = w0[:, int(actions[i])] - alpha * gd
                cost = self.cost_func(w0[:, int(actions[i])], u, y[i])
                cost_hist.append(cost)
        print('last 5 error:', cost_hist[-5:])
        plt.plot(cost_hist)
        plt.show()
        # y_rbf = np.dot(U, w0)
        y_rbf = np.zeros(y.shape)
        for i, u in enumerate(U):
            y_rbf[i] = u.dot(w0[:, int(actions[i])])
        q_table_n = np.zeros(self.q_table.shape)
        count = 0
        for i in range(self.q_table.shape[0]):
            for j in range(self.q_table.shape[1]):
                for k in range(3):
                    q_table_n[i][j][k] = y_rbf[count]
                    count += 1
        policy_n = np.argmax(q_table_n, axis=2)
        self.test(policy_n)
          
    def gen_design_matrix(self, X, J):
        kmeans = KMeans(n_clusters=J, random_state=0).fit(X)
        sigma = np.std(X)
        # sigma = np.std(kmeans.cluster_centers_, axis=1)
        N = X.shape[0]
        U = np.zeros([N, J])
        for i in range(N):
            for j in range(J):
                u = np.linalg.norm(X[i] - kmeans.cluster_centers_[j])
                U[i][j] = np.exp(- np.square(u / sigma))
        print('design matrix shape:', U.shape)
        return U

    def gradient(self, w, X, y):
        m = len(X)
        return (- 1 / m) * (X.T.dot(y - X.dot(w)))

    def cost_func(self, w, X, y):
        # mean square error
        return np.square(y - X.dot(w)).mean() / 2



def main():
    env = gym.make('MountainCar-v0')
    p = RBFModel(env)
    p.approximate()
    env.close()


if __name__ == "__main__":
    main()
    # env = gym.make('MountainCar-v0')
    # p = MountainCar(env)
    # p.train(0.05)
    # p.test()
    # env.close()
