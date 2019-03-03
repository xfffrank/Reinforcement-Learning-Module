import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

num_of_states = 40
episodes = 400
initial_lr = 1.0
min_lr = 0.005
gamma = 0.99
max_iter = 8000
# epsilon = 0.02

env = env.unwrapped
env.seed(0)
np.random.seed(0)


def discretization(env, obs):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_den = (env_high - env_low) / num_of_states
    pos_den = env_den[0]
    vel_den = env_den[1]
    # pos_high = env_high[0]
    pos_low = env_low[0]
    # vel_high = env_high[1]
    vel_low = env_low[1]
    pos_scaled = int((obs[0] - pos_low) / pos_den)
    vel_scaled = int((obs[1] - vel_low) / vel_den)
    return pos_scaled, vel_scaled

def run_simulation(env, policy):
    obs = env.reset()
    while True:
        env.render()
        pos, vel = discretization(env, obs)
        a = policy[pos][vel]
        obs, reward, terminate,_ = env.step(a)
        if terminate: 
            break

def train_and_test(epsilon, simulation=False, plot=False):
    print('epsilon:', epsilon)
    q_table = np.zeros((num_of_states, num_of_states, env.action_space.n))
    Q_hist = []
    for episode in range(episodes):
        # sum of the action values for the current episode
        Q_curr = 0
        obs = env.reset()  # returns an initial observation
        total_reward = 0
        alpha = max(min_lr, initial_lr * (gamma ** (episode // 100)))
        #iter_count = 0
        for i in range(max_iter):
            pos, vel = discretization(env, obs)
            if np.random.uniform(low=0, high=1) < epsilon:
                # exploration
                a = np.random.choice(env.action_space.n)
            else:
                # exploitation
                a = np.argmax(q_table[pos][vel])
            obs, reward, terminate, _ = env.step(a)
            total_reward += abs(obs[0]+0.5)
            pos_, vel_ = discretization(env, obs)
            q_table[pos][vel][a] = (1 - alpha) * q_table[pos][vel][a] + alpha * (reward + gamma * np.max(q_table[pos_][vel_]))
            # update the sum of the action values
            Q_curr += q_table[pos][vel][a]
            #iter_count += 1
            if terminate: 
                # print(iter_count)
                break
        Q_hist.append(Q_curr)
        if episode % 50 == 0:
            print('Episode: %s, Total reward: %s' % (episode, total_reward))
    if simulation:
        solution_policy = np.argmax(q_table, axis=2)
        run_simulation(env, solution_policy)
    # As the learning progresses, a fewer number of actions is needed for completing the task,
    # so the sum of Q value is decreasing.
    if plot:
        plt.plot(Q_hist, label='epsilon=' + str(epsilon))
        plt.xlabel('episodes')
        plt.ylabel('Sum of action value')
        plt.title('Episodes: %s, Max Iterations for one episode: %s' % (episodes, max_iter))
        plt.legend()
        plt.show()


if __name__ == "__main__":
    print('max_iter: %s, episodes: %s' % (max_iter, episodes))
    for epsilon in [0.05]:
        train_and_test(epsilon, simulation=True, plot=False)
    # for epsilon in [0.005, 0.05, 0.5]:
    #     train_and_test(epsilon, simulation=False, plot=True)
    env.close()
