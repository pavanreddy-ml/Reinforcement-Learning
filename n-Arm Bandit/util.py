import matplotlib.pyplot as plt
import numpy as np
import random

class Arm:
    def __init__(self, distribution_type, distribution_args):

        if (distribution_type) not in  ['constant', 'gaussian', 'mixed gaussian']:
            print('Arm distribution not defined.')
            exit()
        self.distribution_type = distribution_type
        self.distribution_args = distribution_args
        self.gaussian_values = None
        
    def pull(self):

        if (self.distribution_type == 'constant'):
            constant = self.distribution_args
            return constant
        elif (self.distribution_type == 'gaussian'):
            mean, variance = self.distribution_args
            standard_deviation = np.sqrt(variance)
            sample = np.random.normal(mean, standard_deviation, 1)[0]
            return sample
        elif (self.distribution_type == 'mixed gaussian'):
            num_gaussians = len(self.distribution_args)

            assert num_gaussians > 1

            samples = []
            for mean, variance in self.distribution_args:
                standard_deviation = np.sqrt(variance)
                sample = np.random.normal(mean, standard_deviation, 1)[0]
                samples.append(sample)

            final_sample = np.random.choice(samples)
            return final_sample
        else:
            print('Arm distribution not defined.')
            exit()

class Bandit:

    def __init__(self, arms, alpha, epsilon, q_init):
        self.arms = arms
        assert len(self.arms) > 0
        self.rewards = []
        self.actions = []
        self.q = q_init
        self.q_num = [0, 0]

        self.step_reward_average = []

        self.arm_averages = []
        self.average_action_values = []

        self.h_funcs = np.array([0, 0])
        
        self.alpha = alpha
        self.epsilon = epsilon
        
    def simulate(self, steps, runs):
        for i in range(steps):
            self.rewards = []
            self.actions = []

            for j in range(runs):
                choice = random.choices([0, 1], weights=[1-self.epsilon, self.epsilon], k=1)

                if choice[0] == 0:
                    action = np.argmax(self.q)
                else:
                    action = np.random.choice([0, 1])

                reward = self.arms[action].pull()

                self.q[action] += self.get_alpha(self.alpha, step=i) * (reward - self.q[action])

                # self.q[action] = ((self.q[action] * self.q_num[action]) + reward) / (self.q_num[action]+1)
                # self.q_num[action] += 1

                self.rewards.append(reward)
                self.actions.append(action)


            if len(self.step_reward_average) == 0:
                step_avg = np.mean(self.rewards)
                self.step_reward_average.append(step_avg)
            else:
                step_avg = np.mean(self.rewards)
                cum_avg = self.step_reward_average[-1] * len(self.step_reward_average)
                cum_avg += step_avg
                cum_avg = cum_avg / (len(self.step_reward_average)+1)

                self.step_reward_average.append(cum_avg)

            if i == 0:
                self.arm_averages = self.get_arm_average()
                self.average_action_values.append(self.q)

    def simulate2(self, steps, runs):
        h_funcs = [0, 0]

        for i in range(steps):
            self.rewards = []
            self.actions = []

            r_hist = []
            for j in range(runs):
                policy = self.softmax(np.array(h_funcs))

                # print(policy)

                action = random.choices([0, 1], weights=list(policy))
                action = action[0]

                reward = self.arms[action].pull()

                self.rewards.append(reward)
                self.actions.append(action)

                r_hist.append(reward)
                avg_r = np.average(self.rewards)

                alp = self.get_alpha(self.alpha, step=i)

                h_funcs[action] = h_funcs[action] + (alp * (reward - avg_r) * (1 - policy[action]))
                h_funcs[:action] = h_funcs[:action] - (alp * (reward - avg_r) * policy[:action])
                h_funcs[action + 1:] = h_funcs[action + 1:] - alp * (reward - avg_r) * policy[action + 1:]

            if len(self.step_reward_average) == 0:
                step_avg = np.mean(self.rewards)
                self.step_reward_average.append(step_avg)
            else:
                step_avg = np.mean(self.rewards)
                cum_avg = self.step_reward_average[-1] * len(self.step_reward_average)
                cum_avg += step_avg
                cum_avg = cum_avg / (len(self.step_reward_average)+1)

                self.step_reward_average.append(cum_avg)

            if i == 0:
                self.arm_averages = self.get_arm_average()
                self.average_action_values.append(self.q)

    def softmax(self, H):
        h = H - np.max(H)
        exp = np.exp(h)
        return exp / np.sum(exp)


    def accumulated_rewards_avg(self, arr):
        x = [arr[0]]
        for i in range(1, len(arr)):
            num = x[-1] * (i)
            num += arr[i]
            num = num / (i+1)
            x.append(num)

        return x

    def get_average_reward(self):
        if len(self.step_reward_average) == 0: return 0
        else: return sum(self.step_reward_average) / len(self.step_reward_average)

    def get_arm_average(self):
        x = []
        for i in range(len(self.arms)):
            arm_avg = 0
            n = 0

            for reward, action in zip(self.rewards, self.actions):
                if action == i:
                    arm_avg += reward
                    n += 1

            if n == 0: n = 1
            x.append(arm_avg / n)

        print("aaaaaa", x)

        return x

    def get_alpha(self, num, step):
        if num == 1:
            return 1
        elif num == 2:
            return 0.9 ** (step+1)
        elif num == 3:
            return 1 / (1 + np.log(1+step+1))
        elif num == 4:
            return 1 / (step+1)
        elif num == 5:
            return 0.1
        else:
            raise ValueError("Invalid Learning Rate")