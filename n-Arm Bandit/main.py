import matplotlib.pyplot as plt

from util import Bandit, Arm

def simulate_bandit(steps, runs, alpha=1.0, epsilon=0.1, q_init=[0, 0], policy='greedy-eps'):

    arms = [
            Arm(distribution_type='gaussian', distribution_args=(5, 10)),
            Arm(distribution_type='mixed gaussian', distribution_args=[(10, 15), (4, 10)])
           ]


    bandit = Bandit(arms=arms, alpha=alpha, epsilon=epsilon, q_init=q_init)
    if policy == 'greedy-eps':
        bandit.simulate(steps=steps, runs=runs)
    else:
        bandit.simulate2(steps=steps, runs=runs)

    ret = {
           'accumulated_avg': bandit.step_reward_average,
           'avg_action_val': bandit.average_action_values
           }

    return ret


if __name__ == "__main__":
    steps = 1000
    runs = 100
    alpha = [1, 2, 3, 4]
    alpha_titles = ['1', '0.9^k', '1 / (1 + Ln(1+k))', '1/k']
    epsilon = [0, 0.1, 0.2, 0.5]
    q_init = [[0, 0], [5, 7], [20, 20]]
    colors = ['r', 'g', 'b', 'y']
    alpha = [5]
    # alpha_titles = ['1 / (1 + Ln(1+k))']

    # Part A
    # plt.figure()
    #
    # for i in range(len(alpha)):
    #     plt.clf()
    #     for j in range(len(epsilon)):
    #         ret = simulate_bandit(steps, runs, alpha=alpha[i], epsilon=epsilon[j])
    #
    #         x = [0] + [i for i in range(1, steps+1)]
    #         y = [0] + ret['accumulated_avg']
    #
    #         plt.plot(x, y, color=colors[j], label='eps =' + str(epsilon[j]), linewidth=1)
    #
    #         # print(ret['avg_action_val'])
    #
    #     plt.xlabel('Steps')
    #     plt.ylabel('Average Accumulated Reward')
    #     plt.title('Alpha: ' + alpha_titles[i])
    #
    #     plt.grid()
    #     plt.legend()
    #
    #     # print(ret['arm_avg'])
    #
    #     plt.show()

    # Part B
    # plt.figure()
    #
    # for i in range(len(q_init)):
    #     ret = simulate_bandit(steps, runs, alpha=5, epsilon=0.1)
    #
    #     x = [0] + [i for i in range(1, steps+1)]
    #     y = [0] + ret['accumulated_avg']
    #
    #     plt.plot(x, y, color=colors[i], label='eps =' + str(q_init[i]), linewidth=1)
    #
    #     plt.xlabel('Steps')
    #     plt.ylabel('Average Accumulated Reward')
    #     plt.title('Alpha = 0.1, Eps=0.1')
    #
    #
    # plt.grid()
    # plt.legend()
    #
    # plt.show()


    # Parc C
    # plt.figure()
    #
    # ret = simulate_bandit(steps, runs, alpha=5, epsilon=0.1, policy='other')
    #
    # x = [0] + [i for i in range(1, steps+1)]
    # y = [0] + ret['accumulated_avg']
    #
    # plt.plot(x, y, color=colors[0], label='Gradient Policy', linewidth=1)
    #
    # ret = simulate_bandit(steps, runs, alpha=5, epsilon=0.1)
    #
    # x = [0] + [i for i in range(1, steps+1)]
    # y = [0] + ret['accumulated_avg']
    #
    # plt.plot(x, y, color=colors[1], label='greedy', linewidth=1)
    #
    # plt.xlabel('Steps')
    # plt.ylabel('Average Accumulated Reward')
    # plt.title('Gradient Policy vs Greedy-epsilon')
    #
    # plt.grid()
    # plt.legend()
    #
    # plt.show()



