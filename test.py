from Algorithms.Environment import Environment        
import matplotlib.pyplot as plt
import numpy as np
from Algorithms.UCB_Matching import *
from scipy.optimize import linear_sum_assignment

p= np.array([[1/4, 1, 1/4], [1/2, 1/4, 1/4], [1/4,1/4,1]])
opt=linear_sum_assignment(-p)
n_exp=1
T=50
regret_ucb=np.zeros((n_exp,T))

for e in range(n_exp):
    learner= UCB_Matching(p.size, *p.shape)
    print(e)
    rew_UCB=[]
    opt_rew=[]
    env=Environment(p.size,p)
    for t in range(T):
        pulled_arms=learner.pull_arm()
        rewards=env.round(pulled_arms)
        rewards = np.add(rewards,[1,3,2])
        learner.update(pulled_arms,rewards)
        rew_UCB.append(rewards.sum())
        opt_rew.append(p[opt].sum())
        print(t)
        print(pulled_arms)
        print(rewards)
        print(learner.confidence)
    regret_ucb[e,:]=np.cumsum(opt_rew)-np.cumsum(rew_UCB)
    
plt.figure(0)
plt.plot(regret_ucb.mean(axis=0))
plt.ylabel('Regret')
plt.xlabel('t')
plt.show()