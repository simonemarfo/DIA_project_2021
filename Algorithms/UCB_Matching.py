from .UCB1_Learner import *
import numpy as np
from scipy.optimize import linear_sum_assignment

class UCB_Matching(UCB1_Learner):
    def __init__(self,n_arms,n_rows,n_cols):
        super().__init__(n_arms)
        self.n_rows = n_rows
        self.n_cols = n_cols
        assert n_arms == n_cols * n_rows
    
    def pull_arm(self):
        upper_conf = self.empirical_means + self.confidence
        upper_conf[np.isinf(upper_conf)] = 1e3
        row_ind, col_ind = linear_sum_assignment(upper_conf.reshape(self.n_rows, self.n_cols), maximize=True)
        return (row_ind,col_ind)

    def update(self,pulled_arms,rewards):
        self.t += 1 
        pulled_arm_flat = np.ravel_multi_index(pulled_arms,(self.n_rows,self.n_cols))
        for pulled_arm, reward in zip(pulled_arm_flat,rewards):
            self.update_observations(pulled_arm,reward)
            self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm]*(self.t-1) + reward) / self.t
        for a in range(self.n_arms):
            n_sample = len(self.rewards_per_arm[a])
            number_pulled = max(1,len(self.rewards_per_arm[a]))
            self.confidence[a] = (2*np.log(self.t) / number_pulled)**0.5 if n_sample > 0 else np.inf
"""
from Environment import Environment        
import matplotlib.pyplot as plt

p= np.array([[1/4, 1, 1/4], [1/2, 1/4, 1/4], [1/4,1/4,1]])
opt=linear_sum_assignment(-p)
n_exp=1
T=3
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
        print ("REW")
        print(rewards)
        learner.update(pulled_arms,rewards)
        rew_UCB.append(rewards.sum())
        opt_rew.append(p[opt].sum())
    regret_ucb[e,:]=np.cumsum(opt_rew)-np.cumsum(rew_UCB)
    print(learner.confidence)
plt.figure(0)
plt.plot(regret_ucb.mean(axis=0))
plt.ylabel('Regret')
plt.xlabel('t')
plt.show()"""