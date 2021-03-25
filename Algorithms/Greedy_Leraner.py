from Learner import *
import numpy as np

"""
    greedy learner always pull the arm that previously give the best reward 
"""
class Greedy_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms)

    def pull_arm(self):
        if(self.t < self.n_arms): # first round try all the possible arms 
            return self.t
        idxs = np.argwhere(self.expected_rewards == self.expected_rewards.max()).reshape(-1) # array composed by the indexes of the heigher elements in self.expected_rewards 
        pulled_arm = np.random.choice(idxs) # randomly choose an element of the passed array
        return pulled_arm
    
    def update(self, pulled_arm,reward):
        self.t += 1
        self.update_observations(pulled_arm,reward)
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.t -1) + reward) / self.t # calculate the reward of that arm that we expect if pulled next
        
        