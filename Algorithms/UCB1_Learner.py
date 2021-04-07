from .Learner import *
import numpy as np

class UCB1_Learner(Learner):
    def __init__(self,n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.zeros(n_arms)

    def pull_arm(self):
        if self.t < self.n_arms:
            arm = self.t
        else:
            upper_bound = self.empirical_means + self.confidence
            arm = np.random.choice(np.where(upper_bound == upper_bound.max())[0])
        return arm
        
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm,reward)
        #self.collected_rewards = np.append(self.collected_rewards, reward)
        self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm]*(self.t-1)+reward)/self.t
        for a in range(self.n_arms):
            number_pulled = max(1,len(self.rewards_per_arm[a]))
            self.confidence[a] = (2*np.log(self.t) / number_pulled)**0.5    
        #self.rewards_per_arm[pulled_arm].append(reward)
        #two commented line has do the same thing of update_observations(), but professor split is. why?