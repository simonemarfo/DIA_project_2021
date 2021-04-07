import numpy as np 

# This envirnoment simulate 4 slot machines with different winning probabilities
class Environment():
    def __init__(self,n_arms,probabilities):
        self.n_arms = n_arms
        self.probabilities = probabilities # array of probability of n_arms length
        
    def round(self, pulled_arm):
        #return 1 or 0 (binomial distribution) according to a random choice in a given probability
        reward = np.random.binomial(1,self.probabilities[pulled_arm])  
        return reward