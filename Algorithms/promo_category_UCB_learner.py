from itertools import permutations
from .UCB_Matching import *

class promo_category_UCB_learner(UCB_Matching):
    def __init__(self,n_arms,n_rows,n_cols, starting_dalay = 1000, normalizing_value = 1):
        super().__init__(n_arms,n_rows,n_cols)
        self.permutations = list(permutations(range(0,self.n_rows)))
        self.tot_rew = np.zeros((self.n_rows,self.n_cols))
        self.support = np.zeros((self.n_rows,self.n_cols))
        self.starting_dalay = starting_dalay
        self.perm_idx = 0 
        self.normalizing_value = normalizing_value
    
    def pull_arm(self):
        # force an exploration phase in which all possible combinations are tested and the additional matrix are filled
        if self.perm_idx <= self.starting_dalay: 
            row_ind = list(range(0,self.n_rows))
            col_ind = self.permutations[self.perm_idx % len(self.permutations)]
            self.perm_idx += 1
            return [row_ind,col_ind]
        else: 
            return super().pull_arm()
    
    def update(self,pulled_arms,reward,category):
        # accept a reward > 1
        # store the reward in the matrix
        self.support[category][pulled_arms[1][category]] += 1
        self.tot_rew[category][pulled_arms[1][category]] += reward
        # update the confidence and empirical means
        if self.perm_idx <= self.starting_dalay:
            pass
        else:
            update_rewards = np.zeros((self.n_rows))
            for c in range(self.n_rows):
                if self.support[c][pulled_arms[1][c]] == 0:
                    pass
                else:
                    update_rewards[c] = self.tot_rew[c][pulled_arms[1][c]] / (self.support[c][pulled_arms[1][c]] * self.normalizing_value)
            super().update(pulled_arms,update_rewards)

    
