from itertools import permutations
from .CD_UCB_Matching import *


class promo_category_UCB_CD_learner(CUMSUM_UCB_Matching):
    def __init__(self, n_arms, n_rows, n_cols, M=100, eps=0.05, h=20, alpha=0.01, starting_delay=1000,normalizing_value=1):
        super().__init__(n_arms, n_rows, n_cols,M,eps,h,alpha)
        self.permutations=list(permutations(range(0,self.n_rows)))
        self.tot_rew=np.zeros((self.n_rows,self.n_cols))
        self.support=np.zeros((self.n_rows,self.n_cols))
        self.perm_idx=0
        self.starting_delay=starting_delay
        self.normalizing_value=normalizing_value

    def pull_arm(self):
        if(self.perm_idx<=self.starting_delay):
            row_idx=list(range(0,self.n_rows))
            col_idx=self.permutations[self.perm_idx % len(self.permutations)]
            self.perm_idx+=1
            return[row_idx,col_idx]
        else:
            return super().pull_arm()
    
    
    def update(self, pulled_arms, reward,category):
        self.support[category][pulled_arms[1][category]]+=1
        self.tot_rew[category][pulled_arms[1][category]]+=reward
        if(self.perm_idx<=self.starting_delay):
            pass
        else:
            self.t+=1
            # calculate the rewards to pass to the learner
            rewards=np.zeros((self.n_rows))
            for c in range(self.n_rows):
                if(self.support[c][pulled_arms[1][c]]==0):
                    pass
                else:
                    rewards[c]=self.tot_rew[c][pulled_arms[1][c]]/(self.support[c][pulled_arms[1][c]]*self.normalizing_value)
            # update/change detection/reset
            pulled_arm_flat = np.ravel_multi_index(pulled_arms, (self.n_rows,self.n_cols))
            for pulled_arm, reward in zip(pulled_arm_flat,rewards):
                if self.change_detection[pulled_arm].update(reward) :#and self.support[pulled_arm//4][pulled_arm%4]>self.starting_delay/self.n_arms:
                    self.detections[pulled_arm].append(self.t)
                    self.valid_rewards_per_arms[pulled_arm] = []
                    self.change_detection[pulled_arm].reset()
                    #self.tot_rew[pulled_arm//4][pulled_arm%4]=0
                    #self.support[pulled_arm//4][pulled_arm%4]=0
                self.update_observations(pulled_arm, reward)
                self.empirical_means[pulled_arm] = np.mean(self.valid_rewards_per_arms[pulled_arm])
            total_valid_samples = sum([len(x) for x in self.valid_rewards_per_arms])
            for a in range(self.n_arms):
                n_samples=len(self.valid_rewards_per_arms[a])
                self.confidence[a] = (2*np.log(total_valid_samples)/n_samples)**0.5 if n_samples>0 else np.inf
    
