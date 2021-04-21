from Context import *    
import matplotlib.pyplot as plt
import numpy as np
from Algorithms.TS_Learner import *
from Algorithms.UCB1_Learner import * 

ctx= Context()
item2_price_full = ctx.item2_full_price # default is 
promotion_assignment = [2,1,0,3]   # class1: P2; class2:P1; class3:P0; class4:P3. is the optimal solution found with n1.py
days= 10

discounted_price = ctx.discuonted_second_item_prices(promotion_assignment) # retrun the discounted prices for every customer category, according to the pormotion assignment

conversion_rate_second = np.zeros((4))
for i in range(4):
   conversion_rate_second[i] = ctx.conversion_rate_second_element(discounted_price[i],i)

# define the prices candidates for the first item
candidates_item1 = [2260.0,1910.0,2130.0, 2010.0, 2340.0]

n_exp= 20
observation=5000
ts_experiments = np.zeros((n_exp,observation))
ucb_experiments = np.zeros((n_exp,observation))
opt_experiments = np.zeros((n_exp,observation))
for e in range(n_exp):
    ts_learner = TS_Learner(len(candidates_item1))
    ucb_learner = UCB1_Learner(len(candidates_item1))
    ts_reward = []
    ucb_reward = []
    opt_reward = []
    ts_avg_reward=[]
    ucb_avg_reward=[]
    opt_avg_reward=[]
    # parameter used to normalize the reward
    maximum_rewards = ( max(candidates_item1) + max(discounted_price) ) 
    for d in range(days):

        # extract the daily customer. It is not unknown and is used to estimate the maximum possible reward
        customer_per_class = ctx.customers_daily_instance() #[176, 185, 406, 248] # 
        daily_customer_weight = customer_per_class.copy()
        tot_client=sum(customer_per_class)
        ts_daily_reward=[]
        ucb_daily_reward=[]
        opt_daily_reward=[]
        # simulate the day client by client, proposing the first item at the price provided by teh learner
        for customer in range(tot_client):

            ts_costumer_reward=0
            ucb_costumer_reward=0
            opt_costumer_reward=0
            # ask to the learner to pull the most promising price that maximize the reward
            ts_pulled_arm = ts_learner.pull_arm()
            ucb_pulled_arm = ucb_learner.pull_arm()
            category = np.random.choice(np.nonzero(customer_per_class)[0])
            customer_per_class[category] -= 1

            ts_buy_or_not_item1 = ctx.purchase_online_first_element(candidates_item1[ts_pulled_arm],category) 
            ucb_buy_or_not_item1 = ctx.purchase_online_first_element(candidates_item1[ucb_pulled_arm],category)
            opt_buy_or_not_item1 = ctx.purchase_online_first_element(min(candidates_item1),category)
            # the profit from the sale of the first item is added to the estimation of the rewenue that the customer buy the second item (depend only form the user category) 
            if ts_buy_or_not_item1:
                #ts_daily_reward += candidates_item1[ts_pulled_arm] + conversion_rate_second[category]*discounted_price[category]
                ts_costumer_reward=candidates_item1[ts_pulled_arm] + conversion_rate_second[category]*discounted_price[category]
            if ucb_buy_or_not_item1:
                #ucb_daily_reward += candidates_item1[ucb_pulled_arm] + conversion_rate_second[category]*discounted_price[category]
                ucb_costumer_reward=candidates_item1[ucb_pulled_arm] + conversion_rate_second[category]*discounted_price[category]
            if (opt_buy_or_not_item1):
                opt_costumer_reward = min(candidates_item1) + conversion_rate_second[category]*discounted_price[category]
                #opt_daily_reward += min(candidates_item1) + conversion_rate_second[category]*discounted_price[category]
            # for each costumer update the learner normalizing the reward
            ts_learner.update(ts_pulled_arm,ts_costumer_reward/maximum_rewards)
            ucb_learner.update(ucb_pulled_arm,ucb_costumer_reward/maximum_rewards)
            # collect the daily rewards
            ts_reward.append(ts_costumer_reward)
            ucb_reward.append(ucb_costumer_reward)
            opt_reward.append(opt_costumer_reward)



        print('___________________')
        print(f'| Day: {d+1} - Experiment {e+1}')
        print(f'| Today customers distribution : {daily_customer_weight}')
        """print(f'| [UCB] - {ctx.items_info[0]["name"]} price: {candidates_item1[ucb_pulled_arm]} € - Today reward:\t{round(ucb_costumer_reward,2)}')
        print(f'| [TS]  - {ctx.items_info[0]["name"]} price: {candidates_item1[ts_pulled_arm]} € - Today reward:\t{round(ts_reward,2)}')
        print(f'| [OPT] - {ctx.items_info[0]["name"]} price: {min(candidates_item1)} € - Today reward:\t{round(opt_reward,2)}')
        print(f'| Prices for {ctx.items_info[1]["name"]}: {discounted_price} € per customer category')"""
        
    # end experiment 
    ts_experiments[e,:]= ts_reward[:observation]
    ucb_experiments[e,:]= ucb_reward[:observation]
    opt_experiments[e,:]= opt_reward[:observation]

plt.figure(0)
plt.xlabel("day")
plt.ylabel("Cumulative reward")
#plt.plot(np.mean(np.cumsum(ucb_reward,axis=1),axis=0),'-', color='red', label = 'UCB Learner')
#plt.plot(np.mean(np.cumsum(ts_reward,axis=1),axis=0),'-', color='green', label = 'TS Learner')
plt.plot(np.mean(np.cumsum(ts_experiments,axis=1),axis=0),'-', color='red', label = 'TS Learner')
plt.plot(np.mean(np.cumsum(ucb_experiments,axis=1),axis=0),'-', color='green', label = 'UCB Learner')
plt.plot(np.mean(np.cumsum(opt_experiments,axis=1),axis=0),'-', color='blue', label = 'Opt')
plt.legend()


plt.figure(1)
plt.xlabel("day")
plt.ylabel("Regret")
plt.plot(np.mean(np.cumsum(opt_experiments,axis=1),axis=0)-np.mean(np.cumsum(ucb_experiments,axis=1),axis=0),'-', color='red', label = 'Optimal - UCB')
plt.plot(np.mean(np.cumsum(opt_experiments,axis=1),axis=0)-np.mean(np.cumsum(ts_experiments,axis=1),axis=0),'-', color='green', label = 'Optimal - TS')
#plt.plot(np.mean(np.cumsum(opt_avg_reward)-np.cumsum(ucb_avg_reward)),'-', color='red', label = 'Optimal - UCB')
#plt.plot(np.cumsum(opt_avg_reward)-np.cumsum(ts_avg_reward),'-', color='green', label = 'Optimal - TS')
plt.legend()

plt.show()



        
