from Context import *    
import matplotlib.pyplot as plt
import numpy as np
from Algorithms.TS_Learner import *
from Algorithms.UCB1_Learner import * 

ctx= Context()
item2_price_full = ctx.item2_full_price
promotion = [2,1,0,3]   #class1 = promo2; class2 = promo1 ....
days= 365
n_exp= 10
discounted_price = np.multiply(np.subtract(1,np.take(ctx.discount_promos,promotion)),item2_price_full)
conversion_rate_second = np.zeros((4))
for i in range(4):
   conversion_rate_second[i] = ctx.conversion_rate_second_element(discounted_price[i],i)
candidates_item1 = [2260.0,1910.0,2130.0, 2010.0, 2340.0]
ts_experiments = np.zeros((n_exp,days))
ucb_experiments = np.zeros((n_exp,days))
opt_experiments = np.zeros((n_exp,days))
for e in range(n_exp):
    ts_learner = TS_Learner(len(candidates_item1))
    ucb_learner = UCB1_Learner(len(candidates_item1))
    ts_daily_reward = []
    ucb_daily_reward = []
    opt_daily_reward = []
    for d in range(days):
        print(f"DAY: {d}")
        opt_reward = 0.0
        customer_per_class = [176, 185, 406, 248] # ctx.daily_customer()
        ts_reward =0.0
        ts_pulled_arm=ts_learner.pull_arm()
        ucb_reward =0.0
        ucb_pulled_arm=ucb_learner.pull_arm()
        #Calculate reward for the pulled arm
        tot_client=sum(customer_per_class)
        maximum_rewards=(tot_client*(max(candidates_item1) + max(discounted_price)))
        for customer in range(tot_client):
            category = np.random.choice(np.nonzero(customer_per_class)[0])
            customer_per_class[category] -= 1
            ts_buy_or_not_item1 = ctx.purchase_online_first_element(candidates_item1[ts_pulled_arm],category)
            ucb_buy_or_not_item1 = ctx.purchase_online_first_element(candidates_item1[ucb_pulled_arm],category)
            opt_buy_or_not_item1 = ctx.purchase_online_first_element(min(candidates_item1),category)
            if ts_buy_or_not_item1:
                ts_reward+=candidates_item1[ts_pulled_arm] + conversion_rate_second[category]*discounted_price[category]
            if ucb_buy_or_not_item1:
                ucb_reward+=candidates_item1[ucb_pulled_arm] + conversion_rate_second[category]*discounted_price[category]
            if (opt_buy_or_not_item1):
                opt_reward+= min(candidates_item1) + conversion_rate_second[category]*discounted_price[category]
        opt_daily_reward.append(opt_reward)
        ts_daily_reward.append(ts_reward)
        ucb_daily_reward.append(ucb_reward)
        ts_learner.update(ts_pulled_arm,ts_reward/maximum_rewards)
        ucb_learner.update(ucb_pulled_arm,ucb_reward/maximum_rewards)
        print(f"ts_pulled_arm: {ts_pulled_arm}; UCB_pulled_arm: {ucb_pulled_arm}")
    ts_experiments[e,:]= ts_daily_reward
    ucb_experiments[e,:]= ucb_daily_reward
    opt_experiments[e,:]= opt_daily_reward

"""
plt.figure(0)
plt.xlabel("day")
plt.ylabel("Daily reward")
plt.plot(ucb_daily_reward,'-o', color='red', label = 'UCB Strategy')
plt.plot(ts_daily_reward,'-o', color='green', label = 'TS Strategy')
plt.legend()"""

plt.figure(0)
plt.xlabel("day")
plt.ylabel("Cumulative reward")
plt.plot(np.mean(np.cumsum(ucb_experiments,axis=1),axis=0),'-', color='red', label = 'UCB Strategy')
plt.plot(np.mean(np.cumsum(ts_experiments,axis=1),axis=0),'-', color='green', label = 'TS Strategy')
plt.legend()


plt.figure(1)
plt.xlabel("day")
plt.ylabel("Regret")
plt.plot(np.mean(np.cumsum(opt_experiments,axis=1),axis=0)-np.mean(np.cumsum(ucb_experiments,axis=1),axis=0),'-', color='red', label = 'UCB Strategy - OPT')
plt.plot(np.mean(np.cumsum(opt_experiments,axis=1),axis=0)-np.mean(np.cumsum(ts_experiments,axis=1),axis=0),'-', color='green', label = 'TS Strategy - OPT')
plt.legend()

plt.show()



        
