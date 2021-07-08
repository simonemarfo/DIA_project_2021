from Context import *    
import matplotlib.pyplot as plt
import numpy as np
from Algorithms.TS_Learner import *
from Algorithms.UCB1_Learner import * 

ctx= Context()
item2_price_full = ctx.item2_full_price # default is 630.0
promotion_assignment = [2,1,0,3]   # class1: P2; class2:P1; class3:P0; class4:P3. is the optimal solution found with n1.py

discounted_price = ctx.discuonted_second_item_prices(promotion_assignment) # retrun the discounted prices for every customer category, according to the pormotion assignment

conversion_rate_second = np.zeros((4))
for i in range(4):
   conversion_rate_second[i] = ctx.conversion_rate_second_element(discounted_price[i],i)

# define the prices candidates for the first item
candidates_item1 = [2260.0,1910.0,2130.0, 2010.0, 2340.0]

days = 10
n_exp = 1
observation = (days//2)*1000
ts_experiments = np.zeros((n_exp,observation))
ucb_experiments = np.zeros((n_exp,observation))
opt_experiments = np.zeros((n_exp,observation))
for e in range(n_exp):
    ts_learner = TS_Learner(len(candidates_item1))
    ucb_learner = UCB1_Learner(len(candidates_item1))
    
    ts_reward = []  # collects the rewards of the clients with the TS strategy
    ucb_reward = [] # collects the rewards of the clients with the UCB strategy
    opt_reward = [] # collects the rewards of the clients with the optiml strategy
    
    maximum_rewards = ( max(candidates_item1) + max(discounted_price)) # parameter used to normalize the reward
    for d in range(days):

        # extract the daily customer. It is known
        customer_per_class = ctx.customers_daily_instance()
        daily_customer_weight = customer_per_class.copy()
        tot_client=sum(customer_per_class)

        # simulate the day client by client, proposing the first item at the price provided by teh learner
        for customer in range(tot_client):

            ts_customer_reward  = 0 
            ucb_customer_reward = 0
            opt_customer_reward = 0
            
            # ask to the learner to pull the most promising price that maximize the reward
            ts_pulled_arm = ts_learner.pull_arm()
            ucb_pulled_arm = ucb_learner.pull_arm()
            # extraction of a client 
            category = np.random.choice(np.nonzero(customer_per_class)[0])
            customer_per_class[category] -= 1

            # propose the item1 with the price suggested by the learner
            ts_buy_or_not_item1 = ctx.purchase_online_first_element(candidates_item1[ts_pulled_arm],category) 
            ucb_buy_or_not_item1 = ctx.purchase_online_first_element(candidates_item1[ucb_pulled_arm],category)
            opt_buy_or_not_item1 = ctx.purchase_online_first_element(min(candidates_item1),category)
            
            # the profit from the sale of the first item is added to the estimation of the rewenue that the customer buy the second item (depend only form the user category) 
            if ts_buy_or_not_item1:
                ts_customer_reward=candidates_item1[ts_pulled_arm] + conversion_rate_second[category]*discounted_price[category]
            if ucb_buy_or_not_item1:
                ucb_customer_reward=candidates_item1[ucb_pulled_arm] + conversion_rate_second[category]*discounted_price[category]
            if (opt_buy_or_not_item1):
                opt_customer_reward = min(candidates_item1) + conversion_rate_second[category]*discounted_price[category]
                
            # for each customer update the learner normalizing the reward
            ts_learner.update(ts_pulled_arm,ts_customer_reward/maximum_rewards)
            ucb_learner.update(ucb_pulled_arm,ucb_customer_reward/maximum_rewards)
            
            print('___________________')
            print(f'| Day: {d+1} - Experiment {e+1}')
            print(f'| Today customers distribution : {daily_customer_weight}')
            print(f'| Customer #{customer} of category: {ctx.classes_info[category]["name"]}: ')
            print(f'|\t[UCB] - {ctx.items_info[0]["name"]} : {candidates_item1[ucb_pulled_arm]} €, {ctx.items_info[1]["name"]} : {discounted_price[category]} € -> Total reward : {round(ucb_customer_reward,2)} €')
            print(f'|\t[TS] - {ctx.items_info[0]["name"]} : {candidates_item1[ts_pulled_arm]} €, {ctx.items_info[1]["name"]} : {discounted_price[category]} € -> Total reward : {round(ts_customer_reward,2)} €')
            print(f'|\t[OPT] - {ctx.items_info[0]["name"]} : {min(candidates_item1)} €, {ctx.items_info[1]["name"]} : {discounted_price[category]} € -> Total reward : {round(opt_customer_reward,2)} €')
            
            # collect all the rewards
            ts_reward.append(ts_customer_reward)
            ucb_reward.append(ucb_customer_reward)
            opt_reward.append(opt_customer_reward)
                       
    # end experiment. save only the first <observation> value
    ts_experiments[e,:] = ts_reward[:observation]
    ucb_experiments[e,:]= ucb_reward[:observation]
    opt_experiments[e,:]= opt_reward[:observation]


plt.figure(1)
plt.xlabel("#sales")
plt.ylabel("Regret")
plt.plot(np.mean(np.cumsum(opt_experiments,axis=1),axis=0)-np.mean(np.cumsum(ucb_experiments,axis=1),axis=0),'-', color='red', label = 'Optimal - UCB')
plt.plot(np.mean(np.cumsum(opt_experiments,axis=1),axis=0)-np.mean(np.cumsum(ts_experiments,axis=1),axis=0),'-', color='green', label = 'Optimal - TS')
plt.title("Cumulative regret")
plt.legend()

plt.show()



        
