from Context import *    
import matplotlib.pyplot as plt
from Algorithms.UCB_Matching import *

ctx= Context()

days = 3 # 365 days of simulations

item1_price_full = 2350.0
item2_price_full = 630.0 

#discount for the second item 
discounted_price = [item2_price_full,
    item2_price_full*(1-ctx.discount_promos[1]),
    item2_price_full*(1-ctx.discount_promos[2]),
    item2_price_full*(1-ctx.discount_promos[3])]

print("\n\n#############\n")
print(f" {ctx.items_info[0]['name']}: {item1_price_full} €\n {ctx.items_info[1]['name']}: {item2_price_full} €\n Discouts (%): {[_*100 for _ in ctx.discount_promos]}")
print(f" Discounted {ctx.items_info[1]['name']}: {discounted_price} €")

# Matrix conversion rate of second item: rows[0..3] are the user categories; columns[0..3] are the discouts; celles are the conversion rate of that class
conversion_rate_second = np.zeros((4,4))
for i in range (0,4): #classes
    for j in range (0,4): #promos
        conversion_rate_second[i,j] = (ctx.conversion_rate_second_element(discounted_price[j], i))

# Computing an optimal solution to be compared with the online solutions
# Is computed using a matching algorithom on a matrix that takes into account the price and conversion rate for the second items, according to the user category and the discount
priced_conversion_rate_second = np.zeros((4,4))
for i in range (0,4): #classes
    for j in range (0,4): #promos
        priced_conversion_rate_second[i,j] = conversion_rate_second[i,j] * discounted_price[j]
opt = linear_sum_assignment(priced_conversion_rate_second, maximize=True) # optimal solution row_ind, col_ind

#
# ONLINE LEARNING AND SIMULATION
#

period_UCB_reward = [] # rewards collected in a period (days) performing the online learning strategy
period_opt_reward = [] # rewards collected in a period (days) performing the online learning strategy

learner = UCB_Matching(conversion_rate_second.size, *conversion_rate_second.shape) # Initialize UCB matching learner
rewards=[0.,0.,0.,0.]
maximum_reward=[0.,0.,0.,0.]
customers=[1.,1.,1.,1.]
cont=0
maximum_daily_reward=0
maximum_daily_reward_customer=1
combination=[[]]
for t in range(days): # Day simulation
    #4. Query the learner to know wath is the best matching strategy category-promotion 

    sub_matching = learner.pull_arm() # suboptimal matching. row_ind, col_ind
    print(learner.test)
    print(learner.empirical_means)
    # 1. Generate daily customers according the Context distributions, divided in categories
    rewards=[0.,0.,0.,0.]
    rewards_to_update=[0.,0.,0.,0.]
    to_append=1
    vet=sub_matching[1].tolist()
    for s in range (len(combination)):
        if(vet==combination[s]):
            to_append=0
    if(to_append):
        combination.append(vet)
    daily_customer = ctx.customers_daily_instance()
    daily_customer_weight=daily_customer.copy()
    cum_UCB_rewards = 0
    cum_opt_rewards = 0
    category=0
    tot_client=sum(daily_customer)
    for customer in range(tot_client): # for each category emulate the user that purchase the good 
        flag=0
        while(flag==0):
            category=np.random.randint(0,4)
            if (daily_customer[category]>0):
                daily_customer[category]-=1
                flag=1
        #2. Purchase simulation of the first element. (no optimization strategy)
        buy_item1 = ctx.purchase_online_first_element(item1_price_full,category) 
        cum_UCB_rewards += buy_item1*item1_price_full
        cum_opt_rewards += buy_item1*item1_price_full

        #3. Propose the second item only if the first one was bought
        if (buy_item1 > 0):
            #5. Propose the second item to the user, using the promotion that retrieved by the learner (according to the user category)                    
            buy_item2 = ctx.purchase_online_second_element(discounted_price[sub_matching[1][category]],category) # 0: not purchased, 1: purchased

            #6. update the learner according to the obtained reward. rewards_to_update is a 4-zeros array, except for the element representing the current user category that contain the obtained reward
            rewards_to_update[category] += buy_item2 * discounted_price[sub_matching[1][category]]

            # store results in the cumulative daily rewards 
            cum_UCB_rewards += (buy_item2 * discounted_price[sub_matching[1][category]])
            cum_opt_rewards += (buy_item2 * discounted_price [opt[1][category]]) # purchase of the second item according to the optimal strategy 
    
    for k in range(4):
        if((rewards_to_update[k]/daily_customer_weight[k])*customers[k]>=maximum_reward[k]*0.9):
            rewards[k]=1
            if((rewards_to_update[k]/daily_customer_weight[k])*customers[k]>=maximum_reward[k]):
                customers[k]=daily_customer_weight[k]
                maximum_reward[k]=rewards_to_update[k]
    """if(sum(rewards_to_update)/tot_client>=(maximum_daily_reward/maximum_daily_reward_customer)*0.9):
        rewards=[1,1,1,1]
        maximum_daily_reward=sum(rewards_to_update)
        maximum_daily_reward_customer=tot_client"""
    if(t<20):
        rewards=[0,0,0,0]
    print(opt)
    print(sub_matching)
    print(combination)
    print(len(combination))
    print(rewards)
    print(rewards_to_update)
    print(daily_customer_weight)
    print(maximum_reward)
    print(customers)
  
    learner.update(sub_matching,rewards)
    period_UCB_reward.append(cum_UCB_rewards)
    period_opt_reward.append(cum_opt_rewards)
    
    print('___________________')
    print(f'| Day: {t+1}')
    print(f'| Today customers distribution : {daily_customer_weight}')
    print(f'| Today cumulative reward (Online strategy):  {cum_UCB_rewards}\n| Today cumulative reward (Optimal strategy): {cum_opt_rewards}\n| - Loss: {cum_opt_rewards - cum_UCB_rewards}')
    print(f'Current confidence per arm of the online learner:\n{learner.confidence}')
    print('___________________\n')

# print(f"Period UCB Reward: {period_UCB_reward}")
# print(f"Period Optimal Reward: {period_opt_reward}")
   
mean_UCB_reward = np.mean(period_UCB_reward)
mean_opt_reward = np.mean(period_opt_reward)
print(f"Mean daily reward using online UCB strategy: {mean_UCB_reward}")
print(f"Mean daily reward using optimal strategy: {mean_opt_reward}")
print(f'Period ({days} days) regret: {np.sum(period_opt_reward) - np.sum(period_UCB_reward)}')

# plot daily reward comparison
plt.figure(0)
plt.xlabel("day")
plt.ylabel("Daily reward")
plt.plot(period_UCB_reward,'-o', color='red', label = 'UCB Strategy')
plt.plot(days * [mean_UCB_reward],'--', color='lightcoral', label = 'Mean UCB Strategy')
plt.plot(period_opt_reward,'-o', color='blue', label = 'Optimal Strategy')
plt.plot(days * [mean_opt_reward],'--', color='cornflowerblue', label = 'Mean Optimal Strategy')
plt.legend()

# plot regret of UCB

plt.figure(1)
plt.plot(np.cumsum(period_opt_reward) - np.cumsum(period_UCB_reward))
plt.ylabel('Regret')
plt.xlabel('day')

plt.show()
        
    