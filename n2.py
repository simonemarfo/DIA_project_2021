from Context import *    
import matplotlib.pyplot as plt
from Algorithms.UCB_Matching import *

ctx= Context()

days = 365 # 365 days of simulations

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

for t in range(days): # Day simulation
    # 1. Generate daily customers according the Context distributions, divided in categories
    daily_customer = ctx.customers_daily_instance()
    cum_UCB_rewards = 0
    cum_opt_rewards = 0
    
    for category in range(len(daily_customer)):
        for customer in range(daily_customer[category]): # for each category emulate the user that purchase the good 

            #2. Purchase simulation of the first element. (no optimization strategy)
            reward_item1 = ctx.purchase_online_first_element(item1_price_full,category) * item1_price_full
            cum_UCB_rewards += reward_item1
            cum_opt_rewards += reward_item1

            #3. Propose the second item only if the first one was bought
            if (reward_item1 > 0):
                #4. Query the learner to know wath is the best matching strategy category-promotion 
                sub_matching = learner.pull_arm() # suboptimal matching. row_ind, col_ind

                #5. Propose the second item to the user, using the promotion that retrieved by the learner (according to the user category)                    
                reward_item2 = ctx.purchase_online_second_element(discounted_price[sub_matching[1][category]],category) # 0: not purchased, 1: purchased

                #6. update the learner according to the obtained reward. rewards_to_update is a 4-zeros array, except for the element representing the current user category that contain the obtained reward
                rewards_to_update = np.zeros ((4))
                rewards_to_update[category] = reward_item2 
                learner.update(sub_matching,rewards_to_update)

                # store results in the cumulative daily rewards 
                cum_UCB_rewards += (reward_item2* discounted_price[sub_matching[1][category]])
                cum_opt_rewards += (ctx.purchase_online_second_element(discounted_price[opt[1][category]],category) * discounted_price [opt[1][category]]) # simulate the purchase of the second item according to the optimal strategy

    period_UCB_reward.append(cum_UCB_rewards)
    period_opt_reward.append(cum_opt_rewards)
    
    print('___________________')
    print(f'| Day: {t+1}')
    print(f'| Today customers distribution : {daily_customer}')
    print(f'| Today cumulative reward (Online strategy): {cum_UCB_rewards}\n| Today cumulative reward (Optimal strategy): {cum_opt_rewards}\n| - Loss: {cum_opt_rewards - cum_UCB_rewards}')
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
        
    