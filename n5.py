from Context import *    
import matplotlib.pyplot as plt
from Algorithms.UCB_Matching import *

ctx= Context()

days = 15 # 365 days of simulations

item1_price_full = 1980.0
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

# experimet parameters
n_exp = 1


observation = (days//2)*1000
experiments = np.zeros((n_exp,observation))
days_experiments = np.zeros((n_exp,days))
for e in range(n_exp):
    period_UCB_reward = [] # rewards collected in a period (days) performing the online learning strategy
    period_opt_reward = [] # rewards collected in a period (days) performing the online learning strategy
    tot_rew = np.zeros((4,4))
    support = np.zeros((4,4))

    day_UCB_reward = [] 
    day_opt_reward = []

    learner = UCB_Matching(conversion_rate_second.size, *conversion_rate_second.shape) # Initialize UCB matching learner
    prev_size = 0
    for d in range(days): # Day simulation
        # 1. Generate daily customers according the Context distributions, divided in categories
        #rewards_to_update=[0.,0.,0.,0.]
        cont = 0
        daily_customer = ctx.customers_daily_instance()
        daily_customer_weight=daily_customer.copy()

        cum_UCB_rewards = 0
        cum_opt_rewards = 0
        category=0
        
        print(tot_rew)
        print(support)
        #input()

        if d == 0: # >= 
            tot_rew = np.zeros((4,4))
            support = np.zeros((4,4))
        else:
            #support = np.multiply(support, days-d) 
            #tot_rew = np.divide(tot_rew,support, out=np.zeros_like(tot_rew), where=support!=0)
            #
            # support = np.ones((4,4))
            print(tot_rew)
            print(support)
            #input()

        tot_client=sum(daily_customer)
        for customer in range(tot_client): # for each category emulate the user that purchase the good 
            customer_UCB_reward = 0
            customer_opt_reward = 0
            customer_item1_reward = 0

            category = np.random.choice(np.nonzero(daily_customer)[0])
            daily_customer[category] -= 1

            #2. Purchase simulation of the first element. (no optimization strategy)
            buy_item1 = ctx.purchase_online_first_element(item1_price_full,category) 
            customer_item1_reward = buy_item1*item1_price_full

            #3. Propose the second item only if the first one was bought
            if (buy_item1 > 0):
                #4. Query the learner to know wath is the best matching strategy category-promotion 
                sub_matching = learner.pull_arm() # suboptimal matching. row_ind, col_ind

                propose_price = discounted_price[sub_matching[1][category]]
                #5. Propose the second item to the user, using the promotion that retrieved by the learner (according to the user category)                    
                buy_item2 = ctx.purchase_online_second_element(propose_price,category) # 0: not purchased, 1: purchased
                # store results in the cumulative daily rewards 
                customer_UCB_reward = buy_item2 * propose_price
                customer_opt_reward = ctx.purchase_online_second_element(discounted_price[opt[1][category]],category) * discounted_price[opt[1][category]] # purchase of the second item according to the optimal strategy 

                support[category][sub_matching[1][category]] += 1
                tot_rew[category][sub_matching[1][category]] += customer_UCB_reward
                update_array = np.zeros((4))
                for c in range(4):
                    if support[c][sub_matching[1][c]] == 0:
                        pass
                    else:
                        update_array[c] = tot_rew[c][sub_matching[1][c]] / (support[c][sub_matching[1][c]] * item2_price_full) 

                print(update_array)
                learner.update(sub_matching,update_array)
  
                #update the learner
                pulled_category = [ [sub_matching[0][category]],[sub_matching[1][category]] ]
                reward = [ customer_UCB_reward / item2_price_full]

                print('___________________')
                print(f'| Day: {d+1} - Experiment {e+1}')
                print(f'| Today customers distribution : {daily_customer_weight}')
                print(f'| Customer #{customer} of category: {ctx.classes_info[category]["name"]}: ')
                print(f'/ <sub matching> : {sub_matching}')
                print(f'\ <opt matching> : {opt}')
                print(f'| UCB propose: {propose_price} -- Opt propose: {discounted_price[opt[1][category]]}')
                print(f'| UCB reward: {customer_UCB_reward} -- Opt reward: {customer_opt_reward}  --> Learner parameters < {pulled_category},{reward} >')
                print(f'| Loss: {customer_opt_reward - customer_UCB_reward} €')
                
            
            # item1 + item2 reward curve 
            period_UCB_reward.append(customer_UCB_reward)#(customer_item1_reward + customer_UCB_reward)
            period_opt_reward.append(customer_opt_reward)#(customer_item1_reward + customer_opt_reward)
        
        day_UCB_reward.append(sum(period_UCB_reward[prev_size:]))
        day_opt_reward.append(sum(period_opt_reward[prev_size:]))
        prev_size = len(period_UCB_reward)

        print(f'Current confidence per arm of the online learner:\n{learner.confidence}')
    experiments[e,:] = np.cumsum(period_opt_reward[:observation]) - np.cumsum(period_UCB_reward[:observation])
    days_experiments[e,:] = np.cumsum(day_opt_reward) - np.cumsum(day_UCB_reward)

# plot daily reward comparison
mean_UCB_reward = np.mean(period_UCB_reward)
mean_opt_reward = np.mean(period_opt_reward)
print(f"Mean daily reward using online UCB strategy: {mean_UCB_reward}")
print(f"Mean daily reward using optimal strategy: {mean_opt_reward}")
print(f'Period ({days} days) regret: {np.sum(period_opt_reward) - np.sum(period_UCB_reward)}')

plt.figure(0)
plt.title("Last experiment daily rewards")
plt.xlabel("day")
plt.ylabel("Daily reward ")
plt.plot(day_UCB_reward,'-o', color='red', label = 'UCB Strategy')
#plt.plot(days * [mean_UCB_reward],'--', color='lightcoral', label = 'Mean UCB Strategy')
plt.plot(day_opt_reward,'-o', color='blue', label = 'Optimal Strategy')
#plt.plot(days * [mean_opt_reward],'--', color='cornflowerblue', label = 'Mean Optimal Strategy')
plt.legend()


# plot regret of UCB

plt.figure(1)
plt.plot(experiments.mean(axis=0))
plt.ylabel('Regret')
plt.xlabel('Client#')


# plot regret of UCB

plt.figure(2)
plt.plot(days_experiments.mean(axis=0))
plt.ylabel('Regret')
plt.xlabel('Days')


plt.show()
        
    