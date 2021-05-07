from Context import *    
import matplotlib.pyplot as plt
from Algorithms.UCB_Matching import *
from Algorithms.TS_Learner import *

ctx = Context()

days = 20 # 365 days of simulations
days_matching = 365-days
item1_full_price=0.0
item2_full_price=0.0
# define the prices candidates for the first and second item
candidates_item1 = [2260.0, 1910.0, 2130.0, 2010.0, 2340.0]
candidates_item2 = [480.0, 550.0, 510.0, 420.0, 650.0]
#discounted_price = ctx.discuonted_second_item_prices(promotion_assignment) # retrun the discounted prices for every customer category, according to the pormotion assignment
# find the optimal solutions 
opt_rew_item1 = np.zeros((5))
opt_rew_item2 = np.zeros((5))

for i in range(len(candidates_item1)):
    for c in range(4):
            opt_rew_item1[i] += ctx.conversion_rate_first_element(candidates_item1[i],c) * candidates_item1[i] * ctx.customersDistribution[c,0]

for i in range(len(candidates_item2)):
    for c in range(4):
        opt_rew_item2[i] += ctx.conversion_rate_second_element(candidates_item2[i],c) * candidates_item2[i] * ctx.customersDistribution[c,0]

opt_item1 = np.argmax(opt_rew_item1)
opt_item2 = np.argmax(opt_rew_item2)



maximum_rewards_item1 = max(candidates_item1) + max(candidates_item2) # parameter used to normalize the reward
maximum_rewards_item2 = max(candidates_item2) # parameter used to normalize the reward

n_exp = 3
observation = (days//2)*1000
experiments = np.zeros((n_exp,observation))
experimets_item1_regret_curve = np.zeros((n_exp,observation))
experimets_item2_regret_curve = np.zeros((n_exp,observation))
experiments_matching = np.zeros((n_exp,days_matching))
for e in range(n_exp):
    ts_learner_item1 = TS_Learner(len(candidates_item1))
    ts_learner_item2 = TS_Learner(len(candidates_item2))

    opt_reward_item1 = []
    opt_reward_item2 = [] 
    ts_reward_item1 = []
    ts_reward_item2 = []    

    for d in range(days):
        # extract the daily customer. It is UNKNOWN
        customer_per_class = ctx.customers_daily_instance() 
        daily_customer_weight = customer_per_class.copy()
        tot_client = sum(customer_per_class)
        # simulate the day client by client
        for customer in range(tot_client):
            customer_reward_item1 = 0.0
            customer_reward_item2 = 0.0
            opt_customer_item1 = 0.0 # opt reward
            opt_customer_item2 = 0.0 # opt reward

            category = np.random.choice(np.nonzero(customer_per_class)[0])
            customer_per_class[category] -= 1

            # ask to the learner to pull the most promising price that maximize the reward
            ts_pulled_arm_item1 = ts_learner_item1.pull_arm()
            ts_pulled_arm_item2 = ts_learner_item2.pull_arm() #select candidates for the second item

            ts_buy_or_not_item1 = ctx.purchase_online_first_element(candidates_item1[ts_pulled_arm_item1],category) 
            opt_buy_or_not_item1 = ctx.purchase_online_first_element(candidates_item1[opt_item1],category)
            # compute the rewenue of the first and second item for both optimal solution and the online learning
            if ts_buy_or_not_item1:
                ts_buy_or_not_item2 = ctx.purchase_online_second_element(candidates_item2[ts_pulled_arm_item2],category) 
                
                # calculate the reward
                customer_reward_item2 = candidates_item2[ts_pulled_arm_item2] * ts_buy_or_not_item2
                customer_reward_item1 = candidates_item1[ts_pulled_arm_item1]

            if (opt_buy_or_not_item1):
                opt_buy_or_not_item2 = ctx.purchase_online_second_element(candidates_item2[opt_item2],category)

                # calculate the reward
                opt_customer_item2 =  candidates_item2[opt_item2] * opt_buy_or_not_item2
                opt_customer_item1 = candidates_item1[opt_item1] 

            # update the learner normalizing the reward. The learner for the second item is updated only the customer buy the first one
            ts_learner_item1.update(ts_pulled_arm_item1, (customer_reward_item1 + customer_reward_item2 )/maximum_rewards_item1)
            if ts_buy_or_not_item1:
                ts_learner_item2.update(ts_pulled_arm_item2, customer_reward_item2/maximum_rewards_item2)

            print('___________________')
            print(f'| Day: {d+1} - Experiment {e+1}')
            print(f'| Today customers distribution : {daily_customer_weight}')
            print(f'| Customer #{customer} of category: {ctx.classes_info[category]["name"]}: ')
            print(f'|\t[TS] - Selected prices -> {ctx.items_info[0]["name"]} : {candidates_item1[ts_pulled_arm_item1]} €, {ctx.items_info[1]["name"]} : {candidates_item2[ts_pulled_arm_item2]} €\n|\t\t{ctx.items_info[0]["name"]} reward : {round(customer_reward_item1,2)} € -- {ctx.items_info[1]["name"]} reward : {round(customer_reward_item2,2)} € -- Total : {round(customer_reward_item1 + customer_reward_item2,2)} €')
            print(f'|\t[OPT] -  Selected prices -> {ctx.items_info[0]["name"]} : {candidates_item1[opt_item1]} €, {ctx.items_info[1]["name"]} : {candidates_item2[opt_item2]} €\n|\t\t{ctx.items_info[0]["name"]} reward : {round(opt_customer_item1,2)} € -- {ctx.items_info[1]["name"]} reward : {round(opt_customer_item2,2)} € -- Total : {round(opt_customer_item1 + opt_customer_item2,2)} €')

            ts_reward_item1.append(customer_reward_item1)
            ts_reward_item2.append(customer_reward_item2)
            opt_reward_item1.append(opt_customer_item1)
            opt_reward_item2.append(opt_customer_item2)
            #print(ts_learner_item2.beta_parameters)
    # end experiment 
    experiments[e,:]= np.cumsum(np.array(opt_reward_item1[:observation]) + np.array(opt_reward_item2[:observation])) - np.cumsum(np.array(ts_reward_item1[:observation]) + np.array(ts_reward_item2[:observation]))
    experimets_item1_regret_curve[e,:]= np.cumsum(opt_reward_item1[:observation]) - np.cumsum(ts_reward_item1[:observation])
    experimets_item2_regret_curve[e,:]= np.cumsum(opt_reward_item2[:observation]) - np.cumsum(ts_reward_item2[:observation])
    
    #-------------------------MATCHING------------------------------------------------------------------
    #Retrieve the best prices from the ts_learner
    item1_price_full = candidates_item1[np.argmax(ts_learner_item1.beta_parameters)//2]
    item2_price_full = candidates_item2[np.argmax(ts_learner_item2.beta_parameters)//2]
    #discount for the second item 
    discounted_price = [item2_price_full,
        item2_price_full*(1-ctx.discount_promos[1]),
        item2_price_full*(1-ctx.discount_promos[2]),
        item2_price_full*(1-ctx.discount_promos[3])]
    #print(discounted_price)
    conversion_rate_second = np.zeros((4,4))
    priced_conversion_rate_second = np.zeros((4,4))
    for i in range (0,4): #classes
        for j in range (0,4): #promos
            conversion_rate_second[i,j] = (ctx.conversion_rate_second_element(discounted_price[j], i))
            priced_conversion_rate_second[i,j] = conversion_rate_second[i,j] * discounted_price[j]*ctx.customersDistribution[i,0]
            
    opt = linear_sum_assignment(priced_conversion_rate_second, maximize=True) # optimal solution row_ind, col_ind
    
    period_UCB_reward = [] # rewards collected in a period (days) performing the online learning strategy
    period_opt_reward = [] # rewards collected in a period (days) performing the online learning strategy

    learner = UCB_Matching(conversion_rate_second.size, *conversion_rate_second.shape) # Initialize UCB matching learner
    max_rew=[0,0,0,0]
    delay = 28
    max_reward_pumping = 1.1
    decimal_digits = 2
    for t in range(days_matching): # Day simulation
        #4. Query the learner to know wath is the best matching strategy category-promotion 

        sub_matching = learner.pull_arm() # suboptimal matching. row_ind, col_ind

        # 1. Generate daily customers according the Context distributions, divided in categories
        rewards_to_update=[0.,0.,0.,0.]

        daily_customer = ctx.customers_daily_instance()
        daily_customer_weight=daily_customer.copy()
        cum_UCB_rewards = 0
        cum_opt_rewards = 0
        category=0
        tot_client=sum(daily_customer)
        for customer in range(tot_client): # for each category emulate the user that purchase the good 
            category = np.random.choice(np.nonzero(daily_customer)[0])
            daily_customer[category] -= 1
            #2. Purchase simulation of the first element. (no optimization strategy)
            buy_item1 = ctx.purchase_online_first_element(item1_price_full,category) 
            cum_UCB_rewards += buy_item1*item1_price_full
            cum_opt_rewards += buy_item1*item1_price_full

            #3. Propose the second item only if the first one was bought
            if (buy_item1 > 0):
                #5. Propose the second item to the user, using the promotion that retrieved by the learner (according to the user category)                    
                buy_item2 = ctx.purchase_online_second_element(discounted_price[sub_matching[1][category]],category) # 0: not purchased, 1: purchased
                buy_item2_opt = ctx.purchase_online_second_element(discounted_price[opt[1][category]],category)
                #6. update the learner according to the obtained reward. rewards_to_update is a 4-zeros array, except for the element representing the current user category that contain the obtained reward
                rewards_to_update[category] += buy_item2 * discounted_price[sub_matching[1][category]]

                # store results in the cumulative daily rewards 
                cum_UCB_rewards += (buy_item2 * discounted_price[sub_matching[1][category]])
                cum_opt_rewards += (buy_item2_opt * discounted_price[opt[1][category]]) # purchase of the second item according to the optimal strategy 
        
        if(t<delay):
            rewards=[0,0,0,0]
            max_rew[0]=max(rewards_to_update[0]/daily_customer_weight[0],max_rew[0])
            max_rew[1]=max(rewards_to_update[1]/daily_customer_weight[1],max_rew[1])
            max_rew[2]=max(rewards_to_update[2]/daily_customer_weight[2],max_rew[2])
            max_rew[3]=max(rewards_to_update[3]/daily_customer_weight[3],max_rew[3])
        else:
            rewards[0]=round(rewards_to_update[0]/(daily_customer_weight[0]*max_rew[0]*1.02),decimal_digits)
            rewards[1]=round(rewards_to_update[1]/(daily_customer_weight[1]*max_rew[1]*1.02),decimal_digits)
            rewards[2]=round(rewards_to_update[2]/(daily_customer_weight[2]*max_rew[2]*1.02),decimal_digits)
            rewards[3]=round(rewards_to_update[3]/(daily_customer_weight[3]*max_rew[3]*1.02),decimal_digits)
        
        print(rewards_to_update)
        print(rewards)
        print(sub_matching[1])
        print(opt[1])
        print(daily_customer_weight)
        learner.update(sub_matching,rewards)
        period_UCB_reward.append(cum_UCB_rewards)
        period_opt_reward.append(cum_opt_rewards)
        print('___________________')
        print("| Discounted price sotto: |")
        print(discounted_price)
        print(f'| Day: {t+1} - Experiment: {e+1}')
        print(f'| Today customers distribution : {daily_customer_weight}')
        print(f'| Today cumulative reward (Online strategy):  {cum_UCB_rewards}\n| Today cumulative reward (Optimal strategy): {cum_opt_rewards}\n| - Loss: {cum_opt_rewards - cum_UCB_rewards}')
        print(f'Current confidence per arm of the online learner:\n{learner.confidence}')
        print('___________________\n')
    experiments_matching[e,:] = np.cumsum(period_opt_reward) - np.cumsum(period_UCB_reward)





# plot regret of UCB

plt.figure(1)
plt.plot(experiments_matching.mean(axis=0))
plt.ylabel('Regret')
plt.xlabel('t')
plt.title("Exp. Matching")
      
            
    
plt.figure(2)
plt.xlabel("#sales")
plt.ylabel("Regret")
plt.plot(np.mean(experiments,axis=0),'-', color='darkorange', label = 'Total regret')
plt.plot(np.mean(experimets_item1_regret_curve,axis=0),'-', color='blue', label = 'Item1 regret')
plt.plot(np.mean(experimets_item2_regret_curve,axis=0),'-', color='green', label = 'Item2 regret')
plt.title("Pricing")
plt.legend()


plt.show()