from Context import *    
import matplotlib.pyplot as plt
from Algorithms.UCB_Matching import *
from Algorithms.TS_Learner import *

ctx = Context()

days = 10 # 365 days of simulations

# define the prices candidates for the first and second item
candidates_item1 = [2260.0, 1910.0, 2130.0, 2010.0, 2340.0]
candidates_item2 = [500.0, 310.0, 350.0, 420.0, 550.0]
#discounted_price = ctx.discuonted_second_item_prices(promotion_assignment) # retrun the discounted prices for every customer category, according to the pormotion assignment

# find the optimal solutions 
optimal_conv_rate_item1 = np.zeros((4))
optimal_conv_rate_item2 = np.zeros((4))
optimal_candidates1 = np.zeros((4)).astype(int)
optimal_candidates2 = np.zeros((4)).astype(int)
for c in range(4):
    for i in range(len(candidates_item1)):
        if(optimal_conv_rate_item1[c] <= ctx.conversion_rate_first_element(candidates_item1[i],c) * candidates_item1[i]):
            optimal_conv_rate_item1[c] = ctx.conversion_rate_first_element(candidates_item1[i],c) * candidates_item1[i] 
            optimal_candidates1[c] = i
    for i in range(len(candidates_item2)):
        if(optimal_conv_rate_item2[c] <= ctx.conversion_rate_second_element(candidates_item2[i],c) * candidates_item2[i]):
            optimal_conv_rate_item2[c] = ctx.conversion_rate_second_element(candidates_item2[i],c) * candidates_item2[i]
            optimal_candidates2[c] = i

maximum_rewards_item1 = max(candidates_item1) + max(candidates_item2) # parameter used to normalize the reward
maximum_rewards_item2 = max(candidates_item2) # parameter used to normalize the reward

n_exp = 1
observation = (days//2)*1000
experiments = np.zeros((n_exp,observation))
experimets_item1_regret_curve = np.zeros((n_exp,observation))
experimets_item2_regret_curve = np.zeros((n_exp,observation))
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
            opt_buy_or_not_item1 = ctx.purchase_online_first_element(candidates_item1[optimal_candidates1[category]],category)
            # compute the rewenue of the first and second item for both optimal solution and the online learning
            if ts_buy_or_not_item1:
                ts_buy_or_not_item2 = ctx.purchase_online_second_element(candidates_item2[ts_pulled_arm_item2],category) 
                
                # calculate the reward
                customer_reward_item2 = candidates_item2[ts_pulled_arm_item2] * ts_buy_or_not_item2
                customer_reward_item1 = candidates_item1[ts_pulled_arm_item1]

            if (opt_buy_or_not_item1):
                opt_buy_or_not_item2 = ctx.purchase_online_second_element(candidates_item2[optimal_candidates2[category]],category)

                # calculate the reward
                opt_customer_item2 =  candidates_item2[optimal_candidates2[category]] * opt_buy_or_not_item2
                opt_customer_item1 = candidates_item1[optimal_candidates1[category]] 

            # update the learner normalizing the reward. The learner for the second item is updated only the customer buy the first one
            ts_learner_item1.update(ts_pulled_arm_item1, (customer_reward_item1 + customer_reward_item2 )/maximum_rewards_item1)
            if ts_buy_or_not_item1:
                ts_learner_item2.update(ts_pulled_arm_item2, customer_reward_item2/maximum_rewards_item2)

            print('___________________')
            print(f'| Day: {d+1} - Experiment {e+1}')
            print(f'| Today customers distribution : {daily_customer_weight}')
            print(f'| Customer #{customer} of category: {ctx.classes_info[category]["name"]}: ')
            print(f'|\t[TS] - Selected prices -> {ctx.items_info[0]["name"]} : {candidates_item1[ts_pulled_arm_item1]} €, {ctx.items_info[1]["name"]} : {candidates_item2[ts_pulled_arm_item2]} €\n|\t\t{ctx.items_info[0]["name"]} reward : {round(customer_reward_item1,2)} € -- {ctx.items_info[1]["name"]} reward : {round(customer_reward_item2,2)} € -- Total : {round(customer_reward_item1 + customer_reward_item2,2)} €')
            print(f'|\t[OPT] -  Selected prices -> {ctx.items_info[0]["name"]} : {candidates_item1[optimal_candidates1[category]]} €, {ctx.items_info[1]["name"]} : {candidates_item2[optimal_candidates2[category]]} €\n|\t\t{ctx.items_info[0]["name"]} reward : {round(opt_customer_item1,2)} € -- {ctx.items_info[1]["name"]} reward : {round(opt_customer_item2,2)} € -- Total : {round(opt_customer_item1 + opt_customer_item2,2)} €')

            ts_reward_item1.append(customer_reward_item1)
            ts_reward_item2.append(customer_reward_item2)
            opt_reward_item1.append(opt_customer_item1)
            opt_reward_item2.append(opt_customer_item2)
        
    # end experiment 
    experiments[e,:]= np.cumsum(np.array(opt_reward_item1[:observation]) + np.array(opt_reward_item2[:observation])) - np.cumsum(np.array(ts_reward_item1[:observation]) + np.array(ts_reward_item2[:observation]))
    experimets_item1_regret_curve[e,:]= np.cumsum(opt_reward_item1[:observation]) - np.cumsum(ts_reward_item1[:observation])
    experimets_item2_regret_curve[e,:]= np.cumsum(opt_reward_item2[:observation]) - np.cumsum(ts_reward_item2[:observation])

plt.figure(1)
plt.xlabel("#sales")
plt.ylabel("Regret")
plt.plot(np.mean(experiments,axis=0),'-', color='darkorange', label = 'Total regret')
plt.plot(np.mean(experimets_item1_regret_curve,axis=0),'-', color='blue', label = 'Item1 regret')
plt.plot(np.mean(experimets_item2_regret_curve,axis=0),'-', color='green', label = 'Item2 regret')
plt.title("Cumulative regret")
plt.legend()


plt.show()