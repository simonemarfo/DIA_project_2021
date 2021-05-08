from Context import *    
import matplotlib.pyplot as plt
from Algorithms.UCB_Matching import *
from Algorithms.TS_Learner import *

ctx = Context()

days = 20 # 365 days of simulations
days_matching = 140-days
item1_full_price=0.0
item2_full_price=0.0
# define the prices candidates for the first and second item
candidates_item1 = [2260.0, 1910.0, 2130.0, 2010.0, 2340.0]
candidates_item2 = [560.0, 530.0, 590.0, 620.0, 650.0]
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

n_exp = 10
observation = (days//2)*1000
observation_matching = (days_matching//2)*1000
experiments = np.zeros((n_exp,observation))
experiments_matching = np.zeros((n_exp,observation_matching))
experimets_item1_regret_curve = np.zeros((n_exp,observation))
experimets_item2_regret_curve = np.zeros((n_exp,observation))
days_experiments_matching = np.zeros((n_exp,days_matching))
for e in range(n_exp):
    ts_learner_item1 = TS_Learner(len(candidates_item1))
    ts_learner_item2 = TS_Learner(len(candidates_item2))

    opt_reward_item1 = []
    opt_reward_item2 = [] 
    ts_reward_item1 = []
    ts_reward_item2 = []    
    permutation = list(permutations(range(0,4)))
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
    period_UCB_reward = [] # rewards collected in a period (days) performing the online learning strategy
    period_opt_reward = [] # rewards collected in a period (days) performing the online learning strategy
    tot_rew = np.zeros((4,4))
    support = np.zeros((4,4))
    item1_price_full = candidates_item1[np.argmax(ts_learner_item1.beta_parameters)//2]
    item2_price_full = candidates_item2[np.argmax(ts_learner_item2.beta_parameters)//2]
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

    day_UCB_reward = [] 
    day_opt_reward = []

    learner = UCB_Matching(conversion_rate_second.size, *conversion_rate_second.shape) # Initialize UCB matching learner
    prev_size = 0
    for d in range(days_matching): # Day simulation
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
        n_cli= 0
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
                if d<3:
                    row_ind = list(range(0,4))
                    col_ind = permutation[n_cli % 24]
                    sub_matching = [row_ind,col_ind]
                    n_cli+=1
                else:
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
                if d<3:
                    #learner.update(sub_matching,[0,0,0,0])
                    pass
                else:
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
    experiments_matching[e,:] = np.cumsum(period_opt_reward[:observation_matching]) - np.cumsum(period_UCB_reward[:observation_matching])
    days_experiments_matching[e,:] = np.cumsum(day_opt_reward) - np.cumsum(day_UCB_reward)

# plot daily reward comparison
mean_UCB_reward = np.mean(period_UCB_reward)
mean_opt_reward = np.mean(period_opt_reward)
print(f"Mean daily reward using online UCB strategy: {mean_UCB_reward}")
print(f"Mean daily reward using optimal strategy: {mean_opt_reward}")
print(f'Period ({days_matching} days) regret: {np.sum(period_opt_reward) - np.sum(period_UCB_reward)}')

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
plt.plot(experiments_matching.mean(axis=0))
plt.ylabel('Regret')
plt.xlabel('Client#')


# plot regret of UCB

plt.figure(2)
plt.plot(days_experiments_matching.mean(axis=0))
plt.ylabel('Regret')
plt.xlabel('Days')


            
    
plt.figure(3)
plt.xlabel("#sales")
plt.ylabel("Regret")
plt.plot(np.mean(experiments,axis=0),'-', color='darkorange', label = 'Total regret')
plt.plot(np.mean(experimets_item1_regret_curve,axis=0),'-', color='blue', label = 'Item1 regret')
plt.plot(np.mean(experimets_item2_regret_curve,axis=0),'-', color='green', label = 'Item2 regret')
plt.title("Pricing")
plt.legend()


plt.show()