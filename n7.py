from Context import *    
import matplotlib.pyplot as plt
from Algorithms.UCB_Matching import *
from Algorithms.SWTS_Learner import *
from itertools import permutations

ctx = Context()

days = 120 # 365 days of simulations

# define the prices candidates for the first and second item
candidates_item1 = [2110.0, 1900.0, 2130.0, 1920.0, 2340.0]
candidates_item2 = [365.0, 400.0, 550.0, 510.0, 380.0]
# TODO parametrico
window_size1 = int(np.sqrt(days*1000)*30)
window_size2 = int(np.sqrt(days*600)*30) # TODO: aggiungere un metodo nel learner per fare un resize della window. Calcolare la window per item due in  runtime, con la media delle proposed item (?) 

# find the optimal solutions
opt_item1=np.zeros((3),dtype=int)
opt_item2=np.zeros((3),dtype=int)
for season in range (3): 
    opt_rew_item1 = np.zeros((5))
    opt_rew_item2 = np.zeros((5))

    for i in range(len(candidates_item1)):
        for c in range(4):
                opt_rew_item1[i] += ctx.conversion_rate_first_element(candidates_item1[i],c,season) * candidates_item1[i] * ctx.customersDistribution[c,0]

    for i in range(len(candidates_item2)):
        for c in range(4):
            opt_rew_item2[i] += ctx.conversion_rate_second_element(candidates_item2[i],c,season) * candidates_item2[i] * ctx.customersDistribution[c,0]

    opt_item1[season] = int(np.argmax(opt_rew_item1))
    opt_item2[season] = int(np.argmax(opt_rew_item2))

print(opt_item1)
print(opt_item2)

maximum_rewards_item1 = max(candidates_item1) + max(candidates_item2) # parameter used to normalize the reward
maximum_rewards_item2 = max(candidates_item2) # parameter used to normalize the reward

n_exp = 1
observation = int((days)*1000*0.9) # observe only 90% of clients 
SWTS_total_experiments = np.zeros((n_exp,observation)) # Sliding window Thompson sampling
SWTS_experimets_item1_regret_curve = np.zeros((n_exp,observation))
SWTS_experimets_item2_regret_curve = np.zeros((n_exp,observation)) 
TS_total_experiments = np.zeros((n_exp,observation)) #Thompson sampling
TS_experimets_item1_regret_curve = np.zeros((n_exp,observation))
TS_experimets_item2_regret_curve = np.zeros((n_exp,observation))
days_SWTS_total_experiments = np.zeros((n_exp,days)) # days experiments plot
days_SWTS_item1_experiments = np.zeros((n_exp,days))
days_SWTS_item2_experiments = np.zeros((n_exp,days))
days_TS_total_experiments = np.zeros((n_exp,days))
days_TS_item1_experiments = np.zeros((n_exp,days))
days_TS_item2_experiments = np.zeros((n_exp,days))


matching_total_experiments = np.zeros((n_exp,observation)) # matching
matching_experimets_item1_regret_curve = np.zeros((n_exp,observation))
matching_experimets_item2_regret_curve = np.zeros((n_exp,observation))
days_matching_total_experiments = np.zeros((n_exp,days))
days_matching_item1_experiments = np.zeros((n_exp,days))
days_matching_item2_experiments = np.zeros((n_exp,days))


for e in range(n_exp):
    SWTS_learner_item1 = SWTS_Learner(len(candidates_item1),window_size1)
    SWTS_learner_item2 = SWTS_Learner(len(candidates_item2),window_size2)
    TS_learner_item1 = TS_Learner(len(candidates_item1))
    TS_learner_item2 = TS_Learner(len(candidates_item2))

    UCB_matching_learner = UCB_Matching(np.zeros((4,4)).size, *np.zeros((4,4)).shape) # Initialize UCB matching learner

    opt_reward_item1 = []
    opt_reward_item2 = [] 
    swts_reward_item1 = []
    swts_reward_item2 = []   
    ts_reward_item1 = []
    ts_reward_item2 = []  
    
    daily_opt_reward_item1 = []
    daily_opt_reward_item2 = [] 
    daily_swts_reward_item1 = []
    daily_swts_reward_item2 = []   
    daily_ts_reward_item1 = []
    daily_ts_reward_item2 = []
    daily_opt_item1_ptr = 0
    daily_opt_item2_ptr = 0
    daily_swts_item1_ptr = 0
    daily_swts_item2_ptr = 0
    daily_ts_item1_ptr = 0
    daily_ts_item2_ptr = 0

    #matching
    matching_reward_item1 = []
    matching_reward_item2 = [] 
    daily_matching_reward_item1 = []
    daily_matching_reward_item2 = []
    daily_matching_item1_ptr = 0
    daily_matching_item2_ptr = 0
    matching_opt_reward_item1 = []
    matching_opt_reward_item2 = [] 
    daily_matching_opt_reward_item1 = []
    daily_matching_opt_reward_item2 = []
    daily_matching_opt_item1_ptr = 0
    daily_matching_opt_item2_ptr = 0

    # matching occurrency matrix 
    tot_rew = np.zeros((4,4))
    support = np.zeros((4,4))
    day_matching_counter = 0 
    matching = False
    permutation = list(permutations(range(0,4)))
    item2_fixed_price = 0 
    for d in range(days):
        season = int(d//((days + 1)//3))
        n_cli= 0 # used to initialize matching phase
        # extract the daily customer. It is UNKNOWN
        customer_per_class = ctx.customers_daily_instance() 
        daily_customer_weight = customer_per_class.copy()
        # switch matching
        if d - (season * (days//3)) == 30:
            tot_rew = np.zeros((4,4))
            support = np.zeros((4,4))
            matching = True
            item2_fixed_price = candidates_item2[np.argmax(SWTS_learner_item2.beta_parameters)//2]
            # calculating optimal matching
            discounted_price = [item2_fixed_price,
                item2_fixed_price*(1-ctx.discount_promos[1]),
                item2_fixed_price*(1-ctx.discount_promos[2]),
                item2_fixed_price*(1-ctx.discount_promos[3])]
            priced_conversion_rate_second = np.zeros((4,4))
            for i in range (0,4): #classes
                for j in range (0,4): #promos
                    priced_conversion_rate_second[i,j] = (ctx.conversion_rate_second_element(discounted_price[j], i)) * discounted_price[j]
            matching_opt = linear_sum_assignment(priced_conversion_rate_second, maximize=True) # optimal solution row_ind, col_ind

        if d - (season * (days //3)) == 0:
            matching = False
            day_matching_counter = 0 

        tot_client = sum(customer_per_class)
        # simulate the day client by client
        for customer in range(tot_client):
            swts_customer_reward_item1 = 0.0
            swts_customer_reward_item2 = 0.0
            ts_customer_reward_item1 = 0.0
            ts_customer_reward_item2 = 0.0
            opt_customer_item1 = 0.0 # opt reward
            opt_customer_item2 = 0.0 # opt reward

            matching_customer_reward_item1 = 0.0
            matching_customer_reward_item2 = 0.0
            matching_opt_customer_item1 = 0.0 
            matching_opt_customer_item2 = 0.0 

            category = np.random.choice(np.nonzero(customer_per_class)[0])
            customer_per_class[category] -= 1

            # ask to the learner to pull the most promising price that maximize the reward
            swts_pulled_arm_item1 = SWTS_learner_item1.pull_arm()
            swts_pulled_arm_item2 = SWTS_learner_item2.pull_arm() #select candidates for the second item

            ts_pulled_arm_item1 = TS_learner_item1.pull_arm()
            ts_pulled_arm_item2 = TS_learner_item2.pull_arm() 

            swts_buy_or_not_item1 = ctx.purchase_online_first_element(candidates_item1[swts_pulled_arm_item1],category,season) 
            ts_buy_or_not_item1 = ctx.purchase_online_first_element(candidates_item1[ts_pulled_arm_item1],category,season) 
            opt_buy_or_not_item1 = ctx.purchase_online_first_element(candidates_item1[opt_item1[season]],category,season)
            # compute the rewenue of the first and second item for both optimal solution and the online learning
            if swts_buy_or_not_item1:                            
                swts_buy_or_not_item2 = ctx.purchase_online_second_element(candidates_item2[swts_pulled_arm_item2],category,season) 
                # calculate the reward
                swts_customer_reward_item2 = candidates_item2[swts_pulled_arm_item2] * swts_buy_or_not_item2
                swts_customer_reward_item1 = candidates_item1[swts_pulled_arm_item1]
                if not matching:
                    matching_customer_reward_item2 = swts_customer_reward_item2
                    matching_customer_reward_item1 = swts_customer_reward_item1

            if ts_buy_or_not_item1:
                ts_buy_or_not_item2 = ctx.purchase_online_second_element(candidates_item2[ts_pulled_arm_item2],category,season) 
                # calculate the reward
                ts_customer_reward_item2 = candidates_item2[ts_pulled_arm_item2] * ts_buy_or_not_item2
                ts_customer_reward_item1 = candidates_item1[ts_pulled_arm_item1]

            if opt_buy_or_not_item1:
                opt_buy_or_not_item2 = ctx.purchase_online_second_element(candidates_item2[opt_item2[season]],category,season)
                # calculate the reward
                opt_customer_item2 =  candidates_item2[opt_item2[season]] * opt_buy_or_not_item2
                opt_customer_item1 = candidates_item1[opt_item1[season]] 
                if not matching:
                    matching_opt_customer_item1 = opt_customer_item1
                    matching_opt_customer_item2 = opt_customer_item2

            #  -------------------- MATCHING --------------
            if opt_buy_or_not_item1 and matching: 
                matching_opt_customer_item1 =  candidates_item1[opt_item1[season]] 
                matching_opt_customer_item2 =  ctx.purchase_online_second_element(discounted_price[matching_opt[1][category]],category) * discounted_price[matching_opt[1][category]]

                
            if swts_buy_or_not_item1 and matching:
                if day_matching_counter < 3:
                    row_ind = list(range(0,4))
                    col_ind = permutation[n_cli % 24]
                    sub_matching = [row_ind,col_ind]
                    n_cli+=1
                else:
                    sub_matching = UCB_matching_learner.pull_arm() # suboptimal matching. row_ind, col_ind

                propose_price =  discounted_price[sub_matching[1][category]]
                UCB_buy_or_not_item2 = ctx.purchase_online_second_element(propose_price,category,season) 
                # calculate the reward
                matching_customer_reward_item2 = propose_price * UCB_buy_or_not_item2
                matching_customer_reward_item1 = candidates_item1[swts_pulled_arm_item1]

                support[category][sub_matching[1][category]] += 1
                tot_rew[category][sub_matching[1][category]] += matching_customer_reward_item2
                update_array = np.zeros((4))
                for c in range(4):
                    if support[c][sub_matching[1][c]] == 0:
                        pass
                    else:
                        update_array[c] = tot_rew[c][sub_matching[1][c]] / (support[c][sub_matching[1][c]] * item2_fixed_price) 
                #learner update
                if day_matching_counter>3:
                    UCB_matching_learner.update(sub_matching,update_array)
                day_matching_counter += 1

            # update the learner normalizing the reward. The learner for the second item is updated only the customer buy the first one
            SWTS_learner_item1.update(swts_pulled_arm_item1, (swts_customer_reward_item1 + swts_customer_reward_item2 )/maximum_rewards_item1)
            TS_learner_item1.update(ts_pulled_arm_item1, (ts_customer_reward_item1 + ts_customer_reward_item2 )/maximum_rewards_item1)
            if swts_buy_or_not_item1:
                    SWTS_learner_item2.update(swts_pulled_arm_item2, (swts_customer_reward_item2)/maximum_rewards_item2)
            if ts_buy_or_not_item1:
                TS_learner_item2.update(ts_pulled_arm_item2, (ts_customer_reward_item2)/maximum_rewards_item2)

            print('___________________')
            print(f'| Day: {d+1} - Experiment {e+1}')
            print(f'| Today customers distribution : {daily_customer_weight}')
            print(f'| Customer #{customer} of category: {ctx.classes_info[category]["name"]}: ')
            print(f'|\t[SWTS] - Selected prices -> {ctx.items_info[0]["name"]} : {candidates_item1[swts_pulled_arm_item1]} €, {ctx.items_info[1]["name"]} : {candidates_item2[swts_pulled_arm_item2]} €\n|\t\t{ctx.items_info[0]["name"]} reward : {round(swts_customer_reward_item1,2)} € -- {ctx.items_info[1]["name"]} reward : {round(swts_customer_reward_item2,2)} € -- Total : {round(swts_customer_reward_item1 + swts_customer_reward_item2,2)} €')
            print(f'|\t[TS] - Selected prices -> {ctx.items_info[0]["name"]} : {candidates_item1[ts_pulled_arm_item1]} €, {ctx.items_info[1]["name"]} : {candidates_item2[ts_pulled_arm_item2]} €\n|\t\t{ctx.items_info[0]["name"]} reward : {round(ts_customer_reward_item1,2)} € -- {ctx.items_info[1]["name"]} reward : {round(ts_customer_reward_item2,2)} € -- Total : {round(ts_customer_reward_item1 + ts_customer_reward_item2,2)} €')
            print(f'|\t[OPT] -  Selected prices -> {ctx.items_info[0]["name"]} : {candidates_item1[opt_item1[season]]} €, {ctx.items_info[1]["name"]} : {candidates_item2[opt_item2[season]]} €\n|\t\t{ctx.items_info[0]["name"]} reward : {round(opt_customer_item1,2)} € -- {ctx.items_info[1]["name"]} reward : {round(opt_customer_item2,2)} € -- Total : {round(opt_customer_item1 + opt_customer_item2,2)} €')

            swts_reward_item1.append(swts_customer_reward_item1)
            swts_reward_item2.append(swts_customer_reward_item2)
            ts_reward_item1.append(ts_customer_reward_item1)
            ts_reward_item2.append(ts_customer_reward_item2)
            opt_reward_item1.append(opt_customer_item1)
            opt_reward_item2.append(opt_customer_item2)

            matching_reward_item1.append(matching_customer_reward_item1)
            matching_reward_item2.append(matching_customer_reward_item2)
            matching_opt_reward_item1.append(matching_opt_customer_item1)
            matching_opt_reward_item2.append(matching_opt_customer_item2)
        #daily append
        daily_opt_reward_item1.append(sum(opt_reward_item1[daily_opt_item1_ptr:]))
        daily_opt_reward_item2.append(sum(opt_reward_item2[daily_opt_item2_ptr:]))
        daily_opt_item1_ptr = len(opt_reward_item1)
        daily_opt_item2_ptr = len(opt_reward_item2)
        daily_swts_reward_item1.append(sum(swts_reward_item1[daily_swts_item1_ptr:]))
        daily_swts_reward_item2.append(sum(swts_reward_item2[daily_swts_item2_ptr:]))
        daily_swts_item1_ptr = len(swts_reward_item1)
        daily_swts_item2_ptr = len(swts_reward_item2)
        daily_ts_reward_item1.append(sum(ts_reward_item1[daily_ts_item1_ptr:]))
        daily_ts_reward_item2.append(sum(ts_reward_item2[daily_ts_item2_ptr:]))
        daily_ts_item1_ptr = len(ts_reward_item1)
        daily_ts_item2_ptr = len(ts_reward_item2)

        daily_matching_reward_item1.append(sum(matching_reward_item1[daily_matching_item1_ptr:]))
        daily_matching_reward_item2.append(sum(matching_reward_item2[daily_matching_item2_ptr:]))
        daily_matching_item1_ptr = len(matching_reward_item1)
        daily_matching_item2_ptr = len(matching_reward_item2)
        daily_matching_opt_reward_item1.append(sum(matching_opt_reward_item1[daily_matching_opt_item1_ptr:]))
        daily_matching_opt_reward_item2.append(sum(matching_opt_reward_item2[daily_matching_opt_item2_ptr:]))
        daily_matching_opt_item1_ptr = len(matching_opt_reward_item1)
        daily_matching_opt_item2_ptr = len(matching_opt_reward_item2)
        
    # end experiment 
    SWTS_total_experiments[e,:]= np.cumsum(np.array(opt_reward_item1[:observation]) + np.array(opt_reward_item2[:observation])) - np.cumsum(np.array(swts_reward_item1[:observation]) + np.array(swts_reward_item2[:observation]))
    SWTS_experimets_item1_regret_curve[e,:]= np.cumsum(opt_reward_item1[:observation]) - np.cumsum(swts_reward_item1[:observation])
    SWTS_experimets_item2_regret_curve[e,:]= np.cumsum(opt_reward_item2[:observation]) - np.cumsum(swts_reward_item2[:observation])

    TS_total_experiments[e,:]= np.cumsum(np.array(opt_reward_item1[:observation]) + np.array(opt_reward_item2[:observation])) - np.cumsum(np.array(ts_reward_item1[:observation]) + np.array(ts_reward_item2[:observation]))
    TS_experimets_item1_regret_curve[e,:]= np.cumsum(opt_reward_item1[:observation]) - np.cumsum(ts_reward_item1[:observation])
    TS_experimets_item2_regret_curve[e,:]= np.cumsum(opt_reward_item2[:observation]) - np.cumsum(ts_reward_item2[:observation])

    matching_total_experiments[e,:]= np.cumsum(np.array(matching_opt_reward_item1[:observation]) + np.array(matching_opt_reward_item2[:observation])) - np.cumsum(np.array(matching_reward_item1[:observation]) + np.array(matching_reward_item2[:observation]))
    matching_experimets_item1_regret_curve[e,:]= np.cumsum(matching_opt_reward_item1[:observation]) - np.cumsum(matching_reward_item1[:observation])
    matching_experimets_item2_regret_curve[e,:]= np.cumsum(matching_opt_reward_item2[:observation]) - np.cumsum(matching_reward_item2[:observation])

    days_SWTS_total_experiments[e:] = np.cumsum(np.add(daily_opt_reward_item1, daily_opt_reward_item2)) - np.cumsum(np.add(daily_swts_reward_item1, daily_swts_reward_item2))
    days_SWTS_item1_experiments[e:] = np.cumsum(daily_opt_reward_item1) - np.cumsum(daily_swts_reward_item1)
    days_SWTS_item2_experiments[e:] = np.cumsum(daily_opt_reward_item2) - np.cumsum(daily_swts_reward_item2)

    days_TS_total_experiments[e:] = np.cumsum(np.add(daily_opt_reward_item1, daily_opt_reward_item2)) - np.cumsum(np.add(daily_ts_reward_item1, daily_ts_reward_item2))
    days_TS_item1_experiments[e:] = np.cumsum(daily_opt_reward_item1) - np.cumsum(daily_ts_reward_item1)
    days_TS_item2_experiments[e:] = np.cumsum(daily_opt_reward_item2) - np.cumsum(daily_ts_reward_item2)

    days_matching_total_experiments[e:] = np.cumsum(np.add(daily_matching_opt_reward_item1, daily_matching_opt_reward_item2)) - np.cumsum(np.add(daily_matching_reward_item1, daily_matching_reward_item2))
    days_matching_item1_experiments[e:] = np.cumsum(daily_matching_opt_reward_item1) - np.cumsum(daily_matching_reward_item1)
    days_matching_item2_experiments[e:] = np.cumsum(daily_matching_opt_reward_item2) - np.cumsum(daily_matching_reward_item2)

# plot regret
plt.figure(1)
plt.xlabel("Days")
plt.ylabel("Regret")
plt.plot(np.mean(days_SWTS_total_experiments,axis=0),'-', color='darkorange', label = 'SWTS - Total regret')
plt.plot(np.mean(days_SWTS_item1_experiments,axis=0),'-', color='blue', label = 'SWTS - Item1 regret')
plt.plot(np.mean(days_SWTS_item2_experiments,axis=0),'-', color='green', label = 'SWTS - Item2 regret')
plt.plot(np.mean(days_TS_total_experiments,axis=0),'-', color='orange', label = 'TS - Total regret')
plt.plot(np.mean(days_TS_item1_experiments,axis=0),'-', color='cornflowerblue', label = 'TS - Item1 regret')
plt.plot(np.mean(days_TS_item2_experiments,axis=0),'-', color='limegreen', label = 'TS - Item2 regret')
plt.plot(np.mean(days_matching_total_experiments,axis=0),'-', color='purple', label = 'UCB Matching - Total regret')
plt.plot(np.mean(days_matching_item1_experiments,axis=0),'-', color='magenta', label = 'UCB Matching - Item1 regret')
plt.plot(np.mean(days_matching_item2_experiments,axis=0),'-', color='deeppink', label = 'UCB Matching - Item2 regret')
plt.title("Pricing")
plt.legend()
      

plt.figure(2)
plt.xlabel("#sales")
plt.ylabel("Regret")
plt.plot(np.mean(SWTS_total_experiments,axis=0),'-', color='darkorange', label = 'SWTS - Total regret')
plt.plot(np.mean(SWTS_experimets_item1_regret_curve,axis=0),'-', color='blue', label = 'SWTS - Item1 regret')
plt.plot(np.mean(SWTS_experimets_item2_regret_curve,axis=0),'-', color='green', label = 'SWTS - Item2 regret')
plt.plot(np.mean(TS_total_experiments,axis=0),'-', color='orange', label = 'TS - Total regret')
plt.plot(np.mean(TS_experimets_item1_regret_curve,axis=0),'-', color='cornflowerblue', label = 'TS - Item1 regret')
plt.plot(np.mean(TS_experimets_item2_regret_curve,axis=0),'-', color='limegreen', label = 'TS - Item2 regret')
plt.plot(np.mean(matching_total_experiments,axis=0),'-', color='purple', label = 'UCB Matching - Total regret')
plt.plot(np.mean(matching_experimets_item1_regret_curve,axis=0),'-', color='magenta', label = 'UCB Matching - Item1 regret')
plt.plot(np.mean(matching_experimets_item2_regret_curve,axis=0),'-', color='deeppink', label = 'UCB Matching - Item2 regret')
plt.title("Pricing")
plt.legend()


plt.show()