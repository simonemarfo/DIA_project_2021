from Context import *    
import matplotlib.pyplot as plt
from Algorithms.promo_category_UCB_learner import *
from Algorithms.TS_Learner import *
from Algorithms.SWTS_Learner import *

#colors
import os
os.system("")

ctx = Context()
days = 365 # 365 days of simulations
n_exp = 1
seasonality = [0*(days//3), 1*(days//3), 2*(days//3)] # days at which the new season start
window_size = int(np.sqrt(days*1000) * 80)
season = 0

# define the prices candidates for the first and second item
candidates_item1 = [2110.0, 1900.0, 2420.0, 2690.0]
candidates_item2 = [360.0, 410.0, 530.0, 600.0]

# retrieve optimal solution for the seasoson with this candidates
opt_prices,opt_matching, best_daily_reward = ctx.correlated_optimal_solution(candidates_item1,candidates_item2,season=0) # return  best_prices[p1,p2],best_matching, best_reward
opt_price_item1 = opt_prices[0]
opt_price_item2 = opt_prices[1]

v_swts_experimets = np.zeros((n_exp,days))
v_ts_experimets = np.zeros((n_exp,days))
for e in range(n_exp):
    # LEARNERS
    swts_learner = SWTS_Learner(len(candidates_item1) * len(candidates_item2), window_size)
    ts_learner = TS_Learner(len(candidates_item1) * len(candidates_item2)) # superarm of couple price_item1, price_item2: <p1,p2>
    normalizing_value = max(candidates_item1) + max(candidates_item2) # value used to normalize the customer reward, used to update the learner
    # UCB Matching learner, one learner for each couple <p1,p2>
    matching_swts_learners = [promo_category_UCB_learner(np.zeros((4,4)).size, *np.zeros((4,4)).shape, 1000 ,max(candidates_item2)) for _ in range(len(candidates_item1) * len(candidates_item2))]
    matching_ts_learners = [promo_category_UCB_learner(np.zeros((4,4)).size, *np.zeros((4,4)).shape, 1000 ,max(candidates_item2)) for _ in range(len(candidates_item1) * len(candidates_item2))]
    v_daily_swts_reward = []
    v_daily_ts_reward = []
    v_daily_opt_reward = []

    for d in range(days):
        # extract the daily customer. It is UNKNOWN
        customer_per_class = ctx.customers_daily_instance() 
        daily_customer_weight = customer_per_class.copy()
        tot_client = sum(customer_per_class)
        daily_swts_reward = 0.0
        daily_ts_reward = 0.0
        daily_opt_reward = 0.0
        if d in seasonality: # new season begin, reset the matching_learner
            season = seasonality.index(d)
            #matching_swts_learners = [promo_category_UCB_learner(np.zeros((4,4)).size, *np.zeros((4,4)).shape, 1000 ,max(candidates_item2)) for _ in range(len(candidates_item1) * len(candidates_item2))]
            #matching_ts_learners = [promo_category_UCB_learner(np.zeros((4,4)).size, *np.zeros((4,4)).shape, 1000 ,max(candidates_item2)) for _ in range(len(candidates_item1) * len(candidates_item2))]
            # retrieve optimal solution for the seasoson with this candidates
            opt_prices,opt_matching, best_daily_reward = ctx.correlated_optimal_solution(candidates_item1,candidates_item2,season=season) # return  best_prices[p1,p2],best_matching, best_reward
            opt_price_item1 = opt_prices[0]
            opt_price_item2 = opt_prices[1]
        
        # simulate the day client by client
        for customer in range(tot_client):
            cus_swts_reward_item1 = 0.0
            cus_swts_reward_item2 = 0.0
            cus_ts_reward_item1 = 0.0
            cus_ts_reward_item2 = 0.0
            opt_customer_reward_item1 = 0.0 # opt reward
            opt_customer_reward_item2 = 0.0 # opt reward

            category = np.random.choice(np.nonzero(customer_per_class)[0])
            customer_per_class[category] -= 1

            # ask to the learner to pull the most promising couple <p1,p2> that maximize the reward
            # SWTS
            swts_pulled_arm = swts_learner.pull_arm() # number between 0..24
            cus_swts_price_item1 = candidates_item1[swts_pulled_arm // len(candidates_item1)]
            cus_swts_price_item2 = candidates_item2[swts_pulled_arm % len(candidates_item2)]
            # TS
            ts_pulled_arm = ts_learner.pull_arm() # number between 0..24
            cus_ts_price_item1 = candidates_item1[ts_pulled_arm // len(candidates_item1)]
            cus_ts_price_item2 = candidates_item2[ts_pulled_arm % len(candidates_item2)]
            # query the corresponding superarm learner and compute the discounted price 
            # SWTS
            sub_swts_matching = matching_swts_learners[swts_pulled_arm].pull_arm() # suboptimal matching. row_ind, col_ind
            cus_swts_price_item2_discounted = cus_swts_price_item2 * (1-ctx.discount_promos[ sub_swts_matching[1][category] ])
            # TS
            sub_ts_matching = matching_ts_learners[ts_pulled_arm].pull_arm() # suboptimal matching. row_ind, col_ind
            cus_ts_price_item2_discounted = cus_ts_price_item2 * (1-ctx.discount_promos[ sub_ts_matching[1][category] ])
            # OPT
            opt_price_item2_discounted = opt_price_item2 * (1-ctx.discount_promos[ opt_matching[1][category] ])

            # purchase simulations
            cus_swts_buy_or_not_item1 = ctx.purchase_online_first_element(cus_swts_price_item1,category,season) 
            cus_ts_buy_or_not_item1 = ctx.purchase_online_first_element(cus_ts_price_item1,category,season)
            opt_buy_or_not_item1 = ctx.purchase_online_first_element(opt_price_item1,category,season)
            cus_swts_buy_or_not_item2 = 0
            cus_ts_buy_or_not_item2 = 0
            opt_buy_or_not_item2 = 0

            # compute the rewenue of the first and second item for both optimal solution and the online learning
            if cus_swts_buy_or_not_item1:
                cus_swts_buy_or_not_item2 = ctx.purchase_online_second_element(cus_swts_price_item2_discounted, category,season) 
            if cus_ts_buy_or_not_item1:
                cus_ts_buy_or_not_item2 = ctx.purchase_online_second_element(cus_ts_price_item2_discounted, category,season)
            if opt_buy_or_not_item1:
                opt_buy_or_not_item2 = ctx.purchase_online_second_element(opt_price_item2_discounted, category,season)
            
            # computing rewards
            cus_swts_reward_item1 = cus_swts_buy_or_not_item1 * cus_swts_price_item1
            cus_swts_reward_item2 = cus_swts_buy_or_not_item2 * cus_swts_price_item2_discounted
            cus_ts_reward_item1 = cus_ts_buy_or_not_item1 * cus_ts_price_item1
            cus_ts_reward_item2 = cus_ts_buy_or_not_item2 * cus_ts_price_item2_discounted
            opt_customer_reward_item1 = opt_buy_or_not_item1 * opt_price_item1
            opt_customer_reward_item2 = opt_buy_or_not_item2 * opt_price_item2_discounted

            # update learners
            swts_learner.update(swts_pulled_arm, (cus_swts_reward_item1 + cus_swts_reward_item2 )/normalizing_value)
            ts_learner.update(ts_pulled_arm, (cus_ts_reward_item1 + cus_ts_reward_item2 )/normalizing_value)

            matching_swts_learners[swts_pulled_arm].update(sub_swts_matching, cus_swts_reward_item2, category=category)
            matching_ts_learners[ts_pulled_arm].update(sub_ts_matching, cus_ts_reward_item2, category=category)

            print('___________________')
            print(f'| Day: {d+1} - Experiment {e+1}')
            print(f'| Today customers distribution : {daily_customer_weight}')
            print(f'| Customer #{customer} of category: {ctx.classes_info[category]["name"]}: ')
            print(f'| {cus_swts_price_item1 = } --- {cus_swts_price_item2 = }')
            print(f'| {cus_ts_price_item1 = } --- {cus_ts_price_item2 = }')
            print(f'| {opt_price_item1 = } --- {opt_price_item2 = }')
            if np.array_equal(sub_swts_matching,opt_matching) :
                print(f'/ <swts matching> : \x1b[6;30;42m{sub_swts_matching}\x1b[0m --> {round(cus_swts_price_item2_discounted,2) = }')
            else:
                print(f'/ <swts matching> : {sub_swts_matching} --> {round(cus_swts_price_item2_discounted,2) = }')
            if np.array_equal(sub_ts_matching,opt_matching):
                print(f'/ <ts   matching> : \x1b[6;30;42m{sub_ts_matching}\x1b[0m --> {round(cus_ts_price_item2_discounted,2) = }')
            else:
                print(f'/ <ts   matching> : {sub_ts_matching} --> {round(cus_ts_price_item2_discounted,2) = }')
            print(f'\ <opt  matching> : {opt_matching} --> {round(opt_price_item2_discounted,2) = }')

            # storing rewards
            daily_swts_reward += (cus_swts_reward_item1 + cus_swts_reward_item2 )
            daily_ts_reward += (cus_ts_reward_item1 + cus_ts_reward_item2 )
            daily_opt_reward += (opt_customer_reward_item1 + opt_customer_reward_item2 )
        v_daily_swts_reward.append(daily_swts_reward)
        v_daily_ts_reward.append(daily_ts_reward)
        v_daily_opt_reward.append(daily_opt_reward)
    v_swts_experimets[e:] = np.cumsum(v_daily_opt_reward) - np.cumsum(v_daily_swts_reward)
    v_ts_experimets[e:] = np.cumsum(v_daily_opt_reward) - np.cumsum(v_daily_ts_reward)
                
# ploting results
plt.figure(1)
plt.xlabel("Days")
plt.ylabel("Regret")
plt.plot(np.mean(v_swts_experimets,axis=0),'-', color='darkorange', label = 'SWTS')
plt.plot(np.mean(v_ts_experimets,axis=0),'-', color='blue', label = 'TS')
plt.axvline(x=seasonality[0],linestyle=':',color='orange')
plt.axvline(x=seasonality[1],linestyle=':',color='orange')
plt.axvline(x=seasonality[2],linestyle=':',color='orange')
plt.title("Regret")
plt.legend()
plt.show()