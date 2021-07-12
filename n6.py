from Context import *    
import matplotlib.pyplot as plt
from Algorithms.promo_category_UCB_learner import *
from Algorithms.TS_Learner import *

ctx = Context()
days = 365 # 365 days of simulations
n_exp = 3

# define the prices candidates for the first and second item
candidates_item1 = [2110.0, 1900.0, 2420.0, 2690.0]
candidates_item2 = [360.0, 410.0, 530.0, 600.0]

# optimal solution for the seasoson with this candidates
opt_prices,opt_matching, best_daily_reward = ctx.correlated_optimal_solution(candidates_item1,candidates_item2,season=0) # return  best_prices[p1,p2],best_matching, best_reward
opt_price_item1 = opt_prices[0]
opt_price_item2 = opt_prices[1]

v_cus_experimets = np.zeros((n_exp,days))
for e in range(n_exp):
    # LEARNERS
    ts_learner = TS_Learner(len(candidates_item1) * len(candidates_item2)) # superarm of couple price_item1, price_item2: <p1,p2>
    normalizing_value = max(candidates_item1) + max(candidates_item2) # value used to normalize the customer reward, used to update the learner
    # UCB Matching learner, one learner for each couple <p1,p2>
    matching_learners = [promo_category_UCB_learner(np.zeros((4,4)).size, *np.zeros((4,4)).shape, 1000 ,max(candidates_item2)) for _ in range(len(candidates_item1) * len(candidates_item2))]
    v_daily_cus_reward = []
    v_daily_opt_reward = []
    for d in range(days):
        # extract the daily customer. It is UNKNOWN
        customer_per_class = ctx.customers_daily_instance() 
        daily_customer_weight = customer_per_class.copy()
        tot_client = sum(customer_per_class)
        daily_cus_reward = 0.0
        daily_opt_reward = 0.0
        # simulate the day client by client
        for customer in range(tot_client):
            customer_reward_item1 = 0.0
            customer_reward_item2 = 0.0
            opt_customer_reward_item1 = 0.0 # opt reward
            opt_customer_reward_item2 = 0.0 # opt reward

            category = np.random.choice(np.nonzero(customer_per_class)[0])
            customer_per_class[category] -= 1

            # ask to the learner to pull the most promising couple <p1,p2> that maximize the reward
            ts_pulled_arm = ts_learner.pull_arm() # number between 0..24
            cus_price_item1 = candidates_item1[ts_pulled_arm // len(candidates_item1)]
            cus_price_item2 = candidates_item2[ts_pulled_arm % len(candidates_item2)]

            # query the corresponding superarm learner 
            sub_matching = matching_learners[ts_pulled_arm].pull_arm() # suboptimal matching. row_ind, col_ind
            cus_price_item2_discounted = cus_price_item2 * (1-ctx.discount_promos[ sub_matching[1][category] ])
            opt_price_item2_discounted = opt_price_item2 * (1-ctx.discount_promos[ opt_matching[1][category] ])
            # purchase simulations
            cus_buy_or_not_item1 = ctx.purchase_online_first_element(cus_price_item1,category) 
            opt_buy_or_not_item1 = ctx.purchase_online_first_element(opt_price_item1,category)
            cus_buy_or_not_item2 = 0
            opt_buy_or_not_item2 = 0

            # compute the rewenue of the first and second item for both optimal solution and the online learning
            if cus_buy_or_not_item1:
                cus_buy_or_not_item2 = ctx.purchase_online_second_element(cus_price_item2_discounted, category) 
            if opt_buy_or_not_item1:
                opt_buy_or_not_item2 = ctx.purchase_online_second_element(opt_price_item2_discounted, category) 
            # computing rewards
            customer_reward_item1 = cus_buy_or_not_item1 * cus_price_item1
            customer_reward_item2 = cus_buy_or_not_item2 * cus_price_item2_discounted
            opt_customer_reward_item1 = opt_buy_or_not_item1 * opt_price_item1
            opt_customer_reward_item2 = opt_buy_or_not_item2 * opt_price_item2_discounted

            # update learners
            ts_learner.update(ts_pulled_arm, (customer_reward_item1 + customer_reward_item2 )/normalizing_value)
            if cus_buy_or_not_item1:
                matching_learners[ts_pulled_arm].update(sub_matching, customer_reward_item2, category=category)
            
            print('___________________')
            print(f'| Day: {d+1} - Experiment {e+1}')
            print(f'| Today customers distribution : {daily_customer_weight}')
            print(f'| Customer #{customer} of category: {ctx.classes_info[category]["name"]}: ')
            print(f'| {cus_price_item1 = } --- {cus_price_item2 = }')
            print(f'| {opt_price_item1 = } --- {opt_price_item2 = }')
            print(f'/ <sub matching> : {sub_matching} --> {round(cus_price_item2_discounted,2) = }')
            print(f'\ <opt matching> : {opt_matching} --> {round(opt_price_item2_discounted,2) = }')

            # storing rewards
            daily_cus_reward += (customer_reward_item1 + customer_reward_item2 )
            daily_opt_reward += (opt_customer_reward_item1 + opt_customer_reward_item2 )
        v_daily_cus_reward.append(daily_cus_reward)
        v_daily_opt_reward.append(daily_opt_reward)
    v_cus_experimets[e:] = np.cumsum(v_daily_opt_reward) - np.cumsum(v_daily_cus_reward)


# ploting results
plt.figure(1)
plt.xlabel("Days")
plt.ylabel("Regret")
plt.plot(np.mean(v_cus_experimets,axis=0),'-', color='darkorange', label = 'Online solution')
plt.title("Regret")
plt.legend()
plt.show()