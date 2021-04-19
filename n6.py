from Context import *    
import matplotlib.pyplot as plt
from Algorithms.UCB_Matching import *
from Algorithms.TS_Learner import *

ctx = Context()

days = 365 # 365 days of simulations

# define the prices candidates for the first item
candidates_item1 = [2260.0, 1910.0, 2130.0, 2010.0, 2340.0]
candidates_item2 = [500.0, 310.0, 350.0, 420.0, 550.0]

#discounted_price = ctx.discuonted_second_item_prices(promotion_assignment) # retrun the discounted prices for every customer category, according to the pormotion assignment

n_exp = 10 
experiments = np.zeros((n_exp,days))
experimets_item2_regret_curve = np.zeros((n_exp,days))
for e in range(n_exp):
    ts_learner_item1 = TS_Learner(len(candidates_item1))
    ts_learner_item2 = TS_Learner(len(candidates_item2))

    daily_regret = []
    daily_regret_item2 = []
    for d in range(days):
        opt_reward = 0.0
        tot_reward = 0.0
        reward_item2 = 0.0
        opt_reward_item2 = 0.0
        proposes_item2 = 0

        # ask to the learner to pull the most promising price that maximize the reward
        ts_pulled_arm_item1 = ts_learner_item1.pull_arm()
        ts_pulled_arm_item2 = ts_learner_item2.pull_arm() #select candidates for the seconf item
        # extract the daily customer. It is not unknown and is used to estimate the maximum possible reward
        customer_per_class = ctx.customers_daily_instance() 
        daily_customer_weight = customer_per_class.copy()
        tot_client = sum(customer_per_class)
        # simulate the day client by client, proposing the first item at the price provided by teh learner
        for customer in range(tot_client):
            category = np.random.choice(np.nonzero(customer_per_class)[0])
            customer_per_class[category] -= 1

            ts_buy_or_not_item1 = ctx.purchase_online_first_element(candidates_item1[ts_pulled_arm_item1],category) 
            opt_buy_or_not_item1 = ctx.purchase_online_first_element(min(candidates_item1),category)
            # the profit from the sale of the first item is added to the estimation of the rewenue that the customer buy the second item (depend only form the user category) 
            if ts_buy_or_not_item1:
                proposes_item2 += 1
                ts_buy_or_not_item2 = ctx.purchase_online_second_element(candidates_item2[ts_pulled_arm_item2],category) 
                reward_item2 += candidates_item2[ts_pulled_arm_item2]*ts_buy_or_not_item2
                # calculate the reward
                tot_reward += candidates_item1[ts_pulled_arm_item1] + candidates_item2[ts_pulled_arm_item2]*ts_buy_or_not_item2
            if (opt_buy_or_not_item1):
                temp = min(candidates_item2)* ctx.purchase_online_second_element(min(candidates_item2),category)
                opt_reward_item2 += temp
                opt_reward += min(candidates_item1) + temp

        maximum_rewards_item1 = (tot_client *( max(candidates_item1) + max(candidates_item2) )) # parameter used to normalize the reward
        maximum_rewards_item2 = (proposes_item2 *( max(candidates_item2) )) # parameter used to normalize the reward
        # end of the day. update the learner normalizing the reward
        ts_learner_item1.update(ts_pulled_arm_item1, tot_reward/maximum_rewards_item1)
        if d>40:
            ts_learner_item2.update(ts_pulled_arm_item2, reward_item2/maximum_rewards_item2)
        else:
            ts_learner_item2.update(ts_pulled_arm_item2, 0)

        # collect the daily rewards
        daily_regret.append(opt_reward - tot_reward)
        daily_regret_item2.append(opt_reward_item2 - reward_item2)

        print('___________________')
        print(f'| Day: {d+1} - Experiment {e+1}')
        print(f'| Today customers distribution : {daily_customer_weight}')
        print(f'| [TS]  - {ctx.items_info[0]["name"]} price: {candidates_item1[ts_pulled_arm_item1]} €, {ctx.items_info[1]["name"]} price: {candidates_item2[ts_pulled_arm_item2]} - Today reward:\t{round(tot_reward,2)}')
        print(f'| [OPT] - {ctx.items_info[0]["name"]} price: {min(candidates_item1)} €, {ctx.items_info[1]["name"]} price: {min(candidates_item2)} - Today reward:\t{round(opt_reward,2)}')

        print(f'Item1 -> {ts_pulled_arm_item1} - Rew: {maximum_rewards_item1} Norm: {tot_reward/maximum_rewards_item1}\nItem2 -> {ts_pulled_arm_item2} - Rew: {maximum_rewards_item2} Norm: {reward_item2/maximum_rewards_item2}\n')
        print(ts_learner_item1.beta_parameters)
        print(ts_learner_item2.beta_parameters)
        
    # end experiment 
    experiments[e,:]= daily_regret
    experimets_item2_regret_curve[e,:]= daily_regret_item2

plt.figure(0)
plt.xlabel("day")
plt.ylabel("Total Regret")
plt.plot(np.mean(np.cumsum(experiments,axis=1),axis=0),'-', color='green', label = 'Optimal - TS')
plt.legend()

plt.figure(1)
plt.xlabel("day")
plt.ylabel("Item2 Regret")
plt.plot(np.mean(np.cumsum(experimets_item2_regret_curve,axis=1),axis=0),'-', color='red', label = 'Optimal - TS')
plt.legend()

plt.show()