from Context import *    
import matplotlib.pyplot as plt
from Algorithms.UCB_Matching import *

ctx= Context()
n_exp = 1
days = 10

item1_price_full = 2350.0
item2_price_full = 630.0 

#Matrix conversion rate second item

discounted_price = [item2_price_full,
    item2_price_full*(1-ctx.discount_promos[1]),
    item2_price_full*(1-ctx.discount_promos[2]),
    item2_price_full*(1-ctx.discount_promos[3])]

# Matching matrix: rows[0..3] are the user categories; columns[0..3] are the discouts; celles are the conversion rate of that class
conversion_rate_second = np.zeros((4,4))
for i in range (0,4): #classes
    for j in range (0,4): #promos
        conversion_rate_second[i,j] = (ctx.conversion_rate_second_element(discounted_price[j], i))

priced_conversion_rate_second = np.zeros((4,4))
opt_rew = []
for i in range (0,4): #classes
    for j in range (0,4): #promos
        priced_conversion_rate_second[i,j] =conversion_rate_second[i,j] * discounted_price[j]
opt=linear_sum_assignment(priced_conversion_rate_second, maximize=True)
period_reward = []

for e in range(n_exp):
    learner= UCB_Matching(conversion_rate_second.size, *conversion_rate_second.shape) #Inizializzazione learner UCB
    #print(e)
    rew_UCB=[]


    for t in range(days):
        #Nuova giornata
        #1. Generazione clienti e distribuzione per le varie classi
        print(f"DAY: {t}")
        daily_customer = ctx.customers_daily_instance()
        #daily_customer= [605, 776, 2131, 1210]
        print(f"DISTRIBUTION CUSTOMER : {daily_customer}")
        cum_rewards = 0
        cum_opt_reward = 0
        
        for category in range(len(daily_customer)):
            for customer in range(daily_customer[category]): # for each category emulate the user that purchase the good
                # buy first item 
                #2. Simulazione acquisto primo elemento, senza ottimizzazioni
                reward_item1 = ctx.purchase_online_first_element(item1_price_full,category) * item1_price_full
                cum_rewards += reward_item1
                cum_opt_reward += reward_item1
                #3. Se acquista il primo elemento, chiamata bandit 
                if (reward_item1 > 0):
                    #4. Chiediamo al learner il matching per l'ottimizzazione promo-classi + #5. Restituzione righe,colonne con max reward
                    sub_matching = learner.pull_arm()
                    #6. Simulazione acquisto tramite Context solo sulla classe-promo suggerita                    
                    reward_item2 = ctx.purchase_online_second_element(discounted_price[sub_matching[1][category]],category) 
                    cum_rewards += (reward_item2* discounted_price[sub_matching[1][category]])
                    rewards_to_update = np.zeros ((4))
                    rewards_to_update[category] = reward_item2
                    learner.update(sub_matching,rewards_to_update)
                    cum_opt_reward += (ctx.purchase_online_second_element(discounted_price[opt[1][category]],category) * discounted_price [opt[1][category]])
        
        opt_rew.append(cum_opt_reward)
        period_reward.append(cum_rewards)
    
        print(f"DAILY REWARD CUMULATIVE: {cum_rewards}")
        print("CONFIDENCE")
        print(learner.confidence)
    print(f"Period Reward: {period_reward}")
    print(f"Optimal Reward: {opt_rew}")
   
    mean_reward=np.mean(period_reward)
    print(f"MEAN REWARD 365d: {mean_reward}")
        
                    
                

    #print(f"Cumulative rewards {cum_rewards}")
                #7. Passare al Learner il vettore di reward partizionato alla classe --> learner.update(pulled_arms,[0,rewards[1],0])
                #8. Salvare i risultati
                #9. Torna al punto 2 per gli altri clienti


#2410829.88
"""
        learner.update(pulled_arms,rewards)
        rew_UCB.append(rewards.sum())
        opt_rew.append(p[opt].sum())
    regret_ucb[e,:]=np.cumsum(opt_rew)-np.cumsum(rew_UCB)
    print(learner.confidence)
plt.figure(0)
plt.plot(regret_ucb.mean(axis=0))
plt.ylabel('Regret')
plt.xlabel('t')
plt.show()"""