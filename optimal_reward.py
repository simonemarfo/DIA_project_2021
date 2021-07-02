from config import *
from Context import *
import numpy as np
from scipy.optimize import linear_sum_assignment

"""
f(x,y) = empirical optimal reward of item1 and item2

given 5 candidates per item
"""

ctx = Context()
candidates_item1 = [2110.0, 1900.0, 2420.0, 2690.0]
candidates_item2 = [360.0, 410.0, 530.0, 600.0]

for season in range(0,3):
    print(f"Season : {item1_probabilities[season]['name']}")
    best_reward = - np.inf
    best_matching = []
    best_prices = []
    for x in range(0,len(candidates_item1)):
        for y in range(0,len(candidates_item2)):
            p1 = candidates_item1[x]
            p2 = candidates_item2[y]
        
            # ite1 reward 
            reward_item1 = 0
            for c in range(0,4):
                reward_item1 += ctx.customersDistribution[c][0] * ctx.conversion_rate_first_element(p1,c,season) * p1

            # item2 conversion matrix
            discounted_price = [p2, p2*(1-ctx.discount_promos[1]), p2*(1-ctx.discount_promos[2]), p2*(1-ctx.discount_promos[3])]
            matrix = np.zeros((4,4))
            for i in range (0,4): #classes
                for j in range (0,4): #promos
                    matrix[i,j] = (ctx.conversion_rate_first_element(p1,c,season) * ctx.conversion_rate_second_element(discounted_price[j], i,season)) * ctx.customersDistribution[i][0] *  discounted_price[j]
            matching_opt = linear_sum_assignment(matrix, maximize=True) # optimal solution row_ind, col_ind

            # reward item2 
            reward_item2 = 0
            for c in range(0,4):
                reward_item2 += matrix[c][matching_opt[1][c]]

            #print(f"   <{p1},{p2}>, reward = {reward_item1 + reward_item2}")
            # save max and opt config
            if (reward_item1 + reward_item2)>best_reward:
                best_reward = reward_item1 + reward_item2
                best_prices = [p1,p2]
                best_matching = matching_opt


    #results: 
    print(f"{best_reward =}")
    print(f"{best_prices = }")
    print(f"{best_matching = }\n\tSport Addicted =>{ctx.discount_promos[best_matching[1][0]] * 100}%\n\tGifter =>{ctx.discount_promos[best_matching[1][1]] * 100}%\n\tAmateur =>{ctx.discount_promos[best_matching[1][2]] * 100}%\n\tGifter =>{ctx.discount_promos[best_matching[1][3]] * 100}%")
    

        
        
