import numpy as np
from scipy.optimize import linear_sum_assignment # library that implement this algorithm
import matplotlib.pyplot as plt
from Context import *

def optimalSolutionsIterations(matching_matrix, verbose = False):
    iteration_matrix = []
    for i in range(4):
        row_ind,col_ind = linear_sum_assignment(matching_matrix,maximize=True) # optimization 
        if verbose:
            print(f"\nScipy linear sum assignment\n\nMatrix:\n{matching_matrix}\nOptimal Matching: rows{row_ind} cols{col_ind} value {matching_matrix[row_ind,col_ind].sum()}")
        # preparing matrix for next iteration
        temp = np.zeros((4,4))
        for ind in range(0,len(row_ind)):
            temp[row_ind[ind],col_ind[ind]] =  matching_matrix[row_ind[ind],col_ind[ind]]
            matching_matrix[row_ind[ind],col_ind[ind]] = -1
        
        iteration_matrix.append(temp)
    return iteration_matrix


# nopromo 
def noPromoDistribution(iteration_matrix, class_final_distribution, verbose = False):
    iter_sum = [] # sum of optimal value at each iteration 
    cell_weight = [] # product of iteration_sum and the optimal value of P0 (no promo)
    iteration_order = [] # refer the class in which the P0 discount is selected as optimal

    for i in range(4):
        iter_sum.append(np.sum(iteration_matrix[i]))
        cell_weight.append(iter_sum[i] * np.max(iteration_matrix[i][:,0]))
        iteration_order.append(np.nonzero(iteration_matrix[i][:,0])[0][0])

    tot_amount = np.sum(cell_weight) # sum 
    
    # for each class determine the percentage of no promos to assign 
    for i in range(4):
        class_final_distribution[iteration_order[i],0] = cell_weight[i] *100 / tot_amount

    if verbose:
        print(f'iteration sum array : {iter_sum}')
        print(f'total weight for each cell : {cell_weight}')
        print(f'iteration order :  {iteration_order}')
        print(f'total amount {tot_amount}')
        print(f'Final class distribution promo matrix :\n{class_final_distribution}')

#promos 
def promoDistribution(iteration_matrix, class_final_distribution, verbose = False):
    # outer-class: 
    iter_sum = [] # sum of optimal value at each iteration 
    cell_weight = [] # product of iteration_sum and the the sum of the revenue applying the promos  (P1,P2,P3)
    iteration_order = [] # refer the class in which the P0 discount is selected as optimal INUTILE
    promo_per_iteration = [] # pergentage of promo assigned at each iterations that must be splitted over classes
    total_revenue_with_promos = [] # total revenue applying the discount

    for i in range(4):
        iter_sum.append(np.sum(iteration_matrix[i]))
        cell_weight.append(iter_sum[i] * np.sum(iteration_matrix[i][:,1:4]))
        total_revenue_with_promos.append(np.sum(iteration_matrix[i][:,1:4]))
        #iteration_order.append(np.nonzero(iteration_matrix[i][:,0])[0][0])

    tot_amount = np.sum(cell_weight) # sum 
    
    # for each class determine the percentage of no promos to assign 
    for i in range(4):
        promo_per_iteration.append(cell_weight[i] *100 / tot_amount)

    # inter-class:
    coordinates = []
    for i in range(4):
        coordinates = np.nonzero(iteration_matrix[i][:,1:4])  # NB: working wit a 3x4 matrix coordinates = [0,1,2],[2,3,0]
        for ind in range(len(coordinates[0])):
            class_final_distribution[coordinates[0][ind], coordinates[1][ind] +1] = (iteration_matrix[i][coordinates[0][ind], coordinates[1][ind] +1] * 100 / total_revenue_with_promos[i]) * promo_per_iteration[i] / 100

    if verbose:
        print(f'iteration sum array : {iter_sum}')
        print(f'total weight for each cell : {cell_weight}')
        print(f'iteration order :  {iteration_order}')
        print(f'total amount {tot_amount}')
        print(f'promo per iteration :{promo_per_iteration}')
        print(f'Final class distribution promo matrix :\n{class_final_distribution}')

def computeClassPromoDistribution(iteration_matrix,class_final_distribution,verbose=False):
    noPromoDistribution(iteration_matrix, class_final_distribution,verbose=verbose) # compute the first column related to P0 (no doscount)
    promoDistribution(iteration_matrix, class_final_distribution,verbose=verbose)   # compute the distribution for promos P1, P2, P3
    # normalize the distributions row by row
    for i in range(0,4):
        sum_per_class=(np.sum(class_final_distribution[i,:]))
        for j in range(0,4):
            class_final_distribution[i,j] = (class_final_distribution[i,j]*100/sum_per_class)/100  # do not cast to integer!
    return class_final_distribution

#
# Experiment 1 
#

item1_price_full = 2350.0
item2_price_full = 630.0 
class_final_distribution = np.zeros((4,4))  # this 4x4 matrix contains the probablilty that to a user, belonging to a category (row) is given a certaind discount (columns)

# context generation 
ctx = Context()
customer_daily = ctx.customers_daily_instance() # return a vector corresponding to numbers of customers per class 
total_clients = np.sum(customer_daily)
no_promo = int(total_clients * ctx.amount_of_no_promos) # percentage no-promo over the daily total number of customers  
total_promo = total_clients - no_promo


# Calculate of the customers that buy the first item
# Use the conversion rate of the first item (at the defined price), as fractions of buyers
first_item_acquirents = np.zeros((4))

for i in range (0,4):
    first_item_acquirents[i]=int(customer_daily[i] * ctx.conversion_rate_first_element(item1_price_full, i))

# knowing the numbers of customers that bought the first item, we aims to maximize the profit making them buy the second item
# Considering as known the conversion rate of each class, in order to maximize the profit we can determine which discout apply to a class 
# Solved as Matching Problem: match every user category to all the four possible discounts (P0, P1, P2, P3) with the pobability to apply it in order to maximize the profit

# discounted price for the second items
discounted_price = [item2_price_full,
    item2_price_full*(1-ctx.discount_promos[1]),
    item2_price_full*(1-ctx.discount_promos[2]),
    item2_price_full*(1-ctx.discount_promos[3])]

# Matching matrix: rows[0..3] are the user categories; columns[0..3] are the discouts; celles are the weights calculated as (conversion_rate * discounted_price * tot_clients) of that class
matching_matrix = np.zeros((4,4))
for i in range (0,4): #classes
    for j in range (0,4): #promos
        matching_matrix[i,j] = int(discounted_price[j]*(ctx.conversion_rate_second_element(discounted_price[j],i))*first_item_acquirents[i])


# the matching is performed iterating over the matching_matrix four times. Every iteration determine the optimal solution of the matching problem, which allow to maximize the profit
# the iteration_matrix save collect all these oprimal solutions
iteration_matrix = optimalSolutionsIterations(matching_matrix=matching_matrix.copy(),verbose=True)

# compiling the class final distribution matrix 
class_final_distribution = computeClassPromoDistribution(iteration_matrix,class_final_distribution,True)

# Output overview
print("\n\n#############\n")
print(f" {ctx.items_info[0]['name']}: {item1_price_full} €\n {ctx.items_info[1]['name']}: {item2_price_full} €\n Discouts (%): {[_*100 for _ in ctx.discount_promos]}")
print(f" Discounted {ctx.items_info[1]['name']}: {discounted_price} €")
print(f" Total daily custemers: {total_clients}\n Daily customers per class: {customer_daily}\n Total promo: {total_promo}\n No promo: {no_promo} --> {ctx.amount_of_no_promos * 100}% of total daily customes")
print("\n")
print(f" Customers that bought {ctx.items_info[0]['name']} : {first_item_acquirents}")
print(f"\n[*]Matching matrix\n\t{matching_matrix}\n[*]Iteration matrix\n\t{iteration_matrix} ")
print(f"\n\nOPTIMAL SOLUTION: PROBABILITY DISTRIBUTION OF PROMOS PER CLASS\n{class_final_distribution.round(2)}\n\n\n")

# testing optimal solution. Comparing rewards of no-discounts vs optimal solutions
n_experiments = 50

daily_reward_no_promotion_srategy = []
daily_reward_best_promotion_srategy = []
daily_reward_promotion_srategy = []

for t in range(n_experiments):
    daily_reward = [0,0,0]
    for category in range(len(customer_daily)):
        for customer in range(customer_daily[category]): # for each category emulate the user that purchase the good
            # buy first item 
            customer_probability = ctx.conversion_rate_first_element(item1_price_full,category)
            reward_item1 = ctx.purchase(customer_probability) * item1_price_full
            reward_item2 = 0.0 
            if(reward_item1 > 0): # propose second item
                ########################
                # NO PROMOTION STRATEGY
                ########################
                customer_probability = ctx.conversion_rate_second_element(item2_price_full,category)
                reward_item2 = ctx.purchase(customer_probability) * item2_price_full
                daily_reward[0] += reward_item1 + reward_item2

                ########################
                # BEST PROMOTION STRATEGY
                ########################
                reward_item2 = 0.0
                d_price = np.min(discounted_price)
                customer_probability = ctx.conversion_rate_second_element(d_price,category)
                reward_item2 = ctx.purchase(customer_probability) * d_price
                daily_reward[1] += reward_item1 + reward_item2


                ########################
                # PROMOTION STRATEGY
                ########################
                reward_item2 = 0.0
                d_price = np.random.choice(discounted_price, p=class_final_distribution[category])
                customer_probability = ctx.conversion_rate_second_element(d_price,category)
                reward_item2 = ctx.purchase(customer_probability) * d_price
                daily_reward[2] += reward_item1 + reward_item2

            #print(f"[NP]Customer {customer}, category {category} -- item1: {reward_item1}€ --- item2: {reward_item2}€\n [$] Daily reward: {daily_reward[0]}€")
            #print(f"[BP]Customer {customer}, category {category} -- item1: {reward_item1}€ --- item2: {reward_item2}€\n [$] Daily reward: {daily_reward[1]}€")
            #print(f"[P]Customer {customer}, category {category} -- item1: {reward_item1}€ --- item2: {reward_item2}€\n [$] Daily reward: {daily_reward[2]}€")              
            
    daily_reward_no_promotion_srategy.append(daily_reward[0])
    daily_reward_best_promotion_srategy.append(daily_reward[1])
    daily_reward_promotion_srategy.append(daily_reward[2])


print(f"No strategy promotion: {np.sum(daily_reward_no_promotion_srategy)}\nBest Promotion strategy: {np.sum(daily_reward_best_promotion_srategy)}\nPromotion strategy: {np.sum(daily_reward_promotion_srategy)}")

plt.figure(0)
plt.xlabel("run")
plt.ylabel("Daily reward")
plt.plot(daily_reward_no_promotion_srategy,'-o', color='black')
plt.plot(daily_reward_best_promotion_srategy,'-o', color='green')
plt.plot(daily_reward_promotion_srategy,'-o', color='red')
plt.legend(['No Promotion Strategy','Best Promotion Strategy','Promotion Strategy'])
plt.show()
