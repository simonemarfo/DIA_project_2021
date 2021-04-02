import numpy as np
from scipy.optimize import linear_sum_assignment # library that implement this algorithm
import matplotlib.pyplot as plt

matrix = np.array([[600,580,640,700],
                   [590,610,720,540],
                   [610,670,530,490],
                   [700,690,630,650]]).astype(float)

total_clients = 700
total_promo = 500
no_promo = 200

class_final_distribution = np.zeros((4,4))
iteration_matrix = []


for i in range(4):
    row_ind,col_ind = linear_sum_assignment(matrix,maximize=True) # optimization 
    print("\nScipy linear sum assignment\n\nMatrix:\n", matrix, "\nOptimal Matching:\n",row_ind,col_ind, matrix[row_ind,col_ind].sum())
    # preparing matrix for next iteration
    temp = np.zeros((4,4))
    for ind in range(0,len(row_ind)):
        temp[row_ind[ind],col_ind[ind]] =  matrix[row_ind[ind],col_ind[ind]]
        matrix[row_ind[ind],col_ind[ind]] = -1
    
    iteration_matrix.append(temp)

print(iteration_matrix)

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

noPromoDistribution(iteration_matrix, class_final_distribution)
promoDistribution(iteration_matrix, class_final_distribution)

print(class_final_distribution)

final_no_promo = np.multiply(class_final_distribution[:,0], no_promo/100 )
final_yes_promo = np.multiply(class_final_distribution[:,1:4], total_promo/100)

print(final_no_promo.round())
print(final_yes_promo.round())