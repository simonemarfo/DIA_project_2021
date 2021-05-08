from Context import * 
import numpy as np
from itertools import combinations
import sys

# redirect all the print into a file
orig_stdout = sys.stdout
sys.stdout = open("test_conversion_rate_out.txt","w")

# start script
ctx = Context()
item1_range = np.arange(1900,2350,5,dtype = int)
comb_item1 = combinations(item1_range,3)
item2_range = np.arange(360,650,5,dtype = int) # parte da 360 per via dello sconto
comb_item2 = combinations(item2_range,3)

all_diff_item1 = []
for el in comb_item1:
    opt_item1=np.zeros((3),dtype=int)
    for season in range (3): 
        opt_rew_item1 = np.zeros((len(el)))
        for i in range(len(el)):
            for c in range(4):
                    opt_rew_item1[i] += ctx.conversion_rate_first_element(el[i],c,season) * el[i] * ctx.customersDistribution[c,0]
        opt_item1[season] = int(np.argmax(opt_rew_item1))
    
    if(opt_item1[0] != opt_item1[1] or opt_item1[0] != opt_item1[2] or opt_item1[2] != opt_item1[1]):
        if(opt_item1[0] != opt_item1[1] and opt_item1[0] != opt_item1[2] and opt_item1[2] != opt_item1[1]):
            print(f'!!!!    {el} ==> {opt_item1} !!!!')
            all_diff_item1.append(el)
        else:
            print(f'{el} ==> {opt_item1}')

print(f'All different : \n{all_diff_item1}')

all_diff_item2 = []
for el in comb_item2:
    opt_item2=np.zeros((3),dtype=int)
    for season in range (3): 
        opt_rew_item2 = np.zeros((len(el)))
        for i in range(len(el)):
            for c in range(4):
                    opt_rew_item2[i] += ctx.conversion_rate_second_element(el[i],c,season) * el[i] * ctx.customersDistribution[c,0]
        opt_item2[season] = int(np.argmax(opt_rew_item2))
    
    if(opt_item2[0] != opt_item2[1] or opt_item2[0] != opt_item2[2] or opt_item2[2] != opt_item2[1]):
        if(opt_item2[0] != opt_item2[1] and opt_item2[0] != opt_item2[2] and opt_item2[2] != opt_item2[1]):
            print(f'!!!!    {el} ==> {opt_item2} !!!!')
            all_diff_item2.append(el)
        else:
            print(f'{el} ==> {opt_item2}')

print(f'All different : \n{all_diff_item2}')

# end script
# restore stdout and open an interactive console
sys.stdout = orig_stdout
print("Script ended")