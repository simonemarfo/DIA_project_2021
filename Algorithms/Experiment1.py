import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from config import *


#Starting prices 
p1=2400
p2=630

#Cost are considered equal to 0

#Average customers per class
avg_customer=[150,200,400,250] 

conversion_rate_prod1=[.66,.5,.4,.5]

discount=[0,.1,.2,.3]

def calculateDiscount(price,discounts):
    to_discount=np.multiply(price,discounts)
    return np.subtract(price,to_discount)

print(calculateDiscount(p2,discount))
