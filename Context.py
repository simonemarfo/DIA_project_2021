import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from config import *

def interpolate(x, y):
    new_x = np.linspace(x.min(), x.max(), num=int((x.max()-x.min())/5)) # step of 5€
    f = interp1d(x, y, kind='quadratic')
    return new_x, f

class Context():
    def __init__(self):
        
        self.classes_info = classes_info # load the class information: {'id':0, 'name':'class-1','color':'r'}
        self.n_classes = len(classes_info)
        
        self.item1_prices = item1_prices # candidates price for first item: [pr0, pr1, pr2, pr3]
        self.item2_prices = item2_prices # candidates price for second item
        self.item1_probabilities = item1_probabilities # item1 seasonal probabilities: [{'id':0,'name':'season1', 'probabilities':[[]], {...}]
        self.item2_probabilities = item2_probabilities 
        self.customersDistribution = np.multiply(gaussDistributionParam,maxDailyCustomers) # mu,sigma parameters for gaussian distribution
        self.amount_of_no_promos = no_promo1
        self.discount_promos= promos     #P0=0% P1=10% P2=20% P3=30%

        
        

    def plot_conversion_rate(self,prices,probabilities, title='Conversion rate'):
        fig, axs = plt.subplots(3,constrained_layout=True)
        fig.suptitle(title)
        for season, ax in enumerate(axs):
            for c in self.classes_info:
                idx = c['id']
                f_x, f = interpolate(np.array(prices), np.array(probabilities[season]['probabilities'][idx]))
                ax.plot(f_x,f(f_x),color=c['color'],label=c['name'])
                ax.scatter(prices,probabilities[season]['probabilities'][idx],marker='.',color=c['color'])
                ax.legend()
                ax.set_title(probabilities[season]['name'])
                ax.set_xlabel("price")
                ax.set_ylabel("probabiliy")
                ax.set_ylim(0,1)

        plt.show()
    
    def plot_item1_conversion_rate(self):
        self.plot_conversion_rate(self.item1_prices,self.item1_probabilities, 'Conversion rate: first item')

    def plot_item2_conversion_rate(self):
        self.plot_conversion_rate(self.item2_prices,self.item2_probabilities,  'Conversion rate: second item')


    def customers_daily_instance(self): #TO COMPLETE 
        dailyCustomer = []
    
        for i in range(0,self.n_classes):
            sample = np.random.normal(self.customersDistribution[i,0],self.customersDistribution[i,1])
            dailyCustomer.append(int(sample))

        return dailyCustomer


    def conversion_rate(self, p, prices, probabilities):
        f_x, f = interpolate(np.array(prices), np.array(probabilities))
        return f(p)

    def conversion_rate_first_element(self, p, customer_class, season = 0): #Season = 0 : default Spring/Summer
        return self.conversion_rate(p, self.item1_prices, self.item1_probabilities[season]['probabilities'][customer_class])

    def conversion_rate_second_element(self, p, customer_class, season = 0): #Season = 0 : default Spring/Summer
        return self.conversion_rate(p, self.item2_prices, self.item2_probabilities[season]['probabilities'][customer_class])

