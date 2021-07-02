import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from config import *

def interpolate(x, y):
    new_x = np.linspace(x.min(), x.max(), num=int((x.max()*1.1-x.min())/5)) # step of 5€
    f = interp1d(x, y, kind='quadratic')
    return new_x, f

class Context():
    def __init__(self):
        
        self.items_info = items_info # items information: [{'id':0, 'name':'name1'},...]
        self.classes_info = classes_info # load the class information: [{'id':0, 'name':'class-1','color':'r'},...]
        self.n_classes = len(classes_info)
        self.item1_full_price=2200.0
        self.item2_full_price=630.0
        self.item1_prices = item1_prices # candidates price for first item: [pr0, pr1, pr2, pr3]
        self.item2_prices = item2_prices # candidates price for second item
        self.item1_probabilities = item1_probabilities # item1 seasonal probabilities: [{'id':0,'name':'season1', 'probabilities':[[]], {...}]
        self.item2_probabilities = item2_probabilities 
        self.customersDistribution = np.multiply(gaussDistributionParam,maxDailyCustomers) # mu,sigma parameters for gaussian distribution
        self.amount_of_no_promos = no_promo1
        self.discount_promos = promos     #P0=0% P1=10% P2=20% P3=30%

        
        

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

    def plot_customers_distribution(self):
        from scipy.stats import norm
        plt.figure(0)
        x_mean = np.mean(self.customersDistribution[:,0])
        x_axis = np.arange(0, x_mean*3 ,10)

        for idx, d in enumerate(self.customersDistribution):
            plt.plot(x_axis , norm.pdf(x_axis,d[0],d[1]), color=self.classes_info[idx]['color'],label=self.classes_info[idx]['name'])
        plt.legend()
        plt.title("Customers distribution")
        plt.show()

    def customers_daily_instance(self): #TO COMPLETE 
        dailyCustomer = []
    
        for i in range(0,self.n_classes):
            sample = np.random.normal(self.customersDistribution[i,0],self.customersDistribution[i,1])
            dailyCustomer.append(int(sample))

        return dailyCustomer


    def conversion_rate(self, current_price, prices, probabilities):                    
        f_x, f = interpolate(np.array(prices), np.array(probabilities))
        return f(current_price)

    def conversion_rate_first_element(self, current_price, customer_class, season = 0): #Season = 0 : default Spring/Summer
        return self.conversion_rate(current_price, self.item1_prices, self.item1_probabilities[season]['probabilities'][customer_class])

    def conversion_rate_second_element(self, current_price, customer_class, season = 0): #Season = 0 : default Spring/Summer
        return self.conversion_rate(current_price, self.item2_prices, self.item2_probabilities[season]['probabilities'][customer_class])

    def purchase(self,probability):  
        return np.random.binomial(1,probability) 

    def purchase_online_first_element(self, current_price, customer_class, season = 0):  
        probability = self.conversion_rate_first_element(current_price, customer_class, season)
        return np.random.binomial(1,probability) 

    def purchase_online_second_element(self, current_price, customer_class, season = 0):  
        probability = self.conversion_rate_second_element(current_price, customer_class, season)
        return np.random.binomial(1,probability) 

    def discuonted_second_item_prices(self, promo_assignment, item2_full_price=0):
        if item2_full_price==0:
            item2_full_price = self.item2_full_price
        return np.multiply(np.subtract(1,np.take(self.discount_promos,promo_assignment)),item2_full_price)
    
    def correlated_optimal_solution(self, candidates_item1,candidates_item2,season=0, verbose=False):
        if verbose: print(f"Correlated optimal solution for season : {item1_probabilities[season]['name']}")
        best_reward = - np.inf
        best_matching = []
        best_prices = []
        for x in range(0,len(candidates_item1)):
            for y in range(0,len(candidates_item2)):
                p1 = candidates_item1[x]
                p2 = candidates_item2[y]
                # item1 reward 
                reward_item1 = 0
                for c in range(0,4):
                    reward_item1 += self.customersDistribution[c][0] * self.conversion_rate_first_element(p1,c,season) * p1
                # item2 conversion matrix
                discounted_price = [p2, p2*(1-self.discount_promos[1]), p2*(1-self.discount_promos[2]), p2*(1-self.discount_promos[3])]
                matrix = np.zeros((4,4))
                for i in range (0,4): #classes
                    for j in range (0,4): #promos
                        matrix[i,j] = (self.conversion_rate_first_element(p1,c,season) * self.conversion_rate_second_element(discounted_price[j], i,season)) * self.customersDistribution[i][0] *  discounted_price[j]
                matching_opt = linear_sum_assignment(matrix, maximize=True) # optimal solution row_ind, col_ind
                # reward item2 
                reward_item2 = 0
                for c in range(0,4):
                    reward_item2 += matrix[c][matching_opt[1][c]]
                # save max and opt config
                if (reward_item1 + reward_item2)>best_reward:
                    best_reward = reward_item1 + reward_item2
                    best_prices = [p1,p2]
                    best_matching = matching_opt
        #results: 
        if verbose:
            print(f"{best_reward =}")
            print(f"{best_prices = }")
            print(f"{best_matching = }\n\tSport Addicted =>{ctx.discount_promos[best_matching[1][0]] * 100}%\n\tGifter =>{self.discount_promos[best_matching[1][1]] * 100}%\n\tAmateur =>{self.discount_promos[best_matching[1][2]] * 100}%\n\tGifter =>{self.discount_promos[best_matching[1][3]] * 100}%")
        return best_prices,best_matching, best_reward

    def uncorrelated_optimal_solution(self, candidates_item1,candidates_item2,season=0, verbose=False):
        if verbose: print(f"Uncorrelated optimal solution for season : {item1_probabilities[season]['name']}")
        best_reward = - np.inf
        best_reward_item1 = -np.inf
        best_reward_item2 = -np.inf
        best_matching = []
        best_prices = [0,0]
        # find max reward for item1
        for p1 in candidates_item1:
            reward_item1 = 0
            for c in range(0,4):
                reward_item1 += self.customersDistribution[c][0] * self.conversion_rate_first_element(p1,c,season) * p1
            if reward_item1 > best_reward_item1:
                best_reward_item1 = reward_item1
                best_prices[0] = p1
        # find max reward for item2
        for p2 in candidates_item2:
            reward_item2 = 0
            for c in range(0,4):
                reward_item2 += self.customersDistribution[c][0] * self.conversion_rate_second_element(p2,c,season) * p2
            if reward_item2 > best_reward_item2:
                best_reward_item2 = reward_item2
                best_prices[1] = p2
        # matching and compute the reward
        # item2 conversion matrix
        discounted_price = [best_prices[1], best_prices[1]*(1-self.discount_promos[1]), best_prices[1]*(1-self.discount_promos[2]), best_prices[1]*(1-self.discount_promos[3])]
        matrix = np.zeros((4,4))
        for i in range (0,4): #classes
            for j in range (0,4): #promos
                matrix[i,j] = (self.conversion_rate_first_element(best_prices[0],c,season) * self.conversion_rate_second_element(discounted_price[j], i,season)) * self.customersDistribution[i][0] *  discounted_price[j]
        matching_opt = linear_sum_assignment(matrix, maximize=True) # optimal solution row_ind, col_ind
        best_matching = matching_opt
        # reward item2 
        reward_item2 = 0
        best_reward_item2 = 0
        for c in range(0,4):
            best_reward_item2 += matrix[c][matching_opt[1][c]]
        best_reward = best_reward_item1 + best_reward_item2
        #results: 
        if verbose:
            print(f"{best_reward =}")
            print(f"{best_reward_item1 =}")
            print(f"{best_reward_item2 =}")
            print(f"{best_prices = }")
            print(f"{best_matching = }\n\tSport Addicted =>{ctx.discount_promos[best_matching[1][0]] * 100}%\n\tGifter =>{self.discount_promos[best_matching[1][1]] * 100}%\n\tAmateur =>{self.discount_promos[best_matching[1][2]] * 100}%\n\tGifter =>{self.discount_promos[best_matching[1][3]] * 100}%")
        return best_prices,best_matching, best_reward


#ctx = Context()
#ctx.plot_item1_conversion_rate()
#ctx.plot_item2_conversion_rate()