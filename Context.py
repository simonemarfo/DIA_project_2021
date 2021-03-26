import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from config import *

def interpolate(x, y):
    new_x = np.linspace(x.min(), x.max(), num=int((x.max()-x.min())/0.5)) # step of 0.50€
    f = interp1d(x, y, kind='quadratic')
    return new_x, f(new_x)

class Context():
    def __init__(self):
        
        self.classes_info = classes_info # load the class information: {'id':0, 'name':'class-1','color':'r'}
        self.n_classes = len(classes_info)
        
        self.item1_prices = item1_prices # candidates price for first item: [pr0, pr1, pr2, pr3]
        self.item2_prices = item2_prices # candidates price for second item
        self.item1_probabilities = item1_probabilities # item1 seasonal probabilities: [{'id':0,'name':'season1', 'probabilities':[[]], {...}]
        self.item2_probabilities = item2_probabilities 
        

    def plot_conversion_rate(self,prices,probabilities, title='Conversion rate'):
        fig, axs = plt.subplots(3,constrained_layout=True)
        fig.suptitle(title)
        for season, ax in enumerate(axs):
            for c in self.classes_info:
                idx = c['id']
                f_x, f_y = interpolate(np.array(prices), np.array(probabilities[season]['probabilities'][idx]))
                ax.plot(f_x,f_y,color=c['color'],label=c['name'])
                ax.scatter(prices,probabilities[season]['probabilities'][idx],marker='o',color=c['color'])
                ax.legend()
                ax.set_title(probabilities[season]['name'])
                ax.set_xlabel("price")
                ax.set_ylabel("probabiliy")

        plt.show()
    
    def plot_item1_conversion_rate(self):
        self.plot_conversion_rate(self.item1_prices,self.item1_probabilities, 'Conversion rate: first item')

    def plot_item2_conversion_rate(self):
        self.plot_conversion_rate(self.item2_prices,self.item2_probabilities,  'Conversion rate: second item')




ctx = Context()
ctx.plot_item1_conversion_rate()
ctx.plot_item2_conversion_rate()