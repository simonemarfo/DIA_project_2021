items = [{'id':0, 'name':'Mountain Bike'},
         {'id':1, 'name':'Garmin'}]
#FIXARE RANGE PREZZI; BELLA MOL; 6 valori proposti da mol;occhio ti guardiamo
item1_prices = [2060.0,2200.0,2290.0,2400.0]
item2_prices = [420.0,500.0,560.0,630.0]

promos = [.0,.1,.2,.3]    #P0=0% P1=10% P2=20% P3=30%

#Customers per class
maxDailyCustomers=1000
gaussDistributionParam =[[.15,.03],
                 [.20,.03],
                 [.40,.05],
                 [.25,.04]]
                 
#Fraction of customer with no promo(P0)
no_promo1=.7
no_promo2=.65

item1_probabilities =[{'id':0,'name':'Spring-Summer', 'probabilities':[[.75,.5,.5,.5],  #MTB - Sport Addicted
                                                                 [.6,.3,.2,.1],         #MTB - Gifter
                                                                 [.6,.4,.3,.1],         #MTB - Amateur
                                                                 [.7,.2,.2,.1]]},       #MTB - Worried
                      {'id':1,'name':'Autumn', 'probabilities':[[.7,.5,.45,.4],
                                                                 [.3,.15,.1,.1],
                                                                 [.5,.4,.2,.1],
                                                                 [.6,.3,.2,.1]]},
                      {'id':2,'name':'Winter', 'probabilities':[[.55,.3,.3,.2],
                                                                 [.7,.45,.4,.25],
                                                                 [.3,.1,.08,.05],
                                                                 [.45,.10,.08,.05]]}
                    ]
                      

item2_probabilities = [{'id':0,'name':'Spring-Summer', 'probabilities':[[.55,.35,.3,.3],    #GARMIN - Sport Addicted ...
                                                                  [.3,.15,.13,.08],
                                                                  [.45,.25,.2,.15],
                                                                  [.5,.15,.1,.05]]},
                       {'id':1,'name':'Autumn', 'probabilities':[[.55,.35,.3,.25],
                                                                  [.25,.12,.12,.09],
                                                                  [.3,.15,.15,.1],
                                                                  [.4,.1,.1,.05]]},
                       {'id':2,'name':'Winter', 'probabilities':[[.7,.43,.4,.3],
                                                                  [.45,.25,.2,.18],
                                                                  [.32,.17,.17,.1],
                                                                  [.55,.25,.2,.15]]},
                    ]
                       

classes_info = [{'id':0, 'name':'Sport Addicted','color':'r'}, #COMPRA A QUALSIASI PREZZO
                {'id':1, 'name':'Gifter','color':'b'},         #REGALA LA BICI
                {'id':2, 'name':'Amateur','color':'g'},        #FA SPORT OGNI TANTO
                {'id':3, 'name':'Worried','color':'y'}]        #STA ATTENTO A SPENDERE