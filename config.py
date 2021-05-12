items_info = [{'id':0, 'name':'Mountain Bike'},
         {'id':1, 'name':'Garmin'}]

classes_info = [{'id':0, 'name':'Sport Addicted','color':'r'}, #COMPRA A QUALSIASI PREZZO
                {'id':1, 'name':'Gifter','color':'b'},         #REGALA LA BICI
                {'id':2, 'name':'Amateur','color':'g'},        #FA SPORT OGNI TANTO
                {'id':3, 'name':'Worried','color':'y'}]        #STA ATTENTO A SPENDERE


item1_prices = [1900.0, 1950.0, 2000.0, 2050.0, 2100.0, 2150.0, 2200.0, 2250.0, 2300.0, 2350.0]
item2_prices = [250.0, 270.0, 300.0, 320.0, 350.0, 400.0, 420.0, 450.0, 500.0, 550.0, 600.0, 630.0, 650.0]

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

item1_probabilities =[{'id':0,'name':'Spring-Summer', 'probabilities':[[.80, .72, .55, .53, .5, .45, .4, .35, .3, .3],              #MTB - Sport Addicted
                                                                        [.70, .65, .5, .4, .4, .38, .35, .32, .25, .20],            #MTB - Gifter
                                                                        [.80, .72, .45, .42, .37, .35, .30, .28, .25, .20],         #MTB - Amateur
                                                                        [.80, .72, .38, .35, .3, .15, .12, .1, .1, .08]]},          #MTB - Worried
                      {'id':1,'name':'Autumn', 'probabilities':[[.60, .55, .51, .40, .35, .31, .27, .22, .17, .14],
                                                                 [.65, .60, .55, .35, .31, .27, .23, .19, .17, .16],
                                                                 [.70, .64, .60, .50, .44, .41, .37, .34, .10, .01],
                                                                 [.50, .45, .41, .25, .2, .05, .02, .01, .01, .01]]},
                      {'id':2,'name':'Winter', 'probabilities':[[.65, .60, .57, .52, .49, .37, .25, .21, .15, .09],
                                                                 [.85, .80, .76, .72, .68, .59, .40, .33, .29, .22],
                                                                 [.50, .48, .45, .41, .39, .25, .2, .15, .1, .1],
                                                                 [.5, .48, .18, .15, .12, .1, .01, .01, .01, .01]]}
                    ]
                      

item2_probabilities = [{'id':0,'name':'Spring-Summer', 'probabilities':[[.9, .88, .79, .72, .68, .58, .56 ,.5, .45, .4, .35, .3, .3],    #GARMIN - Sport Addicted ...
                                                                        [.78, .75 ,.7, .61, .57, .5,.42, .4, .38, .35, .31, .25, .20],
                                                                        [.80, .71, .65, .59, .52, .41, .40 ,.37, .35, .30, .28, .28, .20],
                                                                        [.60, .58 ,.54, .5, .45, .38, .35, .3, .15, .12, .1, .1, .08]]},
                       {'id':1,'name':'Autumn', 'probabilities':[[.9, .84, .77, .72, .68, .66, .56 ,.5, .45, .4, .35, .3, .3],    #GARMIN - Sport Addicted ...
                                                                        [.70, .69 ,.65, .64, .62, .59,.57, .4, .38, .35, .31, .25, .20],
                                                                        [.6, .56 , .52, .42, .40, .37, .33 ,.26, .24, .22, .19, .16, .14],
                                                                        [.60, .50 ,.30, .26, .22, .19, .17, .14, .11, .09, .07, .05, .04]]},
                       {'id':2,'name':'Winter', 'probabilities':[[.75, .67, .64, .60, .56, .50, .49 ,.41, .40, .37, .35, .3, .3],    #GARMIN - Sport Addicted ...
                                                                        [.80, .77 ,.74, .70, .67, .59,.49, .47, .39, .35, .31, .25, .20],
                                                                        [.6, .5, .45, .41, .39, .35, .31 ,.21, .18, .14, .07, .03, .01],
                                                                        [.60, .50 ,.30, .26, .22, .19, .17, .14, .11, .09, .07, .05, .04]]},
                    ]
                       

