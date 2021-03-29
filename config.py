items = [{'id':0, 'name':'Mountain Bike'},
         {'id':1, 'name':'Garmin'}]

item1_prices = [2060.0,2200.0,2290.0,2400.0]
item2_prices = [420.0,500.0,560.0,630.0]

item1_probabilities =[{'id':0,'name':'Spring-Summer', 'probabilities':[[.6,.6,.5,.5],   #MTB - Sport Addicted
                                                                 [.5,.4,.4,.3],         #MTB - Gifter
                                                                 [.5,.4,.4,.3],         #MTB - Amateur
                                                                 [.6,.3,.3,.1]]},       #MTB - Worried
                      {'id':1,'name':'Autumn', 'probabilities':[[.6,.6,.5,.5],
                                                                 [.4,.3,.3,.2],
                                                                 [.4,.3,.3,.2],
                                                                 [.5,.2,.2,.1]]},
                      {'id':2,'name':'Winter', 'probabilities':[[.4,.4,.4,.3],
                                                                 [.5,.45,.4,.25],
                                                                 [.2,.15,.12,.1],
                                                                 [.3,.20,.13,.1]]}
                    ]
                      

item2_probabilities = [{'id':0,'name':'Spring-Summer', 'probabilities':[[.45,.45,.4,.35],    #GARMIN - Sport Addicted ...
                                                                  [.2,.15,.13,.08],
                                                                  [.35,.25,.2,.15],
                                                                  [.4,.15,.1,.05]]},
                       {'id':1,'name':'Autumn', 'probabilities':[[.45,.45,.4,.35],
                                                                  [.15,.12,.12,.09],
                                                                  [.2,.15,.15,.1],
                                                                  [.3,.1,.1,.05]]},
                       {'id':2,'name':'Winter', 'probabilities':[[.55,.53,.5,.48],
                                                                  [.3,.25,.2,.18],
                                                                  [.22,.17,.17,.1],
                                                                  [.4,.2,.15,.15]]},
                    ]
                       

classes_info = [{'id':0, 'name':'Sport Addicted','color':'r'}, #COMPRA A QUALSIASI PREZZO
                {'id':1, 'name':'Gifter','color':'b'},         #REGALA LA BICI
                {'id':2, 'name':'Amateur','color':'g'},        #FA SPORT OGNI TANTO
                {'id':3, 'name':'Worried','color':'y'}]        #STA ATTENTO A SPENDERE