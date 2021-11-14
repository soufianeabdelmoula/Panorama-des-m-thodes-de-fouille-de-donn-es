#!/usr/bin/env python
# -*- coding: utf-8 -*-
from scipy.io import arff
from skmultilearn.dataset import load_from_arff

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#Question 1 
data=arff.loadarff('supermarket.arff')
supermarket=pd.DataFrame(data[0])
print(data)
# convert categorical values into one-hot vectors and ignore ? values
# corresponding to missing values
supermarket_one_hot = pd.get_dummies(supermarket)
supermarket_one_hot.drop(supermarket_one_hot.filter(regex='_b\'\?\'$',axis=1).columns,axis=1,inplace=True)

#Question2 
items = apriori(supermarket_one_hot, min_support=0.1)
print(items)
# option to show all itemsets
pd.set_option('display.max_colwidth',None)
#Question 3 
ass_rules = association_rules(items, metric="confidence", min_threshold=0.7)
print(ass_rules)

#Question 4

# select rules with more than 2 antecedents
#ass_rules.loc[map(lambda x: len(x)>2,ass_rules['antecedents'])]
rules_4ant = ass_rules.loc[map(lambda x: len(x)==4,ass_rules['antecedents'])]
res = rules_4ant.loc[map(lambda x: len(x)==1,rules_4ant['consequents'])]
print(res)

#question 5

confidence = ass_rules[ass_rules.confidence == ass_rules.confidence.max()]
print(f"resultat de confidence :  \n {confidence}.")

lift = ass_rules[ass_rules.lift == ass_rules.lift.max()]
print(f"resultat de lift :  \n {lift}.")

leverage = ass_rules[ass_rules.leverage == ass_rules.leverage.max()]
print(f"resultat de leverage :   \n {leverage}.")

conviction = ass_rules[ass_rules.conviction == ass_rules.conviction.max()]
print(f"resultat de conviction :  \n {conviction}.")



