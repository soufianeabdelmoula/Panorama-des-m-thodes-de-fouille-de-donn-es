#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from R_square_clustering import r_square
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#correlation_circle
def correlation_circle(df,nb_var,x_axis,y_axis):
    fig, axes = plt.subplots(figsize=(8,8))
    axes.set_xlim(-1,1)
    axes.set_ylim(-1,1)
    # label with variable names
    for j in range(nb_var):
        # ignore two first columns of df: Nom and Code^Z
        plt.annotate(df.columns[j+2],(corvar[j,x_axis],corvar[j,y_axis]))
    # axes
    plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
    plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
    # add a circle
    cercle = plt.Circle((0,0),1,color='blue',fill=False)
    axes.add_artist(cercle)
    plt.savefig('fig/acp_correlation_circle_axes_'+str(x_axis)+'_'+str(y_axis))
    plt.close(fig)


#Question 1
res = pd.read_csv('Data_World_Development_Indicators2.csv')
#l'affichage des attribut du fichier
"""print(f'Le résultat du fichier est  :  {res.shape}.')
#l'affichage du type
print(f'Les types des attributs sont  : \n {res.dtypes}.') """
#Question 2 
#https://towardsdatascience.com/imputing-missing-values-using-the-simpleimputer-class-in-sklearn-99706afaff46
ValeurManque = SimpleImputer(strategy='median', missing_values=np.nan)
ValeurManque = ValeurManque.fit(res[res.columns[2:]])  
ValeurManque = ValeurManque.transform(res[res.columns[2:]])  
print(f'La valeur manquant sur ligne 6 est colone 3  est :  {ValeurManque[6,3]}.') 
#https://www.it-swarm-fr.com/fr/python/quelquun-peut-il-mexpliquer-standardscaler/829529418/  
#Question 3
scaler = StandardScaler()
DataScaler =scaler.fit_transform(ValeurManque)
print(DataScaler)
print(ValeurManque)
#http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/fr_Tanagra_ACP_Python.pdf
#question 4
acp = PCA(svd_solver='full')
coordonnee = acp.fit_transform(DataScaler)
n = np.size(DataScaler, 0)
p = np.size(DataScaler, 1)
eigval = float(n-1)/n*acp.explained_variance_
fig = plt.figure()
plt.plot(np.arange(1,p+1),eigval)
plt.title("Scree plot")
plt.ylabel("valeur eigen")
plt.xlabel("numero du facteur")
#plt.show()

code_pays = res["Country Code"]

# plot instances on the first plan (first 2 factors)
fig, axes = plt.subplots(figsize=(10,10))
axes.set_xlim(-7,7)               
axes.set_ylim(-7,7)
for j in range(n):
    plt.annotate(code_pays.values[j],(coordonnee[j,0],coordonnee[j,1]))
plt.plot([-7,7],[0,0],color='silver',linestyle='-',linewidth=2)
plt.plot([0,0],[-7,7],color='silver',linestyle='-',linewidth=2)

plt.title("ACP les deux premiére facteur")
plt.ylabel("Facteur 1")
plt.xlabel("Facteurr 2")
#plt.show()


# plot instances on the second plan (3rd and 4th factors)
fig, axes = plt.subplots(figsize=(10,10))
axes.set_xlim(-7,7)
axes.set_ylim(-7,7)
for j in range(n):
    plt.annotate(code_pays.values[j],(coordonnee[j,2],coordonnee[j,3]))
plt.plot([-7,7],[0,0],color='yellow',linestyle='-',linewidth=2)
plt.plot([0,0],[-7,7],color='yellow',linestyle='-',linewidth=2)
plt.title(" les facteurs 3 et 4 de l’ACP")
plt.ylabel("Facteur 3")
plt.xlabel("Facteurr 4")
#plt.show()

# Question 5 
sqrt_eigval  = np.sqrt(eigval)
corvar = np.zeros((p,p))
for i in range(p):
    corvar[:,i] = acp.components_[i,:] * sqrt_eigval [i]
print(f"le résultat est :  {corvar}.")
correlation_circle(res,p,0,1)
correlation_circle(res,p,2,3)
#Question 6 
variation = range(2, 20)
lst = []
for k in variation:
    sf = KMeans(n_clusters=k)
    sf.fit(DataScaler)
    lst.append(r_square(DataScaler, sf.cluster_centers_, sf.labels_, k))  # 7 cluser pck coef trés grand 0.9

fig = plt.figure()
plt.plot(variation,lst,'bx-')
plt.title('k optimal')
#plt.show()

#Quetion 8

from scipy.cluster.hierarchy import dendrogram, linkage

lst_labels = list(map(lambda pair: pair[0]+str(pair[1]), zip(res['Country Name'].values,res.index)))
linkage_matrix = linkage(DataScaler, 'ward')
fig = plt.figure()
dendrogram(
    linkage_matrix,
    labels=lst_labels
)
plt.title('méthode hiérarchique ascendante')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.show()