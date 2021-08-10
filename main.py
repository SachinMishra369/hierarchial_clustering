import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

#reading dataset
dataset=pd.read_csv('Mall_Customers.csv')

#Reading the data that could make a cluster
X=dataset.iloc[:,[3,4]].values
# print(X)

#Using dendogram to find minimum no of clusters
import scipy.cluster.hierarchy as sch

# Z = sch.linkage(X, method='ward')
# dendogram=sch.dendrogram(Z)



#Visualzing Dendogram
# plt.title("Dendogram")
# plt.xlabel("Customer")
# plt.ylabel("Eucilidian Distance")
# plt.show()

# #training Hierachical cluster model on the dataset
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')

hc_y=clustering.fit_predict(X)

#Visualzing clusters

plt.scatter(X[hc_y==0,0],X[hc_y==0,1],s=100,c='red',label='cluster1')


plt.scatter(X[hc_y==1,0],X[hc_y==1,1],s=100,c='green',label='cluster2')

plt.scatter(X[hc_y==2,0],X[hc_y==2,1],s=100,c='blue',label='cluster3')

plt.scatter(X[hc_y==3,0],X[hc_y==3,1],s=100,c='black',label='cluster4')

plt.scatter(X[hc_y==4,0],X[hc_y==4,1],s=100,c='brown',label='cluster5')

# plt.scatter(hc_y.cluster_centers_[:,0],hc_y.cluster_centers_[:,1],s=400,c='cyan',label='Centroid')

plt.title("Customers Clusters")
plt.xlabel("Annual Income k($)")
plt.ylabel("Spending score(1-100)")
plt.legend()
plt.show()