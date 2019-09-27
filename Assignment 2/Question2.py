# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 23:26:52 2019

@author: Deepika
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 22:48:54 2019

@author: Deepika
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:28:12 2019

@author: Deepika
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 14:32:36 2019

@author: Deepika
"""
from sklearn.metrics.pairwise import cosine_similarity
from copy import deepcopy
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure

figure(num=None, figsize=(12, 6), dpi=80, facecolor='w')
from matplotlib import pyplot as plt

df = pd.read_csv(r"C:\Users\Deepika\Downloads\overdoses.csv")
df['Population'] = df['Population'].str.replace(',', '')
df['Deaths'] = df['Deaths'].str.replace(',', '')
df["Deaths"] = df["Deaths"].apply(float)
df["Population"] = df["Population"].apply(float)

# Getting the values and plotting it
f1 = df['Population'].values
f2 = df['Deaths'].values
x = np.array(list(zip(f1, f2)))
print(x)

x[:, 0] = x[:, 0] / np.max(x[:, 0])
x[:, 1] = x[:, 1] / np.max(x[:, 1])
list_SSE = []
tolerance = 0.0000001


# function to calculate the distance
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


for k in range(2, 16):
    count = 0
    SSE = 0
    list_random = []
    C_num = np.random.randint(0, len(x) - 1)
    list_random.append(C_num)

    # print("List is",list_random)
    # random generation of indices from array
    while len(list_random) < k:
        C_num = np.random.randint(0, len(x) - 1)
        if (C_num not in list_random):
            list_random.append(C_num)

    # print("List is",list_random)
    C_num_arr = np.array(list_random)
    print("C_num_arr for k:", k, C_num_arr)
    ## X coordinates of random centroids
    C_x = [x[i, 0] for i in C_num_arr]
    ### Y coordinates of random centroids
    C_y = [x[i, 1] for i in C_num_arr]
    # C=[x[i]for i in C_num]
    # print("C_y is",C_y)
    C = np.array(list(zip(C_x, C_y)), dtype=np.float64)
    print("C value is for k is", k, C)
    #    plt.scatter(x[:,0], x[:,1], c='#050505', s=7)
    #    plt.scatter(C[:,0], C[:,1], marker='*', s=200, c='g')
    #    plt.ylabel("Deaths")
    #    plt.xlabel('Population')
    # To store the value of centroids when it updates
    C_old = np.zeros(C.shape)
    # print("C_old is",C_old)
    # Cluster Lables(0, 1, 2)
    clusters = np.zeros(len(x))
    # Error func. - Distance between new centroids and old centroids
    error = dist(C, C_old, None)
    # distances = dist(x[0], C)
    # print(error)
    # cluster = np.argmin(distances)

    while abs(error) > tolerance and count != 500:
        # Assigning each value to its closest cluster
        for i in range(len(x)):
            distances = dist(x[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Storing the old centroid values
        # print(clusters)
        C_old = deepcopy(C)
        # print(C_old)
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [x[j] for j in range(len(x)) if clusters[j] == i]

            C[i] = np.mean(points, axis=0)
        # print(C)
        error = dist(C, C_old, None)
        count = count + 1
        # print("error",error)

    # Plotting
    #    colors = ['r', 'g', 'b', 'y', 'c', 'm','coral','chartreuse','deepskyblue','rosybrown','slategray','navy','saddlebrown','darkorange','gray']
    #    fig, ax = plt.subplots(figsize=(12,6))
    #    plt.ylabel("Normalised Deaths")
    #    plt.xlabel('Normalised Population')
    print("clusters for k:", k, clusters)
    for i in range(k):
        points = np.array([x[j] for j in range(len(x)) if clusters[j] == i])
        SSE = (dist(C[i], points, None)) ** 2
        # ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])

    # ax.scatter(C[:, 0], C[:, 1], marker='*', s=100, c='#050505')

    list_SSE.append([SSE, k])
    print("list_SSE with k", list_SSE)
    # For k=5,I need a 50*2 table for row number and  corresponding cluster
    if k is 5:
        k_arr_x = [i for i in range(1, len(x) + 1)]
        # k_arr_x=np.transpose(k_arr_x,axes=None)
        # k_arr_x=np.array(k_arr_x)
        k_arr_y = [clusters[i] for i in range(0, len(x))]
        #    k_arr_y=np.transpose(k_arr_y,axes=None)
        #    k_arr_y=np.array(k_arr_y)
        # print(k_arr_y)

        df = pd.DataFrame({'Row Number': k_arr_x, 'Cluster Number': k_arr_y})
        print(df)

x_coor = [list_SSE[i][1] for i in range(14)]
y_coor = [list_SSE[i][0] for i in range(14)]
plt.plot(x_coor, y_coor)
plt.ylabel("Sum of Squared Errors(J)-After normalising Population and death values")
plt.xlabel("Number of clusters")
plt.title("Graph between J vs k")