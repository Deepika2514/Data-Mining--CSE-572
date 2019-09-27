from copy import deepcopy
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure

figure(num=None, figsize=(12, 6), dpi=80, facecolor='w')
from matplotlib import pyplot as plt

df = pd.read_csv(r"C:\Users\Deepika\Downloads\studies\data mining\homework3\Dataset_2.csv",
                 names=['Feature_1', 'Feature_2', 'Class'])

# Getting the values and plotting it
f1 = df['Feature_1'].values
f2 = df['Feature_2'].values
print(pd.DataFrame({'Feature_1': f1, 'Feature_2': f2}))
x = np.array(list(zip(f1, f2)))

list_SSE = []  # List to store Sum of squared errors
tolerance = 0.0000001  # threshold for error between previous and new centroids


def dist(a, b, ax=1):  # function to calculate the distance
    return np.linalg.norm(a - b, axis=ax)


for k in range(2, 3):  # K=number of clusters
    count = 0
    SSE = 0  # SSE=sum of squared errors
    list_random = []
    C_num = np.random.randint(0, len(x) - 1)
    list_random.append(C_num)
    # random generation of indices from array
    while len(list_random) < k:
        C_num = np.random.randint(0, len(x) - 1)
        if (C_num not in list_random):
            list_random.append(C_num)

    C_num_arr = np.array(list_random)
    C_x = [x[i, 0] for i in C_num_arr]  ## X coordinates of random centroids
    C_y = [x[i, 1] for i in C_num_arr]  ### Y coordinates of random centroids
    C = np.array(list(zip(C_x, C_y)), dtype=np.float64)
    # Plot for initial centroid
    plt.scatter(x[:, 0], x[:, 1], c='#050505', s=7)
    plt.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='g')
    plt.ylabel("Feature 2")
    plt.xlabel("Feature 1")
    plt.title("Data values of Dataset_3-Random centroids chosen")
    C_old = np.zeros(C.shape)
    clusters = np.zeros(len(x))
    error = dist(C, C_old, None)  # Error- Distance between new centroids and old centroids
    while abs(error) > tolerance and count != 500:
        for i in range(len(x)):  # Assigning each value to its closest cluster
            distances = dist(x[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        C_old = deepcopy(C)  # Storing the old centroid values

        for i in range(k):  # Finding the new centroids by taking the average value
            points = [x[j] for j in range(len(x)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)

        error = dist(C, C_old, None)
        count = count + 1
    # Plotting
    colors = ['r', 'b', 'g']
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(k):
        points = np.array([x[j] for j in range(len(x)) if clusters[j] == i])
        SSE = (dist(C[i], points, None)) ** 2
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])

    ax.scatter(C[:, 0], C[:, 1], marker='*', s=100, c='#050505')
    plt.ylabel("Feature 2")
    plt.xlabel("Feature 1")
    plt.title("Data values of Dataset_3")





