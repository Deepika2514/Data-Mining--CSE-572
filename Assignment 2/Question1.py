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
print(pd.DataFrame({'Population': f1, 'Deaths': f2}))
x = np.array(list(zip(f1, f2)))

x[:, 0] = x[:, 0] / np.max(x[:, 0])  # Normalising Population values
x[:, 1] = x[:, 1] / np.max(x[:, 1])  # Normalising Death values
list_SSE = []  # List to store Sum of squared errors
tolerance = 0.0000001  # threshold for error between previous and new centroids


def dist(a, b, ax=1):  # function to calculate the distance
    return np.linalg.norm(a - b, axis=ax)


for k in range(2, 16):  # K=number of clusters
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

    for i in range(k):
        points = np.array([x[j] for j in range(len(x)) if clusters[j] == i])
        SSE = (dist(C[i], points, None)) ** 2

    list_SSE.append([SSE, k])

    if k is 5:
        k_arr_x = [i for i in range(1, len(x) + 1)]
        k_arr_y = [clusters[i] for i in range(0, len(x))]
        df = pd.DataFrame({'Row Number': k_arr_x, 'Cluster Number': k_arr_y})
        print(df)

x_coor = [list_SSE[i][1] for i in range(14)]
y_coor = [list_SSE[i][0] for i in range(14)]
plt.plot(x_coor, y_coor)
plt.text(9, .5, 'Normalised Population=Population/max(Population)')
plt.text(9, .45, 'Normalised Deaths=Deaths/max(Deaths)')
plt.ylabel("Sum of Squared Errors(J)-After normalising Population and death values")
plt.xlabel("Number of clusters")
plt.title("Graph between J vs k")