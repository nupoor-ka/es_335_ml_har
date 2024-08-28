import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py) and Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs

def rr(N,P):
    times_fit=[]
    times_predict=[]
    for i in N:
        for j in P:
            X = pd.DataFrame(np.random.randn(i, j))
            y = pd.Series(np.random.randn(i))
            for criteria in ["squared_loss"]:
                tf1=time.time()
                tree = DecisionTree(criterion=criteria) # Split based on Inf. Gain
                tree.fit(X, y)  
                tf2=time.time()
                times_fit.append(tf2-tf1)
                tp1=time.time()
                y_hat = tree.predict(X)
                tp2=time.time()
                times_predict.append(tp2-tp1)

    plot_runtime(times_fit, times_predict, N, P)

def rd(N,P):
    times_fit=[]
    times_predict=[]
    for i in N:
        for j in P:
            X = pd.DataFrame(np.random.randn(i, j))
            y = pd.Series(np.random.randint(2, size=i), dtype="category")
            for criteria1 in ["entropy", "gini_index"]:
                tf1=time.time()
                tree = DecisionTree(criterion=criteria1)  # Split based on Inf. Gain
                tree.fit(X, y)
                tf2=time.time()
                times_fit.append(tf2-tf1)
                tp1=time.time()
                y_hat = tree.predict(X)
                tp2=time.time()
                times_predict.append(tp2-tp1)
                tree.plot()
                print("Criteria :", criteria1)
                print("Accuracy: ", accuracy(y_hat, y))
                for cls in y.unique():
                    print("Precision: ", precision(y_hat, y, cls))
                    print("Recall: ", recall(y_hat, y, cls))
    times_fit_e=times_fit[:9]
    times_fit_g=times_fit[9:]
    times_predict_e=times_predict[:9]
    times_predict_g=times_predict[9:]
    print('For Entropy')
    plot_runtime(times_fit_e, times_predict_e, N, P)
    print('For Gini')
    plot_runtime(times_fit_g, times_predict_g, N, P)


def dr(N,P):
    # Discrete Input and Real Output

    times_fit=[]
    times_predict=[]
    N = [7,30,45,75]
    P = [2,5,35,50]
    for i in N:
        for j in P:
            X = pd.DataFrame({k: pd.Series(np.random.randint(2, size=i), dtype="category") for k in range(5)})
            y = pd.Series(np.random.randn(i))

            for criteria in ["mse"]:
                tf1=time.time()
                tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
                tree.fit(X, y)
                tf2=time.time()
                times_fit.append(tf2-tf1)
                tp1=time.time()
                y_hat = tree.predict(X)
                tp2=time.time()
                times_predict.append(tp2-tp1)
                tree.plot()
                print("Criteria :", criteria)
                print("RMSE: ", rmse(y_hat, y))
                print("MAE: ", mae(y_hat, y))

    plot_runtime(times_fit, times_predict, N, P)



# Function to plot the results
def plot_runtime(times_fit, times_predict, N, P):
    # Reshaping times to match the dimensions of N x P
    times_fit = np.array(times_fit).reshape(len(N), len(P))
    times_predict = np.array(times_predict).reshape(len(N), len(P))

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # for Plotting fitting times
    for i, n in enumerate(N):
        axs[0].plot(P, times_fit[i], marker='o', label=f'N = {n}')
    axs[0].set_title('Training Time vs. Number of Features')
    axs[0].set_xlabel('Number of Features (P)')
    axs[0].set_ylabel('Training Time (seconds)')
    axs[0].legend()

    # for Plotting prediction times
    for i, n in enumerate(N):
        axs[1].plot(P, times_predict[i], marker='o', label=f'N = {n}')
    axs[1].set_title('Prediction Time vs. Number of Features')
    axs[1].set_xlabel('Number of Features (P)')
    axs[1].set_ylabel('Prediction Time (seconds)')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# Run the functions, Learn the DTs and Show the results/plots


N = [7,30,45,75]
P = [2,5,35,50]
print('Real Input Real Output\n')
rr(N,P)
print('Real Input Discrete Output\n')
rd(N,P)
print('Discrete Input Real Output\n')
dr(N,P)