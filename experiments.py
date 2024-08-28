import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
import sklearn as skl
from sklearn import tree

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py) and Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs

def rr(N,P):
    times_fit=[]
    times_predict=[]
    theo_fit=[]
    theo_predict=[]
    for i in N:
        for j in P:
            X = pd.DataFrame(np.random.randn(i, j))
            y = pd.Series(np.random.randn(i))
            for criteria in ["squared_loss"]:
                tf1=time.time()
                tree1 = DecisionTree(criterion=criteria) # Split based on Inf. Gain
                tree1.fit(X, y)  
                tf2=time.time()
                times_fit.append(tf2-tf1)
                tp1=time.time()
                y_hat = tree1.predict(X)
                tp2=time.time()
                times_predict.append(tp2-tp1)
                tt1=time.time()
                sk_dt=tree.DecisionTreeRegressor(criterion='squared_error')
                sk_dt.fit(X,y)
                tt2=time.time()
                theo_fit.append(tt2-tt1)
                ttp1=time.time()
                yth=sk_dt.predict(X)
                ttp2=time.time()
                theo_predict.append(ttp2-ttp1)
    plot_runtime(times_fit, times_predict,theo_fit,theo_predict, N, P,"squared_loss")

def rd(N,P):
    times_fit=[]
    times_predict=[]
    theo_fit=[]
    theo_predict=[]
    for i in N:
        for j in P:
            X = pd.DataFrame(np.random.randn(i, j))
            y = pd.Series(np.random.randint(2, size=i), dtype="category")
            for criteria1 in ["entropy", "gini_index"]:
                tf1=time.time()
                tree2 = DecisionTree(criterion=criteria1)  # Split based on Inf. Gain
                tree2.fit(X, y)
                tf2=time.time()
                times_fit.append(tf2-tf1)
                tp1=time.time()
                y_hat = tree2.predict(X)
                tp2=time.time()
                times_predict.append(tp2-tp1)
                tt1=time.time()
                if criteria1=='gini_index':
                    criteria='gini'
                else:
                    criteria=criteria1
                sk_dt=tree.DecisionTreeClassifier(criterion=criteria)
                sk_dt.fit(X,y)
                tt2=time.time()
                theo_fit.append(tt2-tt1)
                ttp1=time.time()
                yth=sk_dt.predict(X)
                ttp2=time.time()
                theo_predict.append(ttp2-ttp1)
    times_fit_e=times_fit[:16]
    times_fit_g=times_fit[16:]
    times_predict_e=times_predict[:16]
    times_predict_g=times_predict[16:]
    theo_fit_e=theo_fit[:16]
    theo_fit_g=theo_fit[16:]
    theo_predict_e=theo_predict[:16]
    theo_predict_g=theo_predict[16:]
    plot_runtime(times_fit_e, times_predict_e, theo_fit_e,theo_predict_e,  N, P,"Entropy")
    plot_runtime(times_fit_g, times_predict_g,theo_fit_g,theo_predict_g, N, P,"Gini")


def dr(N,P):
    # Discrete Input and Real Output

    times_fit=[]
    times_predict=[]
    theo_fit=[]
    theo_predict=[]
    N = [7,30,45,75]
    P = [2,5,35,50]
    for i in N:
        for j in P:
            X = pd.DataFrame({k: pd.Series(np.random.randint(2, size=i), dtype="category") for k in range(5)})
            y = pd.Series(np.random.randn(i))

            for criteria in ["mse"]:
                tf1=time.time()
                tree3 = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
                tree3.fit(X, y)
                tf2=time.time()
                times_fit.append(tf2-tf1)
                tp1=time.time()
                y_hat = tree3.predict(X)
                tp2=time.time()
                times_predict.append(tp2-tp1)
                tt1=time.time()
                sk_dt=tree.DecisionTreeRegressor(criterion='friedman_mse')
                sk_dt.fit(X,y)
                tt2=time.time()
                theo_fit.append(tt2-tt1)
                ttp1=time.time()
                yth=sk_dt.predict(X)
                ttp2=time.time()
                theo_predict.append(ttp2-ttp1)

    plot_runtime(times_fit, times_predict,theo_fit,theo_predict, N, P,"mse")



# Function to plot the results

def plot_runtime(times_fit, times_predict, theo_fit, theo_predict, N, P, a):
    # Reshaping times and theoretical values to match the dimensions of N x P
    times_fit = np.array(times_fit).reshape(len(N), len(P))
    times_predict = np.array(times_predict).reshape(len(N), len(P))
    theo_fit = np.array(theo_fit).reshape(len(N), len(P))
    theo_predict = np.array(theo_predict).reshape(len(N), len(P))

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plotting fitting times
    for i, n in enumerate(N):
        axs[0].plot(P, times_fit[i], marker='o', label=f'Actual N = {n}')
        axs[0].plot(P, theo_fit[i], linestyle='--', label=f'Theoretical N = {n}')
    axs[0].set_title(f'Training Time vs. Number of Features for {a}')
    axs[0].set_xlabel('Number of Features (P)')
    axs[0].set_ylabel('Training Time (seconds)')
    axs[0].legend()

    # Plotting prediction times
    for i, n in enumerate(N):
        axs[1].plot(P, times_predict[i], marker='o', label=f'Actual N = {n}')
        axs[1].plot(P, theo_predict[i], linestyle='--', label=f'Theoretical N = {n}')
    axs[1].set_title(f'Prediction Time vs. Number of Features for {a}')
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