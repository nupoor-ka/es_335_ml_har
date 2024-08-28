import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold




np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.

##################################################################

# Q2 a)

#converting to X as 
X_df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
y_series = pd.Series(y, name='Target')

#Normalize the data
scaler = StandardScaler()
X_df_normalized = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns)

#Perform the train-test split
X_train, X_test, y_train, y_test = train_test_split(X_df_normalized, y_series, test_size=0.3, random_state=42)

plt.figure(figsize=(10, 5))

#Plot training
plt.subplot(1, 2, 1)
plt.scatter(X_train['Feature1'], X_train['Feature2'], c=y_train, cmap='viridis')
plt.title('Training Data')

#Plot testing 
plt.subplot(1, 2, 2)
plt.scatter(X_test['Feature1'], X_test['Feature2'], c=y_test, cmap='viridis')
plt.title('Testing Data')

plt.show()

print("X_train")
print(X_train)
print()

print("y_train")
print(y_train)
print()

#since this case is real input, discrete output, we shall use "entropy and gini index" as the criterion.
#creating copies
X_train_copy = X_train.copy()
y_train_copy = y_train.copy()
X_test_copy = X_test.copy()
y_test_copy = y_test.copy()

print("Running the tree with max depth = 5 (default value)\n")

for criteria in ["entropy","gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X_train_copy, y_train_copy)
    y_hat = tree.predict(X_test_copy)
    tree.plot()
    print("Criteria :", criteria)
    print()
    print("Accuracy: ", accuracy(y_hat, y_test_copy))
    
    for cls in y_test_copy.unique():
        print("Precision: ", precision(y_hat, y_test_copy, cls))
        print("Recall: ", recall(y_hat, y_test_copy, cls))

##################################################################

# Q2 b)

depths = list(range(1, 21))

#KFold for cross-validation
kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_depth(depth, X_train, y_train, X_val, y_val):
    """
    Evaluate the decision tree with a specific depth on the validation set.
    """
    tree = DecisionTree(max_depth=depth, criterion='entropy')  # or 'gini_index'
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_val)
    return accuracy(y_val, y_pred)

best_depth = None
best_score = -1

# Nested cross-validation
for depth in depths:
    scores = []
    
    # Inner loop for parameter tuning
    for train_index, val_index in kf_outer.split(X_df_normalized):
        X_train, X_val = X_df_normalized.iloc[train_index], X_df_normalized.iloc[val_index]
        y_train, y_val = y_series.iloc[train_index], y_series.iloc[val_index]
        
        # Evaluate the performance for this depth
        score = evaluate_depth(depth, X_train, y_train, X_val, y_val)
        scores.append(score)
    
    mean_score = np.mean(scores)
    print(f"Depth: {depth}, Mean Accuracy: {mean_score}")

    if mean_score > best_score:
        best_score = mean_score
        best_depth = depth

print("Optimal Depth:", best_depth)
print("Best Mean Accuracy:", best_score)

# Optional: Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(depths, [np.mean([evaluate_depth(d, X_df_normalized.iloc[train_index], y_series.iloc[train_index], X_df_normalized.iloc[val_index], y_series.iloc[val_index]) for train_index, val_index in kf_outer.split(X_df_normalized)]) for d in depths], marker='o',color="green")
plt.xlabel('Tree Depth')
plt.ylabel('Mean Accuracy')
plt.title('Tree Depth vs. Mean Accuracy')
plt.show()