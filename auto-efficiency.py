import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tree.base import DecisionTree
from tree.utils import *
from metrics import *

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn


"""
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
       'acceleration', 'model year', 'origin', 'car name']

Target => mpg (real value)

input => contd.  -> displacement, horsepower, weight, acceleration.
         integer -> cylinders, model year, origin 
         others  -> car name (no use) 

         
"""

##################################################################

# Q3 a)

#firstly we will one hot encode the rows which are discrete 

columns_to_encode = ['cylinders', 'model year', 'origin']

data_encoded = one_hot_encoding(data, columns=columns_to_encode)

#printing the encoded dataframe. 
print(data_encoded)

#dropping redundant row and rectifying the different datatype
data_encoded = data_encoded.drop(columns=['car name'])

data_encoded['horsepower'] = pd.to_numeric(data_encoded['horsepower'], errors='coerce')

#denoting target
X = data_encoded.drop(columns=['mpg'])
y = data_encoded['mpg']

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40) # 80-20 split 

# Display the results
print("X_train:")
print(X_train)
print("\nX_test:")
print(X_test)
print("\ny_train:")
print(y_train)
print("\ny_test:")
print(y_test)

print("#"*25)
print("Self coded tree => Case: Real Input and Real Output (depth = 5)\n")

print(data_encoded.dtypes)

X_train_copy = X_train.copy()
y_train_copy = y_train.copy()
X_test_copy = X_test.copy()
y_test_copy = y_test.copy()

for criteria in ["squared_loss"]:
    tree = DecisionTree(criterion=criteria, case_tree="rr")  # Split based on Inf. Gain
    tree.fit(X_train_copy, y_train_copy)
    y_hat = tree.predict(X_test_copy)
    tree.plot()
    print("Criteria :", criteria)
    print()
    print("RMSE: ", rmse(y_hat, y_test_copy))
    print("MAE: ", mae(y_hat, y_test_copy))
print()
print(y_train_copy)

# Q3 b) 

print("#"*25)
print("Using sklearn's DecisionTreeRegressor (depth = 5) \n")


sklearn_tree = DecisionTreeRegressor(random_state=42, max_depth=5) #maintaing the same depth as our model 

sklearn_tree.fit(X_train, y_train)

y_hat_sklearn = sklearn_tree.predict(X_test)

print("RMSE (sklearn):", rmse(y_test, y_hat_sklearn))
print("MAE (sklearn):", mae(y_test, y_hat_sklearn))

print("#"*25)
print("Comparative analysis with depth variation \n")


# Initialize lists to store metrics at different depths
depths = range(1, 6)  # For example, testing depths from 1 to 20
rmse_custom_tree = []
mae_custom_tree = []
rmse_sklearn_tree = []
mae_sklearn_tree = []

X_train_copy = X_train.copy()
y_train_copy = y_train.copy()
X_test_copy = X_test.copy()
y_test_copy = y_test.copy()

# Loop over depths and calculate metrics for both your custom tree and sklearn tree
for depth in depths:
       # Custom Decision Tree
       tree = DecisionTree(criterion="squared_loss", max_depth=depth, case_tree="rr")
       tree.fit(X_train_copy, y_train_copy)
       y_hat_custom = tree.predict(X_test_copy)
       rmse_custom_tree.append(rmse(y_hat_custom, y_test_copy))
       mae_custom_tree.append(mae(y_hat_custom, y_test_copy))

       # Sklearn Decision Tree
       sklearn_tree = DecisionTreeRegressor(random_state=42, max_depth=depth)
       sklearn_tree.fit(X_train, y_train)
       y_hat_sklearn = sklearn_tree.predict(X_test)
       rmse_sklearn_tree.append(rmse(y_hat_sklearn, y_test))
       mae_sklearn_tree.append(mae(y_hat_sklearn, y_test))
       print(y_train_copy)


# Convert the results to a DataFrame for easier plotting
results_df = pd.DataFrame({
    'Depth': list(depths) * 2,
    'RMSE': rmse_custom_tree + rmse_sklearn_tree,
    'MAE': mae_custom_tree + mae_sklearn_tree,
    'Model': ['Custom Tree'] * len(depths) + ['Sklearn Tree'] * len(depths)
})

# Plot using seaborn joint plot
plt.figure(figsize=(10, 6))

# Plot RMSE and MAE vs Depth for both models
sns.lineplot(data=results_df, x='Depth', y='RMSE', hue='Model', marker='o')
plt.title('RMSE vs Depth for Custom Tree vs Sklearn Tree')
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(data=results_df, x='Depth', y='MAE', hue='Model', marker='o')
plt.title('MAE vs Depth for Custom Tree vs Sklearn Tree')
plt.show()

