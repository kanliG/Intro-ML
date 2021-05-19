# Import statements
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y.
X = data[:, 0:2]
y = data[:, 2]

# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.
sz_depth = 20
sz_leaf = 20
acc = np.zeros(shape=(sz_depth, sz_leaf), dtype=float)
print(acc)
for depth in range(1, sz_depth, 1):
    for leaf in range(1, sz_leaf, 1):
        model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf)

        # TODO: Fit the model.
        model.fit(X, y)

        # TODO: Make predictions. Store them in the variable y_pred.
        y_pred = model.predict(X)

        # TODO: Calculate the accuracy and assign it to the variable acc.
        a = accuracy_score(y, y_pred)
        print('Depth: ', depth, ' - leaf: ', leaf, ' = Acc: ', a)
        acc[depth - 1, leaf - 1] = a

max_val = np.max(acc)
print('Max', max_val)
result = np.where(acc == max_val)
print('Tuple of arrays returned : ', result)
print('List of coordinates of maximum value in Numpy array : ')
# zip the 2 arrays to get the exact coordinates
list_cordinates = list(zip(result[0], result[1]))
# travese over the list of cordinates
for cord in list_cordinates:
    print(cord)