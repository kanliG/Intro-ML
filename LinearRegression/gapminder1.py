# TODO: Add import statements

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
# TODO: Add import statements

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")

# Make and fit the linear regression model
# TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
x = bmi_life_data[['BMI']]
y = bmi_life_data[['Life expectancy']]
bmi_life_model.fit(x, y)

# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
# pred = bmi_life_model.predict(10)
laos_life_exp = bmi_life_model.predict([[21.07931]])
print(laos_life_exp)
