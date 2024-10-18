import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from feature_engine.discretisation import EqualWidthDiscretiser

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

variables = ['MedInc', 'HouseAge', 'AveRooms']
disc = EqualWidthDiscretiser(bins=8, variables=variables, return_boundaries=True)
disc.fit(X_train)
train_t = disc.transform(X_train)
test_t = disc.transform(X_test)

print(X_test)