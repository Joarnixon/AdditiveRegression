import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings
import time

"""
A class that implements an additive logistic regression model.

Parameters:
k (int): The number of features in each column subspace. Default is 2.
n (int): The number of subspaces to generate when using random columns. Default is 2.
random (bool): True - generate random subspaces. False - sequential subspaces. Default is True.

Attributes:
model (LogisticRegression): An instance of the LogisticRegression class from the sklearn.linear_model module.
intercept_ (ndarray): The intercept of the fitted additive logistic regression model.
coef_ (ndarray): The coefficients of the fitted additive logistic regression model.
classes_ (ndarray): The classes of the target variable.
"""
class ALogisticRegression:
    """
    A class that implements an additive logistic regression model.

    Parameters:
    k (int): The number of features in each column subspace. Default is 2.
    n (int): The number of subspaces to generate when using random columns. Default is 2.

    Attributes:
    model (LogisticRegression): An instance of the LogisticRegression class from the sklearn.linear_model module.
    intercept_ (ndarray): The intercept of the fitted additive logistic regression model.
    coef_ (ndarray): The coefficients of the fitted additive logistic regression model.
    classes_ (ndarray): The classes of the target variable.
    """

    def __init__(self, k=2, n=2, **kwargs):
        self.k = k
        self.n = n
        self.kwargs = kwargs
        self.model = LogisticRegression(**kwargs)
        self.intercept_ = None
        self.coef_ = None
        self.classes_ = None
        self.time_evaluation = None

    def __subspace_generator(self, X, shape, n):
        base_cols = shape // n
        remaining_cols = shape % n

        col_sizes = [base_cols + 1 if i < remaining_cols else base_cols for i in range(n)]

        start = 0
        for size in col_sizes:
            yield X[:, start:start+size]
            start += size

    def fit(self, X, y):
        """
        Fits the additive logistic regression model on the input data and target variable.

        Parameters:
        X (ndarray): The input data.
        y (ndarray): The target variable.
        """
        start = time.time()
        X = np.array(X)
        y = np.ravel(y)
        shape = X.shape[1]
        if self.k >= X.shape[1]:
            warnings.warn('Number of features is less than number of models. Fitting regular logistic regression instead')

        for X_subspace in self.__subspace_generator(X, shape, self.k):
            model = LogisticRegression(**self.kwargs)
            model.fit(X_subspace, y)
            if self.coef_ is None:
                self.intercept_ = model.intercept_
                self.coef_ = model.coef_
                self.classes_ = model.classes_
            else:
                self.intercept_ += model.intercept_
                self.coef_ = np.concatenate((self.coef_, model.coef_), axis=1)

            del model
        
        self.model.intercept_ = self.intercept_
        self.model.coef_ = self.coef_
        self.model.classes_ = self.classes_

        end = time.time()
        self.time_evaluation = end - start

    def predict(self, X):
        """
        Makes predictions using the fitted additive logistic regression model on the input data.

        Parameters:
        X (ndarray): The input data.

        Returns:
        ndarray: The predicted classes.
        """
        return self.model.predict(X)

    def track_time(self, X, y, verbose=1):
        """
        Tracks the time taken to fit a regular logistic regression model and the additive logistic regression model on the input data and target variable.

        Parameters:
        X (ndarray): The input data.
        y (ndarray): The target variable.
        random (bool): Determines whether to generate random subspaces or sequential subspaces. Default is False.
        verbose (int): Whether to provide time difference inline. Takes 0 as no verbose and 2 as maximum verbose. Default is 1.

        Returns:
        tuple: The time taken for the regular logistic regression model and the additive logistic regression model separately.
        """
        if self.coef_ is None:
            raise ValueError("Model coefficients are not initialized. Please fit the model first.")
        else:
            start = time.time()
            X = np.array(X)
            y = np.ravel(y)
            LogisticRegression(**self.kwargs).fit(X, y)
            end = time.time()
            full_evaluation = end - start

            if verbose == 2:
                print(f'Time for standard LogisticRegression on the whole dataset: {full_evaluation}')
                print(f'Time for additive LogisticRegression on the partial datasets: {self.time_evaluation}')
                print(f'Difference in: {full_evaluation/self.time_evaluation}')
            elif verbose == 1:
                print(f'Difference in: {full_evaluation/self.time_evaluation}')
            return full_evaluation, self.time_evaluation