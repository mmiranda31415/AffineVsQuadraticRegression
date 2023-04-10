#!/usr/bin/env python
# coding: utf-8

# In[83]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

scaler = StandardScaler()


# In[212]:


class LogisticRegression:
    """
    This class implements a logistic regression algorithm with a quadratic hypothesis in the argument of the sigmoid function
    """
    def __init__(self, lr=0.01, num_iter=100000, reg=0, tol=1e-3):
        self.lr = lr
        self.num_iter = num_iter
        self.reg = reg
        self.tol = tol
        self.w = None
        self.b = None
        self.M = None

    def sigmoid(self, z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z))) #to avoid overflow errors
    
    def quadratic_function( self, x ):
        return self.b + self.w @ x + (self.M @ x) @ x
        
            
    def fit(self, X, y):
        # Initialize weights, bias and the quadratic matrix M
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.M = np.zeros((X.shape[1], X.shape[1]))
        
        X = scaler.fit_transform(X)
        
        X = np.array([np.array(x) for x in X])
        
        # Gradient descent
        for i in range(self.num_iter):
        
            z = np.array([self.quadratic_function(x) for x in X])
            
            h = self.sigmoid(z)
            
            difference = h - y
            
            grad_b = np.sum( difference )
            
            grad_w = ( np.array( [ difference @ xt for xt in X.T ] )  - 2 * self.reg * self.w + 2 * self.M @ self.w  ) / X.shape[0]# + self.reg * self.w
                         
            grad_M = X.T @ np.diag( h * ( 1 - h ) ) @ X - 2 * self.reg * self.M
            
            self.b -= self.lr * grad_b
            self.w -= self.lr * grad_w
            self.M += self.lr * grad_M
            
            # Check for convergence
            if np.max(np.abs(self.lr * grad_w)) < self.tol:
                break

    def predict_prob(self, X):

        h = np.array( [ self.sigmoid( self.quadratic_function(x) ) for x in X ] )
        return h

    def predict(self, X, threshold=0.5):
        return np.array(self.predict_prob(X) >= threshold).astype(int)


# In[214]:


n_samples = 1000
X = np.random.normal(size=(n_samples, 2))
y_nonlinear = lambda X : 2 + 3*X[:,0] + 4*X[:,1] + 0.5*X[:,0]**2 + 0.8*X[:,1]**2
y_prob = 1 / (1 + np.exp(-y_nonlinear(X)))
y = np.random.binomial(n=1, p=y_prob)


# Fit logistic regression model
model = LogisticRegression(reg=0.1)
model.fit(X, y)

# Predict labels for new data
X_test = np.random.normal(size=(10, 2))
y_pred = model.predict(X_test)
y_prob_test = 1 / (1 + np.exp(-y_nonlinear(X_test)))
y_test = np.random.binomial(n=1, p=y_prob_test)
# Print coefficients and predicted labels
print(f"Bias: {model.b}")
print(f"Coefficients: {model.w}")
print(f"Quadratic Coefficients: \n{model.M}")
print(f"Predicted labels: {y_pred}")
print(f"Real labels: {y_test}")

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion matrix:\n{confusion_mat}')


# In[215]:


# Predict labels for new, more numerous data

X_test = np.random.normal(size=(10000, 2))
y_pred = model.predict(X_test)
y_prob_test = 1 / (1 + np.exp(-y_nonlinear(X_test)))
y_test = np.random.binomial(n=1, p=y_prob_test)
# Print coefficients and predicted labels
print(f"Bias: {model.b}")
print(f"Coefficients: {model.w}")
print(f"Quadratic Coefficients: \n{model.M}")

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion matrix:\n{confusion_mat}')


# In[218]:


class LogisticRegressionAffine:
    """
    This class implements a logistic regression algorithm with an affine hypothesis in the argument of the sigmoid function
    """
    def __init__(self, lr=0.01, num_iter=100000, reg=0, tol=1e-3):
        self.lr = lr
        self.num_iter = num_iter
        self.reg = reg
        self.tol = tol
        self.w = None
        self.b = None

    def sigmoid(self, z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z))) #to avoid overflow errors
    
    def affine_function( self, x ):
        return self.b + self.w @ x 
        
            
    def fit(self, X, y):
        # Initialize weights, bias
        self.w = np.zeros(X.shape[1])
        self.b = 0
        self.M = np.zeros((X.shape[1], X.shape[1]))
        
        X = scaler.fit_transform(X)
        
        X = np.array([np.array(x) for x in X])
        
        # Gradient descent
        for i in range(self.num_iter):
        
            z = np.array([self.affine_function(x) for x in X])
            
            h = self.sigmoid(z)
            
            difference = h - y
            
            grad_b = np.sum( difference )
            
            grad_w = ( np.array( [ difference @ xt for xt in X.T ] )  - 2 * self.reg * self.w + 2 * self.M @ self.w  ) / X.shape[0]# + self.reg * self.w
            
            self.b -= self.lr * grad_b
            self.w -= self.lr * grad_w
            
            # Check for convergence
            if np.max(np.abs(self.lr * grad_w)) < self.tol:
                break

    def predict_prob(self, X):

        h = np.array( [ self.sigmoid( self.affine_function(x) ) for x in X ] )
        return h

    def predict(self, X, threshold=0.5):
        return np.array(self.predict_prob(X) >= threshold).astype(int)


# In[219]:


n_samples = 1000
X = np.random.normal(size=(n_samples, 2))
y_nonlinear = lambda X : 2 + 3*X[:,0] + 4*X[:,1] + 0.5*X[:,0]**2 + 0.8*X[:,1]**2
y_prob = 1 / (1 + np.exp(-y_nonlinear(X)))
y = np.random.binomial(n=1, p=y_prob)


# Fit logistic regression model
model_affine = LogisticRegressionAffine(reg=0.1)
model_affine.fit(X, y)

# Predict labels for new data
X_test = np.random.normal(size=(10, 2))
y_pred = model_affine.predict(X_test)
y_prob_test = 1 / (1 + np.exp(-y_nonlinear(X_test)))
y_test = np.random.binomial(n=1, p=y_prob_test)
# Print coefficients and predicted labels
print(f"Bias: {model_affine.b}")
print(f"Coefficients: {model_affine.w}")
print(f"Predicted labels: {y_pred}")
print(f"Real labels: {y_test}")

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion matrix:\n{confusion_mat}')


# In[220]:


# Predict labels for new, more numerous data

X_test = np.random.normal(size=(10000, 2))
y_pred = model_affine.predict(X_test)
y_prob_test = 1 / (1 + np.exp(-y_nonlinear(X_test)))
y_test = np.random.binomial(n=1, p=y_prob_test)
# Print coefficients and predicted labels
print(f"Bias: {model_affine.b}")
print(f"Coefficients: {model_affine.w}")

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion matrix:\n{confusion_mat}')


# In[222]:


from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X, y)

X_test = np.random.normal(size=(10, 2))
y_pred = log_reg.predict(X_test)
y_prob_test = 1 / (1 + np.exp(-y_nonlinear(X_test)))
y_test = np.random.binomial(n=1, p=y_prob_test)


# Use the trained model to make predictions on the test data
y_pred = log_reg.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion matrix:\n{confusion_mat}')


# In[223]:


X_test = np.random.normal(size=(10000, 2))
y_pred = model.predict(X_test)
y_prob_test = 1 / (1 + np.exp(-y_nonlinear(X_test)))
y_test = np.random.binomial(n=1, p=y_prob_test)

y_pred = log_reg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion matrix:\n{confusion_mat}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




