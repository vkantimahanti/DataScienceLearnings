#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


df1 = pd.read_csv('StateDrugUtilization2021.csv')


# In[3]:


df1.head()


# In[4]:


print(df1.describe())


# In[5]:


plt.figure(figsize=(10, 6))
sns.distplot(df1['unity_qty'], bins=20, kde=True, color='blue')
plt.title('Distribution of Unity Quantity')
plt.xlabel('Unity Quantity')
plt.ylabel('Frequency')
plt.show()


# In[6]:


plt.figure(figsize=(10, 6))
sns.distplot(df1['customer_rating'], bins=20, kde=True, color='green')
plt.title('Distribution of Customer Ratings')
plt.xlabel('Customer Rating')
plt.ylabel('Frequency')
plt.show()


# In[7]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df1, x='state')
plt.title('Frequency Distribution of States')
plt.xlabel('State')
plt.ylabel('Frequency')
plt.show()


# In[8]:


df1['Date'] = pd.to_datetime(df1['Date'])
df1.set_index('Date', inplace=True)
plt.figure(figsize=(12, 6))
df1['unity_qty'].resample('M').sum().plot()
plt.title('Monthly Unity Quantity')
plt.xlabel('Date')
plt.ylabel('Unity Quantity')
plt.show()


# In[9]:


corr_matrix = df1.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[10]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=df1, x='state', y='unity_qty')
plt.title('Boxplot of Unity Quantity by State')
plt.xlabel('State')
plt.ylabel('Unity Quantity')
plt.show()


# In[11]:


# Bar plot for mean numerical value by categorical variable
mean_unity_qty_by_state = df1.groupby('state')['unity_qty'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=mean_unity_qty_by_state, x='state', y='unity_qty')
plt.title('Mean Unity Quantity by State')
plt.xlabel('State')
plt.ylabel('Mean Unity Quantity')
plt.show()


# In[12]:


# Missing values analysis
missing_values = df1.isnull().sum()
print('Missing Values:\n', missing_values)


# In[13]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=df1[['unity_qty', 'customer_rating']])
plt.title('Boxplot of Numerical Columns')
plt.xlabel('Columns')
plt.ylabel('Values')
plt.show()


# In[19]:


# Drop rows with missing values
df1.dropna(inplace=True)

# Display the dimensions of X and y
X = df1[['DateINT', 'product_name']]
y = df1['unity_qty']

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


# In[17]:


from sklearn.model_selection import train_test_split

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# In[28]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

# Take only 1000 randomly selected rows
df1 = df1.sample(n=1000, random_state=42)

# Data Preprocessing
# Handle missing values
df1.dropna(inplace=True)

# Convert date column to datetime format
df1['Date'] = pd.to_datetime(df1['Date'], format='%m/%d/%Y')

# Define features and target variable
X = df1.drop(['unity_qty'], axis=1)  # Features
y = df1['unity_qty']  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for categorical variables
categorical_cols = ['product_name']

# One-hot encoding for categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('feature_selection', SelectFromModel(RandomForestRegressor())),
                           ('model', model)])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)


# In[29]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Take only 1000 randomly selected rows
df1 = df1.sample(n=1000, random_state=42)

# Data Preprocessing
# Handle missing values
df1.dropna(inplace=True)

# Convert date column to datetime format
df1['Date'] = pd.to_datetime(df1['Date'], format='%m/%d/%Y')

# Define features and target variable
X = df1.drop(['unity_qty'], axis=1)  # Features
y = df1['unity_qty']  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for categorical variables
categorical_cols = ['product_name']

# One-hot encoding for categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Define the model
model = RandomForestRegressor(random_state=42)

# Create a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Define parameter grid for grid search
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions
predictions = best_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)


# In[32]:


pip install lightgbm


# In[54]:


# Convert 'DateINT' column to datetime format
df1['Date'] = pd.to_datetime(df1['DateINT'], errors='coerce')

# Drop rows with NaT (Not a Time) values
df1.dropna(subset=['Date'], inplace=True)

# Extract year, month, day, and dayofweek from the 'Date' column
df1['year'] = df1['Date'].dt.year
df1['month'] = df1['Date'].dt.month
df1['day'] = df1['Date'].dt.day
df1['dayofweek'] = df1['Date'].dt.dayofweek


# In[55]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Convert 'Date' column to datetime format and set it as index
df1['Date'] = pd.to_datetime(df1['Date'], format='%m/%d/%Y')
df1.set_index('Date', inplace=True)

# Check if the data is sorted by date, if not, sort it
if not df1.index.is_monotonic:
    df1 = df1.sort_index()

# Plot the time series data
plt.figure(figsize=(10, 6))
plt.plot(df1.index, df1['unity_qty'], label='Unity Qty')
plt.title('Time Series Plot of Unity Quantity')
plt.xlabel('Date')
plt.ylabel('Unity Qty')
plt.legend()
plt.show()

# Decompose the time series into its components (trend, seasonality, residual)
decomposition = seasonal_decompose(df1['unity_qty'], model='additive', period=12)  # Assuming a seasonal period of 12 (monthly)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plot the decomposed components
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(df1.index, df1['unity_qty'], label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(df1.index, trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(df1.index, seasonal, label='Seasonality')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(df1.index, residual, label='Residuals')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# In[ ]:




