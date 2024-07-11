import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import ElasticNet
from scipy.optimize import minimize_scalar
 
# Show app title and description
st.set_page_config(page_title="Optimal Price", page_icon="")
st.title("Optimal Price for Luxury Fashion Brands")
 
# Define form for user inputs
with st.form("input_form"):
    brand_name = st.selectbox("Select a Brand", ['Gucci', 'Burberry', 'Prada', 'Hermes', 'Ralph Lauren'])
    cost = st.number_input("Production Cost", min_value=0.0, step=0.01)
    competitor_price = st.number_input("Competitor Price", min_value=0.0, step=0.01)
    submitted = st.form_submit_button("Submit")
 
# Generate random data
np.random.seed(42)
data = {
    'BrandName': np.random.choice(['Gucci', 'Burberry', 'Prada', 'Hermes', 'Ralph Lauren'], 100),
    'ItemNumber': np.random.randint(1, 11, 100),
    'ProductionCost': np.random.uniform(50, 9000, 100),
    'Price': np.random.uniform(100, 9000, 100),
    'CompetitorPrice': np.random.uniform(100, 9000, 100),
    'Demand': np.random.poisson(lam=10, size=100)
}
df = pd.DataFrame(data)
 
# Define features and target
categorical_features = ['BrandName', 'ItemNumber']
numeric_features = ['ProductionCost', 'Price', 'CompetitorPrice']
target_feature = 'Demand'
 
X = df[categorical_features + numeric_features]
y = df[target_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(categories='auto', drop='first', sparse_output=False)
 
# Fit transformers on training data
X_train_numeric = numeric_transformer.fit_transform(X_train[numeric_features])
X_train_categorical = categorical_transformer.fit_transform(X_train[categorical_features])
 
X_train_transformed = np.hstack((X_train_numeric, X_train_categorical))
X_test_numeric = numeric_transformer.transform(X_test[numeric_features])
X_test_categorical = categorical_transformer.transform(X_test[categorical_features])
X_test_transformed = np.hstack((X_test_numeric, X_test_categorical))
 
# Train ElasticNet model
model = ElasticNet()
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0],
    'l1_ratio': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
}
grid_search = GridSearchCV(model, param_grid=param_grid, cv=5)
grid_search.fit(X_train_transformed, y_train)
 
# Evaluate the model
y_pred = grid_search.predict(X_test_transformed)
X_test = X_test.reset_index(drop=True)
 
# Define functions for demand and profit calculation
def demand_function(price, competitor_price, predicted_demand):
    sensitivity = 2
    competitor_influence = 0.75
    return predicted_demand - sensitivity * (price - competitor_price) * competitor_influence
 
def calculate_profit(price, cost, competitor_price, predicted_demand):
    demand = demand_function(price, competitor_price, predicted_demand)
    profit = (price - cost) * demand
    return profit
 
def optimize_price(cost, competitor_price, predicted_demand, price_bounds=(100, 10000)):
    def objective(price):
        return -calculate_profit(price, cost, competitor_price, predicted_demand)
    result = minimize_scalar(objective, bounds=price_bounds, method='bounded')
    optimal_price = result.x
    max_profit = -result.fun
    return optimal_price, max_profit
 
# Handle user input
if submitted:
    # Encode user input
    user_input_categorical = pd.DataFrame([[brand_name, 1]], columns=categorical_features)
    user_input_categorical = categorical_transformer.transform(user_input_categorical)
   
    user_input_numeric = pd.DataFrame([[cost, 0, competitor_price]], columns=numeric_features)
    user_input_numeric = numeric_transformer.transform(user_input_numeric)
   
    user_input_transformed = np.hstack((user_input_numeric, user_input_categorical))
   
    # Predict demand for user input
    user_predicted_demand = grid_search.predict(user_input_transformed)[0]
 
    # Optimize price for user input
    optimal_price, max_profit = optimize_price(cost, competitor_price, user_predicted_demand)
   
    # Display results
    st.write(f'Optimal Price: {optimal_price:.2f}')
    st.write(f'Maximum Profit: {max_profit:.2f}')
 
# Collect results
results = []
 
for idx, row in X_test.iterrows():
    production_cost = row['ProductionCost']
    competitor_price = row['CompetitorPrice']
    predicted_demand = y_pred[idx]
   
    optimal_price, max_profit = optimize_price(production_cost, competitor_price, predicted_demand)
   
    results.append({
        'BrandName': row['BrandName'],
        'ItemNumber': row['ItemNumber'],
        'ProductionCost': production_cost,
        'CompetitorPrice': competitor_price,
        'PredictedDemand': predicted_demand,
        'OptimizedPrice': optimal_price
    })
 
# Format results
results_df = pd.DataFrame(results)
st.subheader("Original Dataset Predictions")
st.table(results_df)
