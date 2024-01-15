import lightgbm as lgb
import shap
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import psycopg2

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a LightGBM model
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt'
}

train_data = lgb.Dataset(X_train, label=y_train)
model = lgb.train(params, train_data, num_boost_round=100)

# Explain the model using SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Connect to Redshift
conn = psycopg2.connect(
    host='your-redshift-host',
    port='your-redshift-port',
    user='your-username',
    password='your-password',
    database='your-database'
)

# Create a cursor
cursor = conn.cursor()

# Create a table to store SHAP values
create_table_query = """
CREATE TABLE IF NOT EXISTS shap_values (
    id SERIAL PRIMARY KEY,
    feature1 FLOAT,
    feature2 FLOAT,
    feature3 FLOAT,
    feature4 FLOAT,
    shap_value1 FLOAT,
    shap_value2 FLOAT,
    shap_value3 FLOAT,
    prediction FLOAT
);
"""
cursor.execute(create_table_query)

# Insert SHAP values into the table
for i in range(len(X_test)):
    insert_query = """
    INSERT INTO shap_values (feature1, feature2, feature3, feature4, shap_value1, shap_value2, shap_value3, prediction)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """
    values = tuple(list(X_test[i]) + list(shap_values[i]) + [model.predict(X_test[i])[0]])
    cursor.execute(insert_query, values)

# Commit changes and close connections
conn.commit()
cursor.close()
conn.close()
