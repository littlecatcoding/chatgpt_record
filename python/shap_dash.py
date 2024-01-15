import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import shap
import lightgbm as lgb
import pandas as pd

# Load your dataset and model
# Replace the following lines with your data loading and model loading code
# For this example, we'll use a dummy dataset and model
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=42)

model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# Create SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Create Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("SHAP Values for LightGBM Model"),
    
    # Dropdown for selecting the instance
    dcc.Dropdown(
        id='instance-dropdown',
        options=[{'label': i, 'value': i} for i in range(len(X_test))],
        value=0,
        style={'width': '50%'}
    ),
    
    # Scatter plot for displaying SHAP values
    dcc.Graph(id='shap-scatter-plot'),
])

# Define callback to update the scatter plot based on the selected instance
@app.callback(
    Output('shap-scatter-plot', 'figure'),
    [Input('instance-dropdown', 'value')]
)
def update_scatter_plot(selected_instance):
    instance_shap_values = shap_values[selected_instance, :]
    feature_names = boston.feature_names

    df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': instance_shap_values})
    
    fig = px.scatter(
        df, x='SHAP Value', y='Feature',
        orientation='h',
        title=f'SHAP Values for Instance {selected_instance}',
        labels={'SHAP Value': 'SHAP Value'},
        height=500
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
