To create a simple Python web application to show SHAP (SHapley Additive exPlanations) values, you can use a web framework like Flask along with the SHAP library. SHAP values help explain the output of machine learning models by attributing the model's prediction to its input features.

Here's a basic example using Flask and SHAP:

1. Install required packages:

```bash
pip install flask shap
```

2. Create a file named `app.py` with the following code:

```python
from flask import Flask, render_template, request
import shap
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load a sample model (replace this with your trained model)
# Here, we use a simple RandomForestRegressor as an example.
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston

boston = load_boston()
X = boston.data
y = boston.target

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Generate SHAP values for a sample instance
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X[0])

# Define route for the web application
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the feature values from the form
        features = [float(request.form[f"feature{i}"]) for i in range(1, len(X[0]) + 1)]

        # Generate SHAP values for the input features
        shap_values = explainer.shap_values(np.array([features]))

        # Plot the SHAP values
        shap.summary_plot(shap_values, features, plot_type="bar")
        plt.savefig('static/shap_plot.png')
        plt.close()

        return render_template('index.html', image_path='static/shap_plot.png')

    # Render the initial form
    return render_template('index.html', image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
```

3. Create a folder named `templates` in the same directory as `app.py`, and create a file named `index.html` inside it:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SHAP Values Web App</title>
</head>
<body>
    <h1>SHAP Values Web App</h1>
    <form method="post" action="/">
        {% if image_path %}
            <img src="{{ image_path }}" alt="SHAP Values Plot">
        {% endif %}
        <h3>Enter Feature Values:</h3>
        {% for i in range(1, features|length + 1) %}
            <label for="feature{{ i }}">Feature {{ i }}:</label>
            <input type="text" id="feature{{ i }}" name="feature{{ i }}" required>
            <br>
        {% endfor %}
        <br>
        <input type="submit" value="Generate SHAP Values">
    </form>
</body>
</html>
```

This example uses a RandomForestRegressor from scikit-learn on the Boston Housing dataset. Replace it with your own trained model and dataset. The web app allows users to input feature values through a form, generates SHAP values, and displays a bar plot of the SHAP values.

Run the application using:

```bash
python app.py
```

Open your web browser and go to `http://127.0.0.1:5000/` to interact with the web application.