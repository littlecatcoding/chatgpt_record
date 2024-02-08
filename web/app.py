from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Sample data for the table
data = [
    {"name": "Cat", "description": "A small domesticated carnivorous mammal with soft fur."},
    {"name": "Dog", "description": "A domesticated carnivorous mammal that typically has a long snout."},
    {"name": "Bird", "description": "A warm-blooded egg-laying vertebrate distinguished by the possession of feathers."},
    # Add more sample data as needed
]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect(url_for('show_cat_image'))
    return render_template('index.html', data=data)

@app.route('/show_cat_image')
def show_cat_image():
    # This route simply returns the HTML with the cat image
    return render_template('cat_image.html')

if __name__ == '__main__':
    app.run(debug=True)