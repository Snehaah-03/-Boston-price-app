from flask import Flask, render_template, request
import pickle
import numpy as np

# Create the Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all values from form and convert to float
        values = [float(x) for x in request.form.values()]
        inputs = [np.array(values)]

        # Make prediction
        prediction = model.predict(inputs)
        result = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f"Estimated House Price: ${result}K")
    except Exception as e:
        return render_template('index.html', prediction_text="❌ Error in input values. Please enter valid numbers.")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)