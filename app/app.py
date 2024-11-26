from flask import Flask, request, render_template, jsonify
import pandas as pd

app = Flask(__name__)
data = pd.read_csv('data/preprocessed_medicines.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user input
    symptoms = request.form['symptoms'].lower().split()

    # Filter medicines matching symptoms
    recommendations = []
    for _, row in data.iterrows():
        if any(symptom in row['Keywords'] for symptom in symptoms):
            recommendations.append(row)

    return render_template('results.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
