from flask import Flask, request, jsonify, render_template
from utils import MedicineRecommender
import os

app = Flask(__name__)

# Initialize the recommender
if os.path.exists('models/conditions.pkl'):
    # Load existing model
    recommender = MedicineRecommender.load_model()
else:
    # Create and save new model
    recommender = MedicineRecommender()
    recommender.load_data('/Users/jihadgarti/Desktop/github-path/Medicine-Recommendation-System/app/data/preprocessed_medicines.csv')
    recommender.save_model()

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    symptoms = data.get('symptoms')
    
    if not symptoms:
        return jsonify({'error': 'No symptoms provided'}), 400
    
    try:
        recommendations = recommender.get_recommendations_from_symptoms(symptoms)
        return jsonify({'results': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
