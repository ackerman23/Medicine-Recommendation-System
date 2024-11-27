import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import pickle
import os

def plot_reviews(data):
    labels = ['Excellent', 'Average', 'Poor']
    sizes = [data['Excellent Review %'].mean(), data['Average Review %'].mean(), data['Poor Review %'].mean()]

    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.savefig('app/static/reviews.png')

class MedicineRecommender:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.conditions = None
        self.condition_embeddings = None
        self.df = None
        
    def load_data(self, data_path):
        """Load and prepare the medicine dataset"""
        self.df = pd.read_csv(data_path)
        self.df['input_text'] = (
            "Keywords: " + self.df['Keywords'] +
            " Composition: " + self.df['Composition'] +
            " Uses: " + self.df['Uses'] +
            " Side Effects: " + self.df['Side_effects'] +
            " Manufacturer: " + self.df['Manufacturer'] +
            " Excellent Reviews: " + self.df['Excellent Review %'].astype(str) +
            " Average Reviews: " + self.df['Average Review %'].astype(str) +
            " Poor Reviews: " + self.df['Poor Review %'].astype(str)
        )
        
        # Extract and encode conditions
        self.conditions = self._extract_conditions(self.df['Uses'])
        self.condition_embeddings = self.model.encode(self.conditions)
        
    def _extract_conditions(self, uses_series):
        """Extract unique conditions from Uses column"""
        conditions = set()
        for uses in uses_series:
            conditions_list = [c.strip().lower() for c in str(uses).split(',')]
            conditions.update(conditions_list)
        return list(conditions)
    
    def predict_condition(self, symptoms, top_k=3):
        """Predict possible conditions based on symptoms"""
        symptoms_embedding = self.model.encode([symptoms])
        similarities = cosine_similarity(symptoms_embedding, self.condition_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.conditions[idx], similarities[idx]) for idx in top_indices]
    
    def recommend_medicine_for_condition(self, condition, top_k=5):
        """Recommend medicines for a specific condition"""
        relevant_medicines = self.df[self.df['Uses'].str.lower().str.contains(condition.lower())]
        
        if len(relevant_medicines) == 0:
            return []
        
        medicine_texts = relevant_medicines['input_text'].tolist()
        medicine_embeddings = self.model.encode(medicine_texts)
        
        query = f"Treatment for {condition}"
        query_embedding = self.model.encode([query])
        
        similarities = cosine_similarity(query_embedding, medicine_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        recommendations = []
        for idx in top_indices:
            recommendations.append({
                'medicine': relevant_medicines['Medicine Name'].iloc[idx],
                'similarity_score': float(similarities[idx]), 
                'uses': relevant_medicines['Uses'].iloc[idx],
                'composition': relevant_medicines['Composition'].iloc[idx]
            })
        
        return recommendations
    
    def get_recommendations_from_symptoms(self, symptoms):
        """Get recommendations based on symptoms"""
        results = []
        predicted_conditions = self.predict_condition(symptoms)
        
        for condition, score in predicted_conditions:
            condition_result = {
                'condition': condition,
                'confidence': float(score),
                'recommendations': self.recommend_medicine_for_condition(condition)
            }
            results.append(condition_result)
        
        return results
    
    def save_model(self, path='models'):
        """Save the model and its data"""
        os.makedirs(path, exist_ok=True)
        
        # Save the conditions and their embeddings
        with open(os.path.join(path, 'conditions.pkl'), 'wb') as f:
            pickle.dump({
                'conditions': self.conditions,
                'condition_embeddings': self.condition_embeddings
            }, f)
        
        # Save the dataframe
        self.df.to_pickle(os.path.join(path, 'medicine_data.pkl'))
    
    @classmethod
    def load_model(cls, path='models', model_name='all-MiniLM-L6-v2'):
        """Load a saved model"""
        recommender = cls(model_name)
        
        # Load conditions and embeddings
        with open(os.path.join(path, 'conditions.pkl'), 'rb') as f:
            data = pickle.load(f)
            recommender.conditions = data['conditions']
            recommender.condition_embeddings = data['condition_embeddings']
        
        # Load the dataframe
        recommender.df = pd.read_pickle(os.path.join(path, 'medicine_data.pkl'))
        
        return recommender
