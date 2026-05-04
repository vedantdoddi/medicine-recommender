from flask import Flask, request, jsonify, render_template
from model import MedicineRecommender

app = Flask(__name__)

# Initialize ML model when the server starts
try:
    recommender = MedicineRecommender('dataset.csv')
except Exception as e:
    print(f"Error loading AI model: {e}")
    recommender = None

@app.route('/')
def home():
    """Serves the front-end interface"""
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def get_recommendation():
    """API Endpoint receiving JSON symptom payload and returning Top Match."""
    if recommender is None:
        return jsonify({"error": "Recommendation system is currently unavailable. Dataset issue."}), 500

    data = request.get_json()
    
    if not data or 'symptoms' not in data:
        return jsonify({"error": "Please provide your symptoms in the JSON payload."}), 400

    symptoms = data['symptoms']
    result = recommender.recommend(symptoms)

    if not result:
        # Graceful handling of unknown symptoms bounds
        return jsonify({
            "found": False,
            "message": "No closely matching medicine was identified for these symptoms. Please consult a doctor immediately."
        }), 200

    # Successful match execution
    return jsonify({
        "found": True,
        "data": result
    }), 200

if __name__ == '__main__':
    # Local development server parameters
    app.run(debug=True, host='127.0.0.1', port=5000)
