from flask import Flask, request, jsonify
from semantic_similarity_model import SemanticSimilarityModel
import os

app = Flask(__name__)

# Load or initialize the model
try:
    model = SemanticSimilarityModel.load_model()
    print("Model loaded successfully.")
except:
    print("No saved model found. Initializing a new model.")
    model = SemanticSimilarityModel()

@app.route('/similarity', methods=['POST'])
def get_similarity():
    """
    API endpoint to compute the semantic similarity between two texts.
    
    Expected request format:
    {
        "text1": "first paragraph...",
        "text2": "second paragraph..."
    }
    
    Returns:
    {
        "similarity score": float
    }
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate input
        if 'text1' not in data or 'text2' not in data:
            return jsonify({"error": "Both 'text1' and 'text2' fields are required"}), 400
        
        # Compute similarity
        similarity = model.compute_similarity(data['text1'], data['text2'])
        
        # Return result
        return jsonify({"similarity score": similarity})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Get port from environment variable (for cloud deployment)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=False)