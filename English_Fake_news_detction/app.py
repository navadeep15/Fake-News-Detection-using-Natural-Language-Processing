from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the saved model and vectorizer
model_path = './model/knn_model.pkl'
vectorizer_path = './model/tfidf_vectorizer.pkl'

# Load model
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load vectorizer
with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Transform the input text using the loaded vectorizer
    text_vec = vectorizer.transform([text])
    
    # Make a prediction using the loaded model
    prediction = model.predict(text_vec)[0]

    # Map prediction to labels
    prediction_label = "Fake News" if prediction == 1 else "True News"

    # Return the prediction result
    return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    app.run(debug=True)