from flask import Flask, render_template, request
import joblib
import pickle

app = Flask(__name__,template_folder='template')
model = pickle.load(open("C:\\Users\\abhay\\AI-Powered Resume Screening System\\resume_model.pkl","rb")) # Load your pre-trained model
vectorizer = pickle.load(open("C:\\Users\\abhay\\AI-Powered Resume Screening System\\vectorized.pkl","rb"))  # Assuming you used a vectorizer for text transformation
encoder=pickle.load(open("C:\\Users\\abhay\\AI-Powered Resume Screening System\\encoder.pkl","rb"))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input resume text from form
        resume_text = request.form['resume_text']

        # Transform text using vectorizer
        transformed_text = vectorizer.transform([resume_text])

        # Predict using the model
        prediction = model.predict(transformed_text)

        # Convert prediction to label
        predicted_label = encoder.inverse_transform(prediction)[0]
        return render_template('index.html', prediction_text=f'Predicted Label: {predicted_label}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')


if __name__ == '__main__':
    app.run(debug=True)
