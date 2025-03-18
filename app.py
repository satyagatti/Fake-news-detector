from flask import Flask, render_template, request
import pickle

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    print("✅ Home page loaded")  # Debugging print
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    news_text = request.form.get("news", "")
    print("User input:", news_text)  # Debugging print
    transformed_text = vectorizer.transform([news_text])
    prediction = model.predict(transformed_text)
    result = "Fake News" if prediction[0] == 1 else "Real News"

    return render_template("index.html", prediction=result)

def handler(event, context):
    return app(event, context)


if __name__ == '__main__':
    app.run(debug=True)
