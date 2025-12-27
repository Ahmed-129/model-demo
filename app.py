import pickle
 
from flask import Flask, request, render_template

# Load trained models and preprocessors
domain_model = pickle.load(open("domain_model.pkl", "rb"))
subdomain_model = pickle.load(open("subdomain_model.pkl", "rb"))

tfidf = pickle.load(open("tfidf.pkl", "rb"))
le_domain = pickle.load(open("domain_encoder.pkl", "rb"))
le_sub = pickle.load(open("subdomain_encoder.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Get skills input from form
    skills = request.form["top_skills"]

    # Transform skills using TF-IDF
    X = tfidf.transform([skills])

    # Predict domain and sub-domain
    domain_pred = domain_model.predict(X)
    subdomain_pred = subdomain_model.predict(X)

    # Decode predictions
    domain = le_domain.inverse_transform(domain_pred)[0]
    subdomain = le_sub.inverse_transform(subdomain_pred)[0]

    return render_template(
        "index.html",
        prediction_text=f"Domain: {domain} | Sub-domain: {subdomain}"
    )


if __name__ == "__main__":
    app.run(debug=True)
