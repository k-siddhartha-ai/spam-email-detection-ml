import gradio as gr
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_spam(email_text):
    if not email_text or not email_text.strip():
        return "Please enter email text"

    vector = vectorizer.transform([email_text])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0][1]

    if prediction == 1:
        return f"This is a SPAM email\nConfidence: {probability*100:.2f}%"
    else:
        return f"This is NOT a spam email\nConfidence: {probability*100:.2f}%"

app = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=4, placeholder="Enter email text here"),
    outputs="text",
    title="Email Spam Detection",
    description="This app is used to check whether the given email text is spam or not spam.",
    examples=[
        ["click here to earn money"],
        ["tommarrow is holiday"],
        ["click to get bonus"],
        ["click to download image"]
    ]
)

if __name__ == "__main__":
    app.launch()
