<h1 align="center">📧 Spam Email Detection using Machine Learning</h1>

<p align="center">
A Machine Learning application that classifies email messages as <b>Spam</b>, <b>Not Spam</b>, or <b>Uncertain</b> using <b>Logistic Regression</b> and <b>CountVectorizer</b>.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-success)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Logistic%20Regression-orange?logo=scikitlearn)
![Gradio](https://img.shields.io/badge/Gradio-Web%20App-FF6F00)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Deployed-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)

</p>

---

# 🚀 Live Demo

🌐 **Try the application here:**

👉 **https://huggingface.co/spaces/Siddhartha001/spam-email-detection-ml**

---

# 📌 Project Overview

Spam emails are one of the most common cybersecurity threats, often containing phishing links, fake prizes, malicious attachments, or financial scams.

This project demonstrates how Machine Learning can automatically classify email messages into:

- 🚨 **Spam Email**
- ✅ **Not Spam Email**
- ⚠️ **Uncertain Email**

Instead of forcing an incorrect prediction, the application returns an **Uncertain** result whenever the model confidence falls within a predefined range.

The application is built using **Python**, **Scikit-Learn**, **Gradio**, and deployed on **Hugging Face Spaces** for real-time predictions.

---

# 🌟 Project Highlights

- 🚀 Developed a real-time **Spam Email Detection** web application.
- 🧠 Built using **Machine Learning (Logistic Regression)**.
- 📝 Applied **Natural Language Processing (NLP)** using **CountVectorizer**.
- 📊 Displays prediction confidence for every email.
- ⚠️ Returns an **Uncertain** prediction for low-confidence classifications.
- 🌐 Successfully deployed on **Hugging Face Spaces**.
- 🎯 Designed a simple and beginner-friendly user interface using **Gradio**.

---

# 🏗️ System Architecture

The following diagram illustrates the complete workflow of the Spam Email Detection system.

<p align="center">
        <img src="images/architecture-diagram.png" alt="Spam Email Detection Architecture" width="100%">
</p>

---

# 🖥️ Application Preview

## 🏠 Home Screen

The main interface where users can enter an email message and receive a prediction.

<p align="center">
  <img src="images/home.png" alt="Home Screen" width="90%">
</p>

---

## 🚨 Spam Email Prediction

Example showing a spam email detected with a high confidence score.

<p align="center">
  <img src="images/spam_prediction.png" alt="Spam Prediction" width="90%">
</p>

---

## ⚠️ Uncertain Email Prediction

If the confidence falls within the predefined threshold, the application returns an **Uncertain** result instead of forcing an incorrect prediction.

<p align="center">
  <img src="images/uncertain_prediction.png" alt="Uncertain Prediction" width="90%">
</p>

---

## ✅ Not Spam Email Prediction

Example of a legitimate email classified as **Not Spam**.

<p align="center">
  <img src="images/normal_prediction.png" alt="Normal Prediction" width="90%">
</p>


---

# ✨ Key Features

- 🚨 Real-time Spam Email Detection
- 🧠 Machine Learning based Classification
- 📝 Natural Language Processing using CountVectorizer
- 📊 Prediction Confidence Score
- ⚠️ Uncertain Prediction for Low Confidence
- 🌐 Interactive Gradio Web Interface
- ☁️ Deployed on Hugging Face Spaces
- 📱 Simple and Responsive User Interface
- 🔍 Easy-to-use Email Testing Interface

---

# ⚙️ Workflow

```text
                User
                  │
                  ▼
        Gradio Web Interface
                  │
                  ▼
            Email Input
                  │
                  ▼
         Text Preprocessing
                  │
                  ▼
          CountVectorizer
                  │
                  ▼
          Feature Vector
                  │
                  ▼
      Logistic Regression Model
                  │
                  ▼
     Prediction + Confidence Score
          │        │         │
          │        │         │
          ▼        ▼         ▼
      🚨 Spam   ✅ Not Spam  ⚠️ Uncertain
                  │
                  ▼
           Display Prediction
```

---

# 🛠️ Technology Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Machine Learning | Scikit-Learn |
| Algorithm | Logistic Regression |
| Natural Language Processing | CountVectorizer |
| Numerical Computing | NumPy |
| Model Serialization | Joblib |
| Web Framework | Gradio |
| Deployment | Hugging Face Spaces |
| Version Control | Git & GitHub |
