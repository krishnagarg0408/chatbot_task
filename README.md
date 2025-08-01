# 🤖 AI Chatbot using NLP (Task 3 - CODTECH Internship)

This is a simple AI-based chatbot built using **Natural Language Processing (NLP)** libraries like **NLTK** and **scikit-learn** in Python. It can answer basic queries using text matching and intent recognition.

---

## 📌 Features

- Uses **NLTK** for sentence/token processing
- Vectorizes input using **CountVectorizer**
- Matches user queries using **cosine similarity**
- Pre-trained on a custom text dataset
- Simple CLI (command-line) interaction

---

## 🛠️ Requirements

Make sure you have Python installed. Install required libraries using pip:

```bash
pip install nltk
pip install scikit-learn
import nltk
nltk.download('punkt')
python chatbot.py
