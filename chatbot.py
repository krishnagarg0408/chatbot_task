import nltk
import random
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# First time run only
# nltk.download('punkt')

# Sample dataset for the chatbot
chat_data = """
Hi there!
Hello!
How can I help you?
What is your name?
I am a chatbot created by Krishna.
What do you do?
I answer questions.
How are you?
I'm just a bot, but I'm doing fine!
Bye
Goodbye!
See you later
Take care!
"""

# Preprocessing
sent_tokens = nltk.sent_tokenize(chat_data.lower())

def preprocess(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

# Generate response
def get_response(user_input):
    user_input = preprocess(user_input)
    sent_tokens.append(user_input)
    
    vectorizer = CountVectorizer().fit_transform(sent_tokens)
    vectors = vectorizer.toarray()
    
    similarity = cosine_similarity(vectors[-1:], vectors)
    index = similarity.argsort()[0][-2]
    flat = similarity.flatten()
    flat.sort()
    score = flat[-2]

    sent_tokens.pop()  # remove user input

    if score == 0:
        return "I'm sorry, I don't understand."
    else:
        return sent_tokens[index]

# Chat loop
print("Chatbot: Hello! Ask me anything or type 'bye' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['bye', 'exit', 'quit']:
        print("Chatbot: Goodbye! 👋")
        break
    else:
        response = get_response(user_input)
        print("Chatbot:", response)
