import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import random


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer


faq_data = {
    "What is the battery life of the laptop?": "The battery lasts up to 10 hours on a full charge.",
    "Does the laptop come with a warranty?": "Yes, it includes a 1-year manufacturer warranty.",
    "Can I upgrade the RAM?": "Yes, you can upgrade the RAM up to 32GB.",
    "Is the laptop good for gaming?": "Yes, it has a dedicated GPU suitable for most modern games.",
    "What is the screen size of the laptop?": "The laptop features a 15.6-inch Full HD display."
}


lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])


questions = list(faq_data.keys())
answers = list(faq_data.values())
preprocessed_questions = [preprocess(q) for q in questions]


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_questions)


def chatbot(user_input):
    user_input_processed = preprocess(user_input)
    user_vec = vectorizer.transform([user_input_processed])
    similarities = cosine_similarity(user_vec, tfidf_matrix)
    
    best_match_idx = np.argmax(similarities)
    confidence = similarities[0][best_match_idx]
    
    if confidence > 0.3:  
        return answers[best_match_idx]
    else:
        return "Sorry, I don't have an answer for that. Can you rephrase?"

if __name__ == "__main__":
    print("🤖 FAQ Bot: Ask me anything about your new laptop! (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("FAQ Bot: Goodbye!")
            break
        response = chatbot(user_input)
        print("FAQ Bot:", response)
