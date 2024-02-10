from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import json
import nltk
from flask_cors import CORS


app = Flask(__name__)
CORS(app) 
# Load the Bhagavad Gita JSON file with explicit encoding
with open('MLData.json', 'r', encoding='utf-8') as file:
    bhagavad_gita_data = json.load(file)

# Combine translation and purport text for each verse
combined_text = {key: f"{value['translation']} {' '.join(value['purport'])}" for key, value in bhagavad_gita_data.items()}

# Preprocess text
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [ps.stem(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words), ' '.join(word_tokenize(text.lower()))  # Return both stemmed and original text

# Update combined_text with preprocessed text
combined_text = {key: preprocess_text(value) for key, value in combined_text.items()}

# Create a TF-IDF vectorizer with n-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, smooth_idf=True)
tfidf_matrix = vectorizer.fit_transform([stemmed_text for stemmed_text, _ in combined_text.values()])

# Define base URL for verse links
base_url = "https://gita-learn.vercel.app/VerseDetail?chapterVerse="

# Define route for handling search query
@app.route('/search', methods=['POST'])
def search():
    user_query = request.json['user_query']

    # Preprocess user query
    user_query_stemmed, user_query_original = preprocess_text(user_query)

    # Vectorize the user query
    query_vector = vectorizer.transform([user_query_stemmed])

    # Calculate cosine similarity between the query and Bhagavad Gita verses
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get the index of the most similar verse
    most_similar_index = cosine_similarities.argmax()

    # Get the verse number and content of the most similar verse
    verse_number = list(combined_text.keys())[most_similar_index]
    verse_content_stemmed, verse_content_original = combined_text[verse_number]

    # Highlight the matched words in the content
    highlighted_content = verse_content_original.replace(user_query_original, f"<span class='text-red-500'>{user_query_original}</span>")


    # Generate the link for the entire verse
    verse_link = f"{base_url}{verse_number}"

    # Prepare response data
    response_data = {
        'user_query': user_query_original,
        'verse_number': verse_number,
        'highlighted_content': highlighted_content,
        'verse_link': verse_link
    }

    return jsonify(response_data)

