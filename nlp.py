import nltk
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from flask import Flask, render_template, request
from datetime import datetime

app = Flask(__name__)

# Rename the global variable to avoid the name clash
nltk_data_downloaded = False  

# Download required NLTK resources (only need to do this once)
@app.before_request
def download_nltk_data():
    """Downloads NLTK data and sets the data path (only once)."""
    global nltk_data_downloaded  # Use the correct global variable name
    if not nltk_data_downloaded:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('vader_lexicon')
        nltk.data.path.append(os.path.join(app.root_path, 'nltk_data'))
        nltk_data_downloaded = True  # Update the global variable

def analyze_text(text):
    print(f"Received text for analysis: {text}")
    """
    Analyzes text to understand its emotional tone, like a friend listening 
    to your thoughts! It identifies key topics and gives you insights into 
    how your words might be perceived.
    """
    # Tokenization: Break down text into individual words
    tokens = word_tokenize(text)
    print(f"Tokens after word_tokenize: {tokens}")

    # Remove stop words for more meaningful analysis
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Lemmatization: Group words with similar meanings
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Sentiment Analysis using VADER
    sid = SentimentIntensityAnalyzer()
    emotions = sid.polarity_scores(text)

    # Sentiment Analysis using TextBlob
    blob = TextBlob(text)

    # Key Phrase Extraction (example using frequency)
    word_freq = {}
    for lemma in lemmas:
        if lemma in word_freq:
            word_freq[lemma] += 1
        else:
            word_freq[lemma] = 1

    key_phrases = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:3]  # Get top 3

    # --- Format output for HTML ---
    analysis_output = []

    # More expressive emotional responses
    if emotions['pos'] > emotions['neg']:
        analysis_output.append(f"<li>🌟 Wow, your message radiates positivity! You're really bringing some sunshine into the world. (Positivity: {emotions['pos']:.2f})</li>")
    elif emotions['neg'] > emotions['pos']:
        analysis_output.append(f"<li>😢 I can sense you're going through something tough right now. It's okay to feel down. I'm here to listen. (Negativity: {emotions['neg']:.2f})</li>")
    else:
        analysis_output.append(f"<li>🧘 Your message seems balanced and calm, like you're taking a peaceful approach to things. (Neutrality: {emotions['neu']:.2f})</li>")

    # Emotional intensity analysis
    if emotions['compound'] > 0.5:
        analysis_output.append("<li>😊 Your words are full of energy and enthusiasm! Keep that energy flowing!</li>")
    elif emotions['compound'] < -0.5:
        analysis_output.append("<li>😔 You sound deeply upset or worried. It’s okay, tough times pass. Stay strong!</li>")

    # Subjectivity insights
    if blob.sentiment.subjectivity > 0.7:
        analysis_output.append(f"<li>💭 You're sharing some deep personal thoughts and emotions. It's great to express yourself!</li>")
    elif blob.sentiment.subjectivity > 0.5:
        analysis_output.append(f"<li>🗣 You're being expressive, and it's clear you're talking from a personal perspective.</li>")
    else:
        analysis_output.append(f"<li>📊 Your words seem more matter-of-fact, you're keeping things clear and objective.</li>")

    # Overall sentiment feedback
    if blob.sentiment.polarity > 0.5:
        analysis_output.append(f"<li>✨ Overall, your message shines with positivity. Keep it up!</li>")
    elif blob.sentiment.polarity > 0:
        analysis_output.append(f"<li>🙂 Your message has a nice positive vibe!</li>")
    elif blob.sentiment.polarity < -0.5:
        analysis_output.append(f"<li>😔 It seems like you're really going through a rough time. Remember, I'm here if you need to talk.</li>")
    elif blob.sentiment.polarity < 0:
        analysis_output.append(f"<li>😕 There's a bit of negativity in your message. It's okay to feel this way sometimes.</li>")
    else:
        analysis_output.append(f"<li>🤔 Your message seems neutral, just expressing some thoughts.</li>")

    # Key phrases
    key_phrases_output = ", ".join([phrase[0] for phrase in key_phrases])
    analysis_output.append(f"<li>🔑 Key themes I'm picking up on: {key_phrases_output}</li>")

    print(f"Analysis results: {analysis_output}")
    return {
        'vader_emotions': emotions,
        'textblob_sentiment': blob.sentiment,
        'key_phrases': key_phrases,
        'analysis_output': analysis_output  # Add the formatted output
    }

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_text = request.form["text"]
        try:
            analysis_results = analyze_text(user_text)
        except Exception as e:
            return render_template("index.html", error=f"An error occurred: {str(e)}")
        return render_template("index.html", results=analysis_results, text=user_text)
    else:
        return render_template("index.html", results=None)

@app.after_request
def log_request(response):
    """Logs each request to the console."""
    timestamp = datetime.now().strftime("%d/%b/%Y %H:%M:%S")
    print(f"{request.remote_addr} - - [{timestamp}] \"{request.method} {request.path} HTTP/1.1\" {response.status_code} -")
    return response

if __name__ == "__main__":
    app.run()  # debug defaults to False