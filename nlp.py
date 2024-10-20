import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from flask import Flask, render_template, request
from datetime import datetime

# Download required NLTK resources (only need to do this once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

app = Flask(__name__)

def analyze_text(text):
    """
    Analyzes text to understand its emotional tone, like a friend listening 
    to your thoughts! It identifies key topics and gives you insights into 
    how your words might be perceived. 
    """
    # Tokenization: Break down text into individual words
    tokens = word_tokenize(text)

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

    # analysis_output.append(f"<li>💬 You said: {text}</li>")

    # More expressive emotional responses
    if emotions['pos'] > emotions['neg']:
        analysis_output.append(f"<li>🤔  Here's what I understand from what you've shared:  Wow, you sound incredibly upbeat and positive! 😄  Your words radiate joy! (Positivity: {emotions['pos']:.2f})</li>")
    elif emotions['neg'] > emotions['pos']:
        analysis_output.append(f"<li>🤔  Here's what I understand from what you've shared:  It seems like you're carrying a bit of weight on your shoulders. 😔 I'm here for you if you want to talk. (Negativity: {emotions['neg']:.2f})</li>")
    else:
        analysis_output.append(f"<li>🤔  Here's what I understand from what you've shared:  You seem to be approaching things with a calm and neutral perspective. 🤔  It's refreshing to see such balance!</li>")

    # More descriptive language for subjectivity
    if blob.sentiment.subjectivity > 0.7:
        analysis_output.append(f"<li> Your words are deeply personal and full of emotion. It's clear you're speaking from the heart. 🤔</li>")
    elif blob.sentiment.subjectivity > 0.5:
        analysis_output.append(f"<li> You're expressing yourself in a very personal and subjective way. 🤔</li>")
    else:
        analysis_output.append(f"<li> Your words come across as quite objective and matter-of-fact. 🧐</li>")

    # More nuanced overall sentiment descriptions
    if blob.sentiment.polarity > 0.5:
        analysis_output.append(f"<li> Overall, your message is a burst of sunshine!  Your positivity is contagious! ✨</li>")
    elif blob.sentiment.polarity > 0:
        analysis_output.append(f"<li> Overall, your message gives off a positive vibe! ✨</li>")
    elif blob.sentiment.polarity < -0.5:
        analysis_output.append(f"<li> I'm truly sensing the weight of your words. 😔  Remember, you're not alone in this. </li>")
    elif blob.sentiment.polarity < 0:
        analysis_output.append(f"<li> I'm sensing a bit of negativity in your words. 😔</li>")
    else:
        analysis_output.append(f"<li> Your message seems pretty neutral.  Just sharing your thoughts? 🤔</li>")

    key_phrases_output = ", ".join([phrase[0] for phrase in key_phrases])
    analysis_output.append(f"<li>🔑 Key themes I'm picking up on: {key_phrases_output}</li>")

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
        analysis_results = analyze_text(user_text)
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
    app.run(debug=True)
