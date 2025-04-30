import streamlit as st
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
import nltk
import pandas as pd
import re
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from deep_translator import GoogleTranslator
from langdetect import detect
import plotly.express as px
import plotly.graph_objects as go

# Download necessary NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set page title and configuration
st.set_page_config(
    page_title="Analyseur de Sentiment de Tweets",
    page_icon="🐦",
    layout="centered"
)

# Initialize translator
translator = GoogleTranslator(source='auto', target='en')

def translate_text(text, source_lang='fr', target_lang='en'):
    """
    Traduit le texte d'une langue à une autre en utilisant deep_translator
    """
    try:
        translation = translator.translate(text)
        return translation
    except Exception as e:
        st.error(f"Erreur de traduction: {e}")
        return text  # Return original text in case of error

def clean_tweet(tweet):
    """Nettoie le tweet en supprimant les liens, mentions et caractères spéciaux"""
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#(\w+)', r'\1', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    return tweet

def lemmatize_text(text, language='en'):
    """
    Lemmatize text (réduction des mots à leur forme racine)
    Works better for English, but we'll use it for the translated text
    """
    if language != 'en':
        return text
        
    try:
        nltk.data.find('wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def detect_language(text):
    """Détecte si le texte est en français ou en anglais using langdetect"""
    try:
        lang = detect(text)
        return lang
    except:
        # Fallback to simple word-based detection
        french_words = ["je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "et", "ou", 
                       "mais", "donc", "car", "est", "sont", "être", "avoir", "le", "la", "les", 
                       "un", "une", "des", "du", "de", "ce", "cette", "ces", "mon", "ton", "son"]
        english_words = ["i", "you", "he", "she", "we", "they", "it", "and", "or", "but", 
                        "is", "are", "be", "have", "has", "the", "a", "an", "this", "that", 
                        "these", "those", "my", "your", "his", "her"]
        text_lower = text.lower()
        french_count = sum(1 for word in french_words if f" {word} " in f" {text_lower} ")
        english_count = sum(1 for word in english_words if f" {word} " in f" {text_lower} ")
        return "fr" if french_count > english_count else "en"

def analyze_sentiment(text, language=None):
    """Analyze sentiment of input text using multiple approaches"""
    if not text.strip():
        return None, None, None, None, None
    
    cleaned_text = clean_tweet(text)
    if language is None:
        language = detect_language(cleaned_text)
    
    translated_text = None
    if language == "fr":
        translated_text = translate_text(cleaned_text, source_lang='fr', target_lang='en')
        processed_text = lemmatize_text(translated_text, language='en')
    else:
        processed_text = lemmatize_text(cleaned_text, language='en')
        translated_text = cleaned_text
    
    if language == "fr":
        blob_original = TextBlob(cleaned_text, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
        textblob_polarity_original = blob_original.sentiment[0]  # Extract polarity from tuple
    else:
        blob_original = TextBlob(cleaned_text)
        textblob_polarity_original = blob_original.sentiment.polarity
    
    blob_processed = TextBlob(processed_text)
    textblob_polarity_processed = blob_processed.sentiment.polarity
    sid = SentimentIntensityAnalyzer()
    vader_scores = sid.polarity_scores(processed_text)
    
    weight_textblob_original = 0.2
    weight_textblob_processed = 0.3
    weight_vader = 0.5
    
    normalized_vader = (vader_scores['compound'] + 1) / 2
    normalized_textblob_original = (textblob_polarity_original + 1) / 2
    normalized_textblob_processed = (textblob_polarity_processed + 1) / 2
    
    combined_score = (
        (normalized_textblob_original * weight_textblob_original) + 
        (normalized_textblob_processed * weight_textblob_processed) + 
        (normalized_vader * weight_vader)
    )
    
    final_score = 1 + combined_score * 4
    return final_score, vader_scores, textblob_polarity_original, language, translated_text

def generate_wordcloud(text, language):
    """Génère un nuage de mots basé sur le texte"""
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    
    if language == "fr":
        stop_words = set(stopwords.words('french'))
    else:
        stop_words = set(stopwords.words('english'))
    
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white',
                          stopwords=stop_words,
                          max_words=50).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)  # Rewind buffer to the beginning
    plt.close(fig)
    return buf

def get_emoji_sentiment(score):
    """Renvoie un emoji correspondant au sentiment"""
    if score <= 2:
        return "😢 😠 👎"
    elif score < 4:
        return "😐 😑 🤔"
    else:
        return "😊 🥰 👍"

def perform_advanced_nlp_analysis(text, language):
    """Performs advanced NLP analysis using NLTK"""
    tokens = word_tokenize(text.lower())
    if language == "fr":
        stops = set(stopwords.words('french'))
    else:
        stops = set(stopwords.words('english'))
    
    words = [word for word in tokens if word.isalpha() and word not in stops]
    from nltk import FreqDist
    fdist = FreqDist(words)
    most_common = fdist.most_common(10)
    
    if language == "en":
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
        except LookupError:
            nltk.download('averaged_perceptron_tagger_eng')
        
        pos_tags = nltk.pos_tag(tokens)
        pos_counts = {
            'Nouns': len([word for word, pos in pos_tags if pos.startswith('N')]),
            'Verbs': len([word for word, pos in pos_tags if pos.startswith('V')]),
            'Adjectives': len([word for word, pos in pos_tags if pos.startswith('J')]),
            'Adverbs': len([word for word, pos in pos_tags if pos.startswith('R')])
        }
    else:
        pos_counts = None
    
    return {
        'word_count': len(tokens),
        'most_common': most_common,
        'pos_counts': pos_counts
    }

def main():
    st.markdown("""
    <style>
    .title {
        text-align: center;
        color: #1DA1F2;
    }
    .subheader {
        text-align: center;
        color: #424242;
    }
    .sentiment-positive {
        color: #4CAF50;
        font-weight: bold;
        font-size: 1.2em;
    }
    .sentiment-neutral {
        color: #FFC107;
        font-weight: bold;
        font-size: 1.2em;
    }
    .sentiment-negative {
        color: #F44336;
        font-weight: bold;
        font-size: 1.2em;
    }
    .language-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 15px;
        background-color: #E1E8ED;
        color: #657786;
        font-weight: bold;
        margin-right: 10px;
    }
    .emoji-sentiment {
        font-size: 1.5em;
        text-align: center;
        margin: 15px 0;
    }
    .result-card {
        background-color: #F7F9FA;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .translation-box {
        background-color: #f8f9fa;
        border-left: 3px solid #1DA1F2;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='title'>Analyseur de Sentiment de Tweets</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Analysez le ton émotionnel de vos tweets en français ou anglais</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        text_input = st.text_area("Entrez votre tweet ici / Enter your tweet here:", 
                                  height=100, 
                                  max_chars=280,
                                  help="Maximum 280 caractères comme sur Twitter")
    with col2:
        language_option = st.radio(
            "Langue / Language:",
            ["Auto-détection / Auto-detect", "Français", "English"],
            index=0
        )
        language_map = {
            "Auto-détection / Auto-detect": None,
            "Français": "fr",
            "English": "en"
        }
        selected_language = language_map[language_option]
    
    if text_input:
        char_count = len(text_input)
        st.caption(f"{char_count}/280 caractères")
    
    with st.expander("Exemples de tweets / Sample tweets"):
        st.markdown("### Français")
        st.markdown("- 🇫🇷 J'adore cette nouvelle mise à jour! C'est vraiment génial, tout fonctionne parfaitement.")
        st.markdown("- 🇫🇷 Je suis déçu par la qualité du service, rien ne marche comme prévu. #frustré")
        st.markdown("- 🇫🇷 Le temps est agréable aujourd'hui, ni trop chaud ni trop froid.")
        st.markdown("### English")
        st.markdown("- 🇬🇧 I absolutely love this new update! It's really amazing, everything works perfectly.")
        st.markdown("- 🇬🇧 I'm disappointed with the quality of service, nothing works as expected. #frustrated")
        st.markdown("- 🇬🇧 The weather is nice today, neither too hot nor too cold.")
    
    if st.button("Analyser / Analyze"):
        if not text_input.strip():
            st.warning("Veuillez entrer du texte à analyser / Please enter text to analyze.")
        else:
            with st.spinner('Analyse en cours... / Analysis in progress...'):
                final_score, vader_scores, textblob_polarity, detected_language, translated_text = analyze_sentiment(text_input, selected_language)
                if detected_language == "fr":
                    nlp_analysis_original = perform_advanced_nlp_analysis(clean_tweet(text_input), "fr")
                    nlp_analysis_translated = perform_advanced_nlp_analysis(translated_text, "en")
                else:
                    nlp_analysis_original = perform_advanced_nlp_analysis(clean_tweet(text_input), "en")
                    nlp_analysis_translated = None
            
            if final_score is not None:
                lang_display = "Français" if detected_language == "fr" else "English"
                st.markdown(f"<div style='text-align: center;'><span class='language-badge'>{lang_display}</span></div>", unsafe_allow_html=True)
                
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.markdown("### Résultats de l'analyse / Analysis Results")
                
                sentiment_category = ""
                sentiment_text = ""
                if final_score <= 2:
                    sentiment_text = "Négatif" if detected_language == "fr" else "Negative"
                    sentiment_category = f"<span class='sentiment-negative'>{sentiment_text}</span>"
                elif final_score < 4:
                    sentiment_text = "Neutre" if detected_language == "fr" else "Neutral"
                    sentiment_category = f"<span class='sentiment-neutral'>{sentiment_text}</span>"
                else:
                    sentiment_text = "Positif" if detected_language == "fr" else "Positive"
                    sentiment_category = f"<span class='sentiment-positive'>{sentiment_text}</span>"
                
                stars = "⭐" * round(final_score)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Sentiment:** {sentiment_category}", unsafe_allow_html=True)
                    st.markdown(f"**Score:** {stars} ({final_score:.1f}/5)")
                
                with col2:
                    emojis = get_emoji_sentiment(final_score)
                    st.markdown(f"<div class='emoji-sentiment'>{emojis}</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                tab1, tab2, tab3, tab4 = st.tabs(["Détails / Details", "Graphiques / Charts", "Nuage de mots / Word Cloud", "Analyse NLTK"])
                
                with tab1:
                    st.subheader("Tweet nettoyé / Cleaned Tweet")
                    st.markdown(f"> {clean_tweet(text_input)}")
                    if detected_language == "fr":
                        st.subheader("Traduction automatique / Auto Translation")
                        st.markdown(f"<div class='translation-box'>{translated_text}</div>", unsafe_allow_html=True)
                    
                    st.subheader("Scores VADER")
                    if detected_language == "fr":
                        st.caption("Analyse effectuée sur la traduction anglaise pour plus de précision")
                    
                    scores_display = {
                        "Positif / Positive": vader_scores['pos'],
                        "Neutre / Neutral": vader_scores['neu'],
                        "Négatif / Negative": vader_scores['neg'],
                        "Composé / Compound": vader_scores['compound']
                    }
                    
                    for key, value in scores_display.items():
                        st.markdown(f"**{key}:** {value:.3f}")
                    
                    st.subheader("TextBlob Polarité / Polarity")
                    if detected_language == "fr":
                        st.markdown(f"**Polarité (texte original) / Polarity (original text):** {textblob_polarity:.3f}")
                        st.caption("La polarité est aussi calculée sur la traduction anglaise et contribue au score final")
                    else:
                        st.markdown(f"**Polarité / Polarity:** {textblob_polarity:.3f} (-1 très négatif / very negative, +1 très positif / very positive)")
                
                with tab2:
                    st.subheader("Scores VADER")
                    if detected_language == "fr":
                        st.caption("Basé sur la traduction anglaise")
                    
                    vader_df = pd.DataFrame({
                        'Type': ['Positif / Positive', 'Neutre / Neutral', 'Négatif / Negative', 'Composé / Compound'],
                        'Score': [
                            vader_scores['pos'], 
                            vader_scores['neu'], 
                            vader_scores['neg'], 
                            vader_scores['compound']
                        ]
                    })
                    fig_vader = px.bar(vader_df, x='Type', y='Score', title="Scores VADER")
                    st.plotly_chart(fig_vader)
                    
                    st.subheader("TextBlob Polarité / Polarity")
                    polarity_data = pd.DataFrame({
                        'range': ['Négatif / Negative', 'Neutre / Neutral', 'Positif / Positive'],
                        'values': [
                            max(0, -textblob_polarity), 
                            max(0, 1 - abs(textblob_polarity)), 
                            max(0, textblob_polarity)
                        ]
                    })
                    fig_polarity = px.bar(polarity_data, x='range', y='values', title="TextBlob Polarité")
                    st.plotly_chart(fig_polarity)
                
                with tab3:
                    if detected_language == "fr" and translated_text:
                        wordcloud_col1, wordcloud_col2 = st.columns(2)
                        with wordcloud_col1:
                            st.subheader("Nuage de mots (Français)")
                            wordcloud_buffer_fr = generate_wordcloud(text_input, "fr")
                            st.image(wordcloud_buffer_fr, caption="Nuage de mots (Français)")
                        with wordcloud_col2:
                            st.subheader("Word Cloud (English)")
                            wordcloud_buffer_en = generate_wordcloud(translated_text, "en")
                            st.image(wordcloud_buffer_en, caption="Word Cloud (English)")
                    else:
                        wordcloud_buffer = generate_wordcloud(text_input, detected_language)
                        st.image(wordcloud_buffer, caption="Nuage de mots / Word Cloud")
                
                with tab4:
                    st.subheader("Analyse NLTK avancée / Advanced NLTK Analysis")
                    st.markdown("#### Mots les plus fréquents / Most frequent words")
                    if nlp_analysis_original['most_common']:
                        freq_df = pd.DataFrame(nlp_analysis_original['most_common'], columns=['Mot / Word', 'Fréquence / Frequency'])
                        st.table(freq_df)
                    
                    if nlp_analysis_original['pos_counts']:
                        st.markdown("#### Parties du discours / Parts of Speech")
                        pos_df = pd.DataFrame({
                            'Type': ['Noms / Nouns', 'Verbes / Verbs', 'Adjectifs / Adjectives', 'Adverbes / Adverbs'],
                            'Nombre / Count': [
                                nlp_analysis_original['pos_counts']['Nouns'],
                                nlp_analysis_original['pos_counts']['Verbs'],
                                nlp_analysis_original['pos_counts']['Adjectives'],
                                nlp_analysis_original['pos_counts']['Adverbs']
                            ]
                        })
                        fig_pos = px.bar(pos_df, x='Type', y='Nombre / Count', title="Parties du discours")
                        st.plotly_chart(fig_pos)
                    
                    if nlp_analysis_translated:
                        st.markdown("#### Comparaison des mots fréquents / Frequent Words Comparison")
                        st.markdown("**Français vs English Translation**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Français - Original**")
                            if nlp_analysis_original['most_common']:
                                freq_df_fr = pd.DataFrame(nlp_analysis_original['most_common'][:5], 
                                                     columns=['Mot', 'Fréquence'])
                                st.table(freq_df_fr)
                        with col2:
                            st.markdown("**English - Translation**")
                            if nlp_analysis_translated['most_common']:
                                freq_df_en = pd.DataFrame(nlp_analysis_translated['most_common'][:5], 
                                                     columns=['Word', 'Frequency'])
                                st.table(freq_df_en)
                
                if final_score <= 3:
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.subheader("💡 Conseils pour améliorer le sentiment")
                    st.markdown("- Utilisez des mots positifs comme 'excellent', 'fantastique', 'merveilleux'")
                    st.markdown("- Évitez les négations et les mots négatifs")
                    st.markdown("- Exprimez de la gratitude ou de l'enthousiasme")
                    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()