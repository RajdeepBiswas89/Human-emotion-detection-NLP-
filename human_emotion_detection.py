import streamlit as st
import pickle
import numpy as np
from PIL import Image
import time

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #F5F5F5;
    }
    .title {
        color: #4B0082;
        text-align: center;
        font-size: 3em;
        margin-bottom: 0.5em;
    }
    .subheader {
        color: #6A5ACD;
        font-size: 1.5em;
        margin-bottom: 1em;
    }
    .sidebar .sidebar-content {
        background-color: #E6E6FA;
    }
    .stButton>button {
        background-color: #9370DB;
        color: white;
        border-radius: 5px;
        padding: 0.5em 1em;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #8A2BE2;
    }
    .result-box {
        padding: 2em;
        border-radius: 10px;
        background-color: #FFF;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-top: 2em;
    }
    .emoji {
        font-size: 2em;
        margin-right: 0.5em;
    }
</style>
""", unsafe_allow_html=True)

# Load models and vectorizers
@st.cache_resource
def load_models():
    with open('bow_vectorizer.pkl', 'rb') as f:
        bow_vectorizer = pickle.load(f)
    with open('bow_naive_bayes.pkl', 'rb') as f:
        bow_nb = pickle.load(f)
    with open('bow_logistic_regression.pkl', 'rb') as f:
        bow_lr = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('tfidf_naive_bayes.pkl', 'rb') as f:
        tfidf_nb = pickle.load(f)
    with open('tfidf_logistic_regression.pkl', 'rb') as f:
        tfidf_lr = pickle.load(f)
    with open('emotion_mapping.pkl', 'rb') as f:
        emotion_mapping = pickle.load(f)
    return bow_vectorizer, bow_nb, bow_lr, tfidf_vectorizer, tfidf_nb, tfidf_lr, emotion_mapping

bow_vectorizer, bow_nb, bow_lr, tfidf_vectorizer, tfidf_nb, tfidf_lr, emotion_mapping = load_models()

# Reverse emotion mapping for display
reverse_emotion_mapping = {v: k for k, v in emotion_mapping.items()}

# Emoji mapping
emoji_dict = {
    'anger': '😠',
    'fear': '😨',
    'joy': '😊',
    'love': '❤️',
    'sadness': '😢',
    'surprise': '😮'
}

# App layout
st.markdown('<div class="title">Emotion Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Discover the emotion behind your text</div>', unsafe_allow_html=True)

# Sidebar for model selection
with st.sidebar:
    st.header("Settings")
    model_type = st.radio(
        "Select Model Type:",
        ('Bag of Words (BOW)', 'TF-IDF')
    )
    algorithm = st.radio(
        "Select Algorithm:",
        ('Naive Bayes', 'Logistic Regression')
    )
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("This app classifies text into emotions using machine learning models trained on emotional text data.")

# Main content area
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area("Enter your text here:", height=150, 
                             placeholder="Type something like 'I feel happy today!'")
with col2:
    st.markdown("### Examples:")
    st.markdown("- I'm so excited!")
    st.markdown("- This makes me angry")
    st.markdown("- I feel lonely today")

if st.button("Analyze Emotion"):
    if user_input:
        with st.spinner('Analyzing...'):
            time.sleep(1)  # Simulate processing time
            
            # Process input based on selected model
            if model_type == 'Bag of Words (BOW)':
                input_vec = bow_vectorizer.transform([user_input])
                if algorithm == 'Naive Bayes':
                    prediction = bow_nb.predict(input_vec)[0]
                    confidence = bow_nb.predict_proba(input_vec)[0][prediction]
                else:
                    prediction = bow_lr.predict(input_vec)[0]
                    confidence = bow_lr.predict_proba(input_vec)[0][prediction]
            else:  # TF-IDF
                input_vec = tfidf_vectorizer.transform([user_input])
                if algorithm == 'Naive Bayes':
                    prediction = tfidf_nb.predict(input_vec)[0]
                    confidence = tfidf_nb.predict_proba(input_vec)[0][prediction]
                else:
                    prediction = tfidf_lr.predict(input_vec)[0]
                    confidence = tfidf_lr.predict_proba(input_vec)[0][prediction]
            
            emotion = reverse_emotion_mapping[prediction]
            emoji = emoji_dict.get(emotion, '❓')
            
            # Display result with animation
            with st.empty():
                for i in range(3):
                    st.markdown(f"<div style='text-align: center; font-size: 1.5em;'>Analyzing{'.' * i}</div>", 
                                unsafe_allow_html=True)
                    time.sleep(0.3)
                
                st.markdown(f"""
                <div class="result-box">
                    <h2 style='text-align: center; color: #4B0082;'>
                        <span class="emoji">{emoji}</span>
                        {emotion.capitalize()}
                    </h2>
                    <p style='text-align: center;'>
                        Model: {model_type} with {algorithm}<br>
                        Confidence: {confidence:.2%}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to analyze!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit | Machine Learning Emotion Classifier</p>
</div>
""", unsafe_allow_html=True)