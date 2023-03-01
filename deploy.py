import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import rake_nltk as rake
import nltk 
nltk.download('stopwords')

# Load data
data = pd.read_csv('E:/NLP Project_Research Paper Recommendation/dataset.csv')

# Extract keywords from abstracts using RAKE
rake_object = rake.Rake()
keywords = []
for abstract in data['abstracts']:
    rake_keywords = rake_object.extract_keywords_from_text(abstract)
    keyword_scores = rake_object.get_word_degrees()
    keywords.append(list(keyword_scores.keys()))

# Convert keywords list to string
keywords_str = []
for k in keywords:
    keywords_str.append(' '.join(k))

# Vectorize keywords using CountVectorizer
vectorizer = CountVectorizer()
vectorized_keywords = vectorizer.fit_transform(keywords_str)

# Perform Truncated SVD
svd = TruncatedSVD(n_components=50)
reduced_keywords = svd.fit_transform(vectorized_keywords)

# Define function to get recommendations
def get_recommendations(user_input, data, reduced_data, n_recommendations=5):
    # Vectorize user input abstract
    vectorized_user_input = vectorizer.transform([user_input])
    # Reduce user input abstract to same dimensionality as research paper abstracts
    reduced_user_input = svd.transform(vectorized_user_input)
    # Calculate cosine similarity between user input and research paper abstracts
    similarities = cosine_similarity(reduced_user_input, reduced_data)
    # Get indices of most similar papers
    indices = similarities.argsort()[0][::-1][:n_recommendations]
    # Return recommended papers
    return data.iloc[indices]

def get_abst(name):
    description = data.loc[data['titles'] == name, 'abstracts'].iloc[0]
    return description

# Define Streamlit app
st.title('Research Paper Recommender')

# Get user input
option = st.selectbox(
    'List of Scientific Research Papers',
    data['titles'])

if st.button('Submit'):
    # Get recommendations
    abstr=get_abst(option)
    recommendations = get_recommendations(abstr, data, reduced_keywords)
    # Display recommendations
    st.write('Top 5 Recommendations:')
    for i, row in recommendations.iterrows():
        st.write(f'{i + 1}. {row["titles"]}')
        st.write(row['abstracts'])

