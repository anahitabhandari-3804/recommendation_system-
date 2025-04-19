import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Example simplified assessments catalog
assessments_catalog = [
    {"name": "Java Programming Test", "description": "Tests proficiency in Java programming for backend developers.", "duration": 30, "remote_testing": "Yes", "adaptive_support": "Yes", "test_type": ["Cognitive", "Technical"], "url": "https://www.shl.com/java-test"},
    {"name": "Personality & Cognitive Skills Test", "description": "Assesses cognitive abilities and personality traits for management roles.", "duration": 40, "remote_testing": "Yes", "adaptive_support": "Yes", "test_type": ["Cognitive", "Personality"], "url": "https://www.shl.com/personality-test"},
    {"name": "Data Science Assessment", "description": "Assesses skills in machine learning, data analysis, and Python programming.", "duration": 45, "remote_testing": "Yes", "adaptive_support": "Yes", "test_type": ["Cognitive", "Technical"], "url": "https://www.shl.com/data-science-test"},
    {"name": "Sales Aptitude Test", "description": "Assesses aptitude for sales, communication skills, and client interaction.", "duration": 30, "remote_testing": "Yes", "adaptive_support": "No", "test_type": ["Cognitive", "Personality"], "url": "https://www.shl.com/sales-test"},
    {"name": "Managerial Skills Assessment", "description": "Tests leadership, decision-making, and problem-solving abilities.", "duration": 50, "remote_testing": "Yes", "adaptive_support": "Yes", "test_type": ["Cognitive", "Personality"], "url": "https://www.shl.com/managerial-test"},
    # Add more assessments as needed
]

st.set_page_config(page_title="SHL AI Recommender", layout="wide", page_icon="🔍")

# --- Custom Style ---
st.markdown("""
    <style>
    .title-text {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1F4E79;
    }
    .subtitle-text {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 1rem;
    }
    .recommend-box {
        border: 1px solid #e6e6e6;
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        background-color: #2C3E50;  /* Dark background */
        color: #ECF0F1;  /* Light font color */
    }
    .recommend-box a {
        color: #ECF0F1;  /* Link color */
        text-decoration: none;
    }
    .recommend-box a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown('<div class="title-text">🔍 SHL Assessment AI Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Get personalized assessment suggestions using AI & NLP</div>', unsafe_allow_html=True)

# --- Input ---
st.markdown("### 📄 Paste your job description or hiring requirement")
query = st.text_area("**Job Description or Talent Requirement:**", placeholder="Example: We’re hiring a Python backend engineer with strong analytical & cognitive skills. Assessment should be under 40 minutes.", height=180)

# --- Recommendation Logic ---
def filter_relevant_assessments(query, assessments_catalog):
    # Using TF-IDF Vectorizer to match the job description with the catalog
    corpus = [assessment['description'] for assessment in assessments_catalog]
    corpus.append(query)  # Add query to the corpus for comparison
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Compute cosine similarity between the job description (last item) and the catalog items
    cosine_similarities = np.dot(tfidf_matrix[-1], tfidf_matrix[:-1].T).toarray().flatten()
    
    # Get the assessments that have the highest similarity score
    relevant_assessments = []
    for i, similarity in enumerate(cosine_similarities):
        if similarity > 0.1:  # Adjust threshold as needed
            relevant_assessments.append(assessments_catalog[i])

    return relevant_assessments

if st.button("🚀 Recommend Assessments"):
    if not query.strip():
        st.error("Please enter a job description or query.")
    else:
        with st.spinner("Generating recommendations using Gemini & TF-IDF..."):
            # Filter relevant assessments based on the query
            relevant_recommendations = filter_relevant_assessments(query, assessments_catalog)

        if not relevant_recommendations:
            st.warning("No suitable assessments found based on your criteria.")
        else:
            st.success(f"✅ Found {len(relevant_recommendations)} recommended assessments!")
            for rec in relevant_recommendations:
                with st.container():
                    st.markdown(f"""
                    <div class="recommend-box">
                        <h4>{rec['name']}</h4>
                        <p>{rec['description']}</p>
                        <ul>
                            <li><strong>Test Type:</strong> {", ".join(rec['test_type'])}</li>
                            <li><strong>Duration:</strong> {rec['duration']} mins</li>
                            <li><strong>Remote Testing:</strong> {rec['remote_testing']}</li>
                            <li><strong>Adaptive/IRT:</strong> {rec['adaptive_support']}</li>
                        </ul>
                        <a href="{rec['url']}" target="_blank">🔗 Learn More</a>
                    </div>
                    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<hr>
<div style='text-align: center; font-size: 0.9rem; color: gray;'>
    Built with ❤️ using Streamlit, FastAPI, Gemini, and TF-IDF by <b>[Your Name]</b><br>
    For internship demonstration & smart hiring recommendations.
</div>
""", unsafe_allow_html=True)
