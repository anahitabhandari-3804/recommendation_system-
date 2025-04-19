from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example SHL assessment catalog (simplified)
assessments_catalog = [
    {"name": "Java Programming Test", 
     "description": "Tests proficiency in Java programming for backend developers.", 
     "duration": 30, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Cognitive", "Technical"], 
     "url": "https://www.shl.com/en/assessments/java-programming-test/"},
    
    {"name": "Personality & Cognitive Skills Test", 
     "description": "Assesses cognitive abilities and personality traits for management roles.", 
     "duration": 40, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Cognitive", "Personality"], 
     "url": "https://www.shl.com/en/assessments/personality-cognitive-skills-test/"},
    
    {"name": "Data Science & Machine Learning Test", 
     "description": "Evaluates skills in data analysis, machine learning, and algorithm design for data scientists.", 
     "duration": 45, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Cognitive", "Technical"], 
     "url": "https://www.shl.com/en/assessments/data-science-machine-learning-test/"},
    
    {"name": "Front-End Development Test", 
     "description": "Assesses proficiency in front-end technologies such as HTML, CSS, and JavaScript.", 
     "duration": 30, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Technical"], 
     "url": "https://www.shl.com/en/assessments/front-end-development-test/"},
    
    {"name": "Project Management Skills Test", 
     "description": "Tests knowledge in project management methodologies, such as Agile and Waterfall.", 
     "duration": 35, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Cognitive", "Behavioral"], 
     "url": "https://www.shl.com/en/assessments/project-management-skills-test/"},
    
    {"name": "Sales & Negotiation Skills Test", 
     "description": "Evaluates the ability to close deals and negotiate effectively in a sales environment.", 
     "duration": 30, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Cognitive", "Behavioral"], 
     "url": "https://www.shl.com/en/assessments/sales-negotiation-skills-test/"},
    
    {"name": "Cybersecurity Knowledge Test", 
     "description": "Tests proficiency in cybersecurity principles, tools, and threat mitigation techniques.", 
     "duration": 40, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Cognitive", "Technical"], 
     "url": "https://www.shl.com/en/assessments/cybersecurity-knowledge-test/"},
    
    {"name": "Customer Support Skills Test", 
     "description": "Assesses problem-solving and communication skills for customer support roles.", 
     "duration": 30, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Behavioral"], 
     "url": "https://www.shl.com/en/assessments/customer-support-skills-test/"},
    
    {"name": "HR Management Skills Test", 
     "description": "Evaluates knowledge in human resources management, including recruitment and employee relations.", 
     "duration": 35, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Cognitive", "Behavioral"], 
     "url": "https://www.shl.com/en/assessments/hr-management-skills-test/"},
    
    {"name": "Marketing Strategy Test", 
     "description": "Tests skills in creating effective marketing strategies and understanding market trends.", 
     "duration": 40, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Cognitive", "Behavioral"], 
     "url": "https://www.shl.com/en/assessments/marketing-strategy-test/"},
    
    {"name": "Python Programming Test", 
     "description": "Assesses proficiency in Python programming and problem-solving abilities for backend developers.", 
     "duration": 40, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Cognitive", "Technical"], 
     "url": "https://www.shl.com/en/assessments/python-programming-test/"},
    
    {"name": "Cloud Computing & DevOps Test", 
     "description": "Tests knowledge and skills in cloud platforms, infrastructure management, and DevOps practices.", 
     "duration": 50, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Technical"], 
     "url": "https://www.shl.com/en/assessments/cloud-computing-devops-test/"},
    
    {"name": "Digital Marketing Test", 
     "description": "Assesses expertise in digital marketing strategies, SEO, SEM, and social media marketing.", 
     "duration": 40, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Cognitive", "Behavioral"], 
     "url": "https://www.shl.com/en/assessments/digital-marketing-test/"},
    
    {"name": "Business Analytics Test", 
     "description": "Tests skills in analyzing business data, using analytics tools, and generating insights for decision-making.", 
     "duration": 45, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Cognitive", "Technical"], 
     "url": "https://www.shl.com/en/assessments/business-analytics-test/"},
    
    {"name": "Financial Analysis & Accounting Test", 
     "description": "Assesses skills in financial analysis, accounting principles, and reporting for finance roles.", 
     "duration": 45, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Cognitive", "Technical"], 
     "url": "https://www.shl.com/en/assessments/financial-analysis-accounting-test/"},
    
    {"name": "UI/UX Design Skills Test", 
     "description": "Evaluates proficiency in UI/UX design, wireframing, and user research for design professionals.", 
     "duration": 40, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Technical"], 
     "url": "https://www.shl.com/en/assessments/ui-ux-design-skills-test/"},
    
    {"name": "Product Management Skills Test", 
     "description": "Assesses knowledge of product lifecycle management, market research, and customer feedback integration.", 
     "duration": 50, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Cognitive", "Behavioral"], 
     "url": "https://www.shl.com/en/assessments/product-management-skills-test/"},
    
    {"name": "Business Intelligence & Data Visualization Test", 
     "description": "Tests ability to work with business intelligence tools, analyze data, and visualize business insights.", 
     "duration": 45, 
     "remote_testing": "Yes", 
     "adaptive_support": "Yes", 
     "test_type": ["Cognitive", "Technical"], 
     "url": "https://www.shl.com/en/assessments/bi-data-visualization-test/"}
]

# Recommendation function based on job description query
def recommend_assessments(query):
    # Prepare the list of descriptions from the assessment catalog
    descriptions = [assessment["description"] for assessment in assessments_catalog]
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(descriptions + [query])  # Include query at the end
    
    # Calculate cosine similarity between the query and all descriptions
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    
    # Sort by highest similarity and select top recommendations
    sorted_idx = cosine_sim.argsort()[-10:][::-1]  # Get top 10
    recommendations = []
    
    for idx in sorted_idx:
        rec = assessments_catalog[idx]
        recommendations.append({
            "name": rec["name"],
            "description": rec["description"],
            "duration": rec["duration"],
            "remote_testing": rec["remote_testing"],
            "adaptive_support": rec["adaptive_support"],
            "test_type": rec["test_type"],
            "url": rec["url"]
        })
    
    return recommendations
