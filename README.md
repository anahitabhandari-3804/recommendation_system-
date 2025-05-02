# SHL Assessment Recommendation System

This project implements an AI-powered assessment recommendation system using FastAPI for the backend API and Streamlit for the frontend interface. The system leverages Natural Language Processing (NLP) and machine learning techniques to recommend relevant SHL assessments based on a given job description or talent requirement.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Setup and Installation](#setup-and-installation)
4. [Running the Application](#running-the-application)
5. [API Endpoints](#api-endpoints)
6. [Frontend Interface (Streamlit)](#frontend-interface-streamlit)
7. [Deployment](#deployment)
8. [License](#license)

## Project Overview

This system provides personalized assessment recommendations based on the job descriptions or queries provided by the user. The backend API is built with **FastAPI**, and the frontend interface is built with **Streamlit**.

The system uses a pre-defined catalog of SHL assessments and matches them with user queries using text similarity techniques such as **TF-IDF** and advanced NLP models.

## Technologies Used

- **FastAPI**: For building the backend API to handle assessment recommendations.
- **Streamlit**: For creating a user-friendly frontend interface.
- **TF-IDF**: For text similarity matching between job descriptions and assessments.
- **Google Gemini**: For enhanced natural language processing (NLP).
- **Uvicorn**: ASGI server to run the FastAPI app.
- **Pydantic**: For data validation in FastAPI.

## Setup and Installation

### Prerequisites

Ensure you have Python 3.7+ installed. You can download Python from [python.org](https://www.python.org/).

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/recommendation_system.git
   cd shl-assessment-recommender
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Start the FastAPI Backend
```bash
uvicorn app.main:app --reload
```
The backend API will be available at `http://localhost:8000`

### Start the Streamlit Frontend
```bash
streamlit run streamlit_app.py
```
The frontend interface will be available at the URL provided by Streamlit (typically `http://localhost:8501`).

## API Endpoints

- `POST /recommend`: Accepts a JSON job description and returns recommended assessments.
  ```json
  {
    "description": "We are looking for a software engineer with strong Python and problem-solving skills."
  }
  ```
- Returns:
  ```json
  {
    "recommendations": ["Python Programming Assessment", "Logical Reasoning Test"]
  }
  ```

## Frontend Interface (Streamlit)

- Upload or type a job description.
- View the top recommended SHL assessments.
- Simple and clean UI optimized for recruiters and hiring managers.

## Deployment

the deployment link:https://dehdppoy4atrueked8u2kd.streamlit.app/
the sample input and output:
![image](https://github.com/user-attachments/assets/73d159ec-25af-4298-8037-98776c6886c1)
![image](https://github.com/user-attachments/assets/6f8efa6c-ae08-4dd3-a99e-c95cd5feab5f)


## License

This project is licensed under the MIT License. See the LICENSE file for more details.
