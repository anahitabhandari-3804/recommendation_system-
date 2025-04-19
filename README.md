# recommendation_system
# SHL Assessment Recommendation System

This project implements an AI-powered assessment recommendation system using FastAPI for the backend API and Streamlit for the frontend interface. The system leverages Natural Language Processing (NLP) and machine learning techniques to recommend relevant SHL assessments based on a given job description or talent requirement.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Setup and Installation](#setup-and-installation)
4. [Running the Application](#running-the-application)
5. [API Endpoints](#api-endpoints)
6. [Frontend Interface (Streamlit)](#frontend-interface-streamlit)
7. [License](#license)

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

