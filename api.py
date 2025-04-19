from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from shl_recommendation_system import recommend_assessments
import logging

app = FastAPI()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request body model for job descriptions
class JobDescriptionRequest(BaseModel):
    query: str

# Health Check Endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Assessment Recommendation Endpoint
@app.post("/recommend")
def recommend(request: JobDescriptionRequest):
    try:
        query = request.query
        if not query.strip():
            raise HTTPException(status_code=400, detail="Job description cannot be empty.")

        recommendations = recommend_assessments(query)
        if not recommendations:
            return {"message": "No suitable assessments found."}

        return {"recommendations": recommendations}
    
    except Exception as e:
        logger.error(f"Error in recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
