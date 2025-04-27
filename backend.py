from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import research_agent, usecase_agent, resource_agent, validation_agent
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Increase timeout settings
TIMEOUT = 300  # 5 minutes

class ResearchRequest(BaseModel):
    company_or_industry_name: str
    user_name: str = "User"

class UseCaseRequest(BaseModel):
    industry_info: dict
    user_name: str = "User"

class ResourceRequest(BaseModel):
    usecases: list
    user_name: str = "User"

class ValidateRequest(BaseModel):
    usecases: list
    resources: list
    user_name: str = "User"

class FullPipelineRequest(BaseModel):
    company_or_industry_name: str
    user_name: str = "User"

@app.post("/research")
async def research(req: ResearchRequest):
    try:
        result = await asyncio.wait_for(
            research_agent.run(req.company_or_industry_name),
            timeout=TIMEOUT
        )
        return {"result": result}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Research operation timed out")
    except Exception as e:
        logger.error(f"Error in research endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/usecases")
async def usecases(req: UseCaseRequest):
    try:
        result = await asyncio.wait_for(
            usecase_agent.run(req.industry_info),
            timeout=TIMEOUT
        )
        return {"result": result}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Use case generation timed out")
    except Exception as e:
        logger.error(f"Error in usecases endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/resources")
async def resources(req: ResourceRequest):
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(resource_agent.run, req.usecases),
            timeout=TIMEOUT
        )
        return {"result": result}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Resource collection timed out")
    except Exception as e:
        logger.error(f"Error in resources endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate")
async def validate(req: ValidateRequest):
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(validation_agent.run, req.usecases, req.resources, req.user_name),
            timeout=TIMEOUT
        )
        return {"result": result}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Validation timed out")
    except Exception as e:
        logger.error(f"Error in validate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/full_pipeline")
async def full_pipeline(req: FullPipelineRequest):
    try:
        # Run all operations with timeout
        industry_info = await asyncio.wait_for(
            research_agent.run(req.company_or_industry_name),
            timeout=TIMEOUT
        )
        
        usecases = await asyncio.wait_for(
            usecase_agent.run(industry_info),
            timeout=TIMEOUT
        )
        
        resources = await asyncio.wait_for(
            asyncio.to_thread(resource_agent.run, usecases),
            timeout=TIMEOUT
        )
        
        validated = await asyncio.wait_for(
            asyncio.to_thread(validation_agent.run, usecases, resources, user_name=req.user_name),
            timeout=TIMEOUT
        )
        
        return {
            "industry_info": industry_info,
            "usecases": usecases,
            "resources": resources,
            "validated": validated
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Pipeline operation timed out")
    except Exception as e:
        logger.error(f"Error in full pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 