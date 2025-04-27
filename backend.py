from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agents import research_agent, usecase_agent, resource_agent, validation_agent
import asyncio
import logging
import redis
from typing import Dict, List, Optional
import os

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

# Timeout settings
RESEARCH_TIMEOUT = 180  # 3 minutes
USECASE_TIMEOUT = 120   # 2 minutes
RESOURCE_TIMEOUT = 120  # 2 minutes
VALIDATION_TIMEOUT = 60 # 1 minute

# Redis connection
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=int(os.getenv('REDIS_DB', 0)),
        decode_responses=True
    )
    redis_client.ping()  # Test connection
except redis.ConnectionError as e:
    logger.error(f"Redis connection failed: {str(e)}")
    redis_client = None

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

async def _handle_timeout(operation: str, timeout: int):
    """Handle timeout for operations"""
    try:
        await asyncio.sleep(timeout)
        raise asyncio.TimeoutError(f"{operation} operation timed out after {timeout} seconds")
    except asyncio.CancelledError:
        pass

@app.post("/research")
async def research(req: ResearchRequest):
    try:
        result = await asyncio.wait_for(
            research_agent.run(req.company_or_industry_name),
            timeout=RESEARCH_TIMEOUT
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
            timeout=USECASE_TIMEOUT
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
            timeout=RESOURCE_TIMEOUT
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
            timeout=VALIDATION_TIMEOUT
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
        # Run all operations with timeout and error handling
        results = {}
        errors = {}

        # Research phase
        try:
            results['industry_info'] = await asyncio.wait_for(
                research_agent.run(req.company_or_industry_name),
                timeout=RESEARCH_TIMEOUT
            )
        except Exception as e:
            errors['research'] = str(e)
            results['industry_info'] = None

        # Use case phase
        if results['industry_info']:
            try:
                results['usecases'] = await asyncio.wait_for(
                    usecase_agent.run(results['industry_info']),
                    timeout=USECASE_TIMEOUT
                )
            except Exception as e:
                errors['usecases'] = str(e)
                results['usecases'] = None

        # Resource phase
        if results.get('usecases'):
            try:
                results['resources'] = await asyncio.wait_for(
                    asyncio.to_thread(resource_agent.run, results['usecases']),
                    timeout=RESOURCE_TIMEOUT
                )
            except Exception as e:
                errors['resources'] = str(e)
                results['resources'] = None

        # Validation phase
        if results.get('usecases') and results.get('resources'):
            try:
                results['validated'] = await asyncio.wait_for(
                    asyncio.to_thread(validation_agent.run, results['usecases'], results['resources'], req.user_name),
                    timeout=VALIDATION_TIMEOUT
                )
            except Exception as e:
                errors['validation'] = str(e)
                results['validated'] = None

        # Return results with any errors
        return {
            "results": results,
            "errors": errors if errors else None
        }

    except Exception as e:
        logger.error(f"Error in full pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 