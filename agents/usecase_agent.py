import os
import json
import logging
import redis
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import re
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Initialize Redis with connection pool for better performance
redis_pool = redis.ConnectionPool.from_url(REDIS_URL)
r = redis.Redis(connection_pool=redis_pool)

class UsecaseAgent:
    def __init__(self):
        self.cache_ttl = 86400  # 24 hours
        #self._initialize_models()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _initialize_models(self):
        """
        Initialize ML models for analysis.
        """
        try:
            # Initialize sentence transformer for semantic search
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("ML models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ML models: {str(e)}")
            raise

    async def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama with the local DeepSeek model.
        """
        try:
            response = ollama.generate(
                model='deepseek-r1',
                prompt=prompt
            )
            return response.response
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            raise

    async def run(self, industry_research: Dict) -> Dict:
        """
        Main entry point for the use case agent.
        """
        cache_key = f"usecase:{industry_research['industry']}"
        
        # Try to get from cache first
        cached_result = await self._get_from_cache(cache_key)
        if cached_result:
            logger.info(f"Retrieved cached use cases for {industry_research['industry']}")
            return cached_result

        try:
            # Generate use cases
            use_cases = await self._generate_use_cases(industry_research)
            if not use_cases:
                return await self._get_fallback_result(industry_research)
            
            # Validate use cases
            validated_cases = await self._validate_use_cases(use_cases, industry_research)
            if not validated_cases:
                return await self._get_fallback_result(industry_research)
            
            # Score use cases
            scored_cases = await self._score_use_cases(validated_cases, industry_research)
            
            # Cache the results
            await self._cache_result(cache_key, scored_cases)
            
            return scored_cases

        except Exception as e:
            logger.error(f"Error in use case agent: {str(e)}")
            return await self._get_fallback_result(industry_research)

    async def _generate_use_cases(self, industry_research: Dict) -> List[Dict]:
        """
        Generate use cases using Ollama with local DeepSeek model.
        """
        try:
            # Prepare prompt
            prompt = f"""
            Based on the following industry research, generate 5 innovative AI and Generative AI use cases in JSON format:
            
            Industry: {industry_research['industry']}
            Segment: {industry_research['segment']}
            Key Offerings: {', '.join(industry_research['key_offerings'])}
            Strategic Focus: {', '.join(industry_research['strategic_focus'])}
            Vision: {industry_research['vision']}
            Products: {', '.join(industry_research['products'])}
            
            Format the response as a JSON array:
            [
                {{
                    "title": "Use Case Title",
                    "description": "Brief description",
                    "business_value": "Specific benefits",
                    "technical_approach": "Implementation details",
                    "innovation_potential": "Unique aspects"
                }}
            ]
            """
            
            # Generate use cases
            generated_text = await self._call_ollama(prompt)
            
            try:
                # Parse JSON response
                use_cases = json.loads(generated_text)
                return use_cases
            except json.JSONDecodeError:
                # If JSON parsing fails, try to parse the text
                use_cases = []
                for text in generated_text.split('\n\n'):
                    use_case = self._parse_use_case(text)
                    if use_case:
                        use_cases.append(use_case)
                return use_cases
        except Exception as e:
            logger.error(f"Error generating use cases: {str(e)}")
            return []

    def _parse_use_case(self, text: str) -> Optional[Dict]:
        """
        Parse generated text into structured use case.
        """
        try:
            # Extract sections using regex
            title_match = re.search(r"Title:\s*(.*?)(?=\n|$)", text)
            desc_match = re.search(r"Description:\s*(.*?)(?=\n|$)", text)
            value_match = re.search(r"Business Value:\s*(.*?)(?=\n|$)", text)
            approach_match = re.search(r"Technical Approach:\s*(.*?)(?=\n|$)", text)
            innovation_match = re.search(r"Innovation Potential:\s*(.*?)(?=\n|$)", text)
            
            if not all([title_match, desc_match, value_match, approach_match, innovation_match]):
                return None
            
            return {
                'title': title_match.group(1).strip(),
                'description': desc_match.group(1).strip(),
                'business_value': value_match.group(1).strip(),
                'technical_approach': approach_match.group(1).strip(),
                'innovation_potential': innovation_match.group(1).strip(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error parsing use case: {str(e)}")
            return None

    async def _validate_use_cases(self, use_cases: List[Dict], industry_research: Dict) -> List[Dict]:
        """
        Validate use cases for relevance and feasibility.
        """
        validated_cases = []
        
        for use_case in use_cases:
            try:
                # Check relevance
                relevance_score = await self._check_relevance(use_case, industry_research)
                
                # Check feasibility
                feasibility_score = await self._check_feasibility(use_case)
                
                if relevance_score > 0.7 and feasibility_score > 0.7:
                    use_case['relevance_score'] = relevance_score
                    use_case['feasibility_score'] = feasibility_score
                    validated_cases.append(use_case)
            except Exception as e:
                logger.error(f"Error validating use case: {str(e)}")
                continue
        
        return validated_cases

    async def _check_relevance(self, use_case: Dict, industry_research: Dict) -> float:
        """
        Check use case relevance to industry.
        """
        try:
            # Generate embeddings
            use_case_embedding = self.sentence_model.encode(use_case['description'])
            industry_embedding = self.sentence_model.encode(industry_research['industry'])
            
            # Calculate similarity
            similarity = util.pytorch_cos_sim(use_case_embedding, industry_embedding)[0][0].item()
            
            return similarity
        except Exception as e:
            logger.error(f"Error checking relevance: {str(e)}")
            return 0.0

    async def _check_feasibility(self, use_case: Dict) -> float:
        """
        Check use case technical feasibility using Ollama.
        """
        try:
            # Prepare prompt
            prompt = f"""
            Analyze the technical feasibility of this use case and respond with just a number between 0 and 1:
            
            Title: {use_case['title']}
            Description: {use_case['description']}
            Technical Approach: {use_case['technical_approach']}
            
            Consider:
            1. Technical complexity
            2. Resource requirements
            3. Implementation challenges
            4. Current technology readiness
            
            Respond with just the numerical score.
            """
            
            # Call Ollama
            score_text = await self._call_ollama(prompt)
            
            # Extract score from response
            score_match = re.search(r"(\d+\.?\d*)", score_text)
            if score_match:
                return float(score_match.group(1))
            return 0.0
        except Exception as e:
            logger.error(f"Error checking feasibility: {str(e)}")
            return 0.0

    async def _score_use_cases(self, use_cases: List[Dict], industry_research: Dict) -> Dict:
        """
        Score use cases based on multiple criteria.
        """
        try:
            scored_cases = []
            
            for use_case in use_cases:
                # Calculate business value score
                business_value_score = await self._calculate_business_value_score(use_case, industry_research)
                
                # Calculate innovation score
                innovation_score = await self._calculate_innovation_score(use_case)
                
                # Calculate technical complexity score
                complexity_score = await self._calculate_complexity_score(use_case)
                
                # Calculate overall score
                overall_score = (
                    use_case['relevance_score'] * 0.3 +
                    use_case['feasibility_score'] * 0.2 +
                    business_value_score * 0.2 +
                    innovation_score * 0.2 +
                    complexity_score * 0.1
                )
                
                scored_case = {
                    **use_case,
                    'business_value_score': business_value_score,
                    'innovation_score': innovation_score,
                    'complexity_score': complexity_score,
                    'overall_score': overall_score
                }
                
                scored_cases.append(scored_case)
            
            # Sort by overall score
            scored_cases.sort(key=lambda x: x['overall_score'], reverse=True)
            
            return {
                'use_cases': scored_cases,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error scoring use cases: {str(e)}")
            return await self._get_fallback_result(industry_research)

    async def _calculate_business_value_score(self, use_case: Dict, industry_research: Dict) -> float:
        """
        Calculate business value score.
        """
        try:
            # Generate embeddings
            value_embedding = self.sentence_model.encode(use_case['business_value'])
            industry_embedding = self.sentence_model.encode(industry_research['industry'])
            
            # Calculate similarity
            similarity = util.pytorch_cos_sim(value_embedding, industry_embedding)[0][0].item()
            
            return similarity
        except Exception as e:
            logger.error(f"Error calculating business value score: {str(e)}")
            return 0.0

    async def _calculate_innovation_score(self, use_case: Dict) -> float:
        """
        Calculate innovation score.
        """
        try:
            # Generate embeddings
            innovation_embedding = self.sentence_model.encode(use_case['innovation_potential'])
            
            # Compare with common innovation patterns
            innovation_patterns = [
                "novel approach",
                "unique solution",
                "breakthrough technology",
                "disruptive innovation",
                "game-changing"
            ]
            
            pattern_embeddings = self.sentence_model.encode(innovation_patterns)
            similarities = util.pytorch_cos_sim(innovation_embedding, pattern_embeddings)[0]
            
            return float(torch.max(similarities))
        except Exception as e:
            logger.error(f"Error calculating innovation score: {str(e)}")
            return 0.0

    async def _calculate_complexity_score(self, use_case: Dict) -> float:
        """
        Calculate technical complexity score using Ollama.
        """
        try:
            # Prepare prompt
            prompt = f"""
            Analyze the technical complexity of this use case and respond with just a number between 0 and 1:
            
            Title: {use_case['title']}
            Technical Approach: {use_case['technical_approach']}
            
            Consider:
            1. Technical requirements
            2. Implementation effort
            3. Resource needs
            4. Integration challenges
            
            Respond with just the numerical score.
            """
            
            # Call Ollama
            score_text = await self._call_ollama(prompt)
            
            # Extract score from response
            score_match = re.search(r"(\d+\.?\d*)", score_text)
            if score_match:
                return float(score_match.group(1))
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating complexity score: {str(e)}")
            return 0.0

    async def _get_from_cache(self, key: str) -> Optional[Dict]:
        """
        Get result from cache.
        """
        try:
            cached = r.get(key)
            if cached:
                return json.loads(cached)
            return None
        except Exception as e:
            logger.error(f"Error getting from cache: {str(e)}")
            return None

    async def _cache_result(self, key: str, result: Dict) -> None:
        """
        Cache result.
        """
        try:
            r.setex(key, self.cache_ttl, json.dumps(result))
        except Exception as e:
            logger.error(f"Error caching result: {str(e)}")

    async def _get_fallback_result(self, industry_research: Dict) -> Dict:
        """
        Get fallback result when generation fails.
        """
        return {
            'use_cases': [],
            'timestamp': datetime.now().isoformat()
        }

# Create a singleton instance
usecase_agent = UsecaseAgent() 