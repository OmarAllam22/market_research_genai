import os
import json
import logging
import redis
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv
import wikipedia
import asyncio
from fake_useragent import UserAgent
import re
from urllib.parse import urlparse
import aiohttp
import requests
from bs4 import BeautifulSoup
import base64
from PIL import Image
import io
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import random
from collections import deque
import time
from functools import wraps

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research_agent.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Get Gemini API keys from environment variables
GEMINI_API_KEYS = [
    os.getenv('GEMINI_API_KEY_1'),
    os.getenv('GEMINI_API_KEY_2'),
    os.getenv('GEMINI_API_KEY_3')
]

# Validate API keys
if not all(GEMINI_API_KEYS):
    raise ValueError("All three Gemini API keys must be set in environment variables")

# Initialize Redis with connection pool for better performance
redis_pool = redis.ConnectionPool.from_url(REDIS_URL)
r = redis.Redis(connection_pool=redis_pool)

# Industry-specific sources
INDUSTRY_SOURCES = {
    'mckinsey': 'https://www.mckinsey.com/capabilities/quantumblack/our-insights',
    'deloitte': 'https://www2.deloitte.com/insights/us/en/topics/digital-transformation.html',
    'nexocode': 'https://nexocode.com/blog/',
    'gartner': 'https://www.gartner.com/en/insights',
    'forrester': 'https://www.forrester.com/insights/'
}

# AI-specific sources
AI_SOURCES = {
    'huggingface': 'https://huggingface.co/blog',
    'arxiv': 'https://arxiv.org/search/',
    'paperswithcode': 'https://paperswithcode.com/',
    'aihub': 'https://aihub.org/',
    'deepmind': 'https://deepmind.com/blog'
}

# Performance monitoring decorator
def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in {func.__name__} after {execution_time:.2f} seconds: {str(e)}")
            raise
    return wrapper

class GeminiLoadBalancer:
    def __init__(self, api_keys: List[str]):
        self.api_keys = deque(api_keys)
        self.current_key = self.api_keys[0]
        self.usage_count = 0
        self.max_usage = 10  # Maximum requests per key before rotation
        self.failed_keys = set()  # Track failed keys
        self.key_errors = {}  # Track error counts per key
        self.max_errors = 3  # Maximum errors before marking key as failed

    def get_next_key(self) -> str:
        """
        Get the next API key using round-robin load balancing with error handling.
        """
        self.usage_count += 1
        
        # If current key has reached max usage or is in failed keys, rotate
        if self.usage_count >= self.max_usage or self.current_key in self.failed_keys:
            self.usage_count = 0
            self._rotate_keys()
            
        return self.current_key

    def _rotate_keys(self):
        """
        Rotate to the next available key.
        """
        original_key = self.current_key
        while True:
            self.api_keys.rotate(1)
            self.current_key = self.api_keys[0]
            
            # If we've checked all keys and none are available, reset failed keys
            if self.current_key == original_key:
                logger.warning("All keys have failed, resetting failed keys")
                self.failed_keys.clear()
                self.key_errors.clear()
                break
                
            # If current key is not failed, use it
            if self.current_key not in self.failed_keys:
                break

    def mark_key_error(self, key: str, error: Exception):
        """
        Track errors for a key and mark it as failed if too many errors.
        """
        if key not in self.key_errors:
            self.key_errors[key] = 0
            
        self.key_errors[key] += 1
        
        if self.key_errors[key] >= self.max_errors:
            logger.error(f"Key {key} has failed {self.max_errors} times, marking as failed")
            self.failed_keys.add(key)
            
        # If current key is marked as failed, rotate to next key
        if self.current_key in self.failed_keys:
            self._rotate_keys()

    def get_available_keys_count(self) -> int:
        """
        Get the number of available keys.
        """
        return len(self.api_keys) - len(self.failed_keys)

    def get_key_stats(self) -> Dict:
        """
        Get statistics about key usage and errors.
        """
        return {
            'total_keys': len(self.api_keys),
            'available_keys': self.get_available_keys_count(),
            'failed_keys': len(self.failed_keys),
            'key_errors': self.key_errors,
            'current_usage': self.usage_count
        }

class ResearchAgent:
    def __init__(self):
        self.ua = UserAgent()
        self.cache_ttl = 86400  # 24 hours
        self.max_retries = 3
        self.timeout = 30
        self.session = None
        self.vision_threshold = 0.7  # Threshold for image analysis confidence
        self.load_balancer = GeminiLoadBalancer(GEMINI_API_KEYS)
        self.performance_metrics = {
            'total_searches': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'average_search_time': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    async def _create_session(self):
        """
        Create an aiohttp session for API calls.
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def _close_session(self):
        """
        Close the aiohttp session.
        """
        if self.session:
            await self.session.close()
            self.session = None

    def _get_gemini_client(self) -> ChatGoogleGenerativeAI:
        """
        Get a Gemini client with the next available API key.
        """
        api_key = self.load_balancer.get_next_key()
        if not api_key:
            raise ValueError("No valid Gemini API key available")
            
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_key,
                temperature=0.7,
                max_output_tokens=2048,
                timeout=30  # Add timeout
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            self.load_balancer.mark_key_error(api_key, e)
            raise

    def _get_gemini_vision_client(self) -> ChatGoogleGenerativeAI:
        """
        Get a Gemini Vision client with the next available API key.
        """
        api_key = self.load_balancer.get_next_key()
        if not api_key:
            raise ValueError("No valid Gemini API key available")
            
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=api_key,
                temperature=0.7,
                max_output_tokens=2048,
                timeout=30  # Add timeout
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Vision client: {str(e)}")
            self.load_balancer.mark_key_error(api_key, e)
            raise

    async def _call_gemini(self, prompt: str, images: List[str] = None) -> str:
        """
        Call Gemini API with load balancing and error handling.
        """
        max_retries = 3
        retry_delay = 2
        last_error = None
        
        for attempt in range(max_retries):
            try:
                api_key = self.load_balancer.get_next_key()
                if not api_key:
                    raise ValueError("No valid Gemini API key available")
                
                if images:
                    client = self._get_gemini_vision_client()
                    messages = [
                        SystemMessage(content="You are an expert industry analyst. Analyze the provided images and text to extract relevant insights."),
                        HumanMessage(content=prompt)
                    ]
                    for image_data in images:
                        messages.append(HumanMessage(content=image_data))
                else:
                    client = self._get_gemini_client()
                    messages = [
                        SystemMessage(content="You are an expert industry analyst. Analyze the provided text to extract relevant insights."),
                        HumanMessage(content=prompt)
                    ]

                response = await client.agenerate([messages])
                if not response or not response.generations:
                    raise ValueError("Empty response from Gemini API")
                    
                return response.generations[0][0].text
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                
                # Mark the key as failed if it's a rate limit or authentication error
                if isinstance(e, (ValueError, ConnectionError)) or "rate limit" in str(e).lower():
                    self.load_balancer.mark_key_error(api_key, e)
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All {max_retries} attempts failed")
                    raise last_error

    async def _extract_images_from_url(self, url: str) -> List[str]:
        """
        Extract images from a URL and convert them to base64.
        """
        try:
            headers = {'User-Agent': self.ua.random}
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            images = []
            
            for img in soup.find_all('img'):
                img_url = img.get('src')
                if not img_url:
                    continue
                
                # Handle relative URLs
                if not img_url.startswith(('http://', 'https://')):
                    img_url = urlparse(url).join(img_url).geturl()
                
                try:
                    img_response = requests.get(img_url, headers=headers)
                    if img_response.status_code == 200:
                        # Convert image to base64
                        img_data = base64.b64encode(img_response.content).decode('utf-8')
                        images.append(img_data)
                except Exception as e:
                    logger.warning(f"Error downloading image {img_url}: {str(e)}")
                    continue

            return images
        except Exception as e:
            logger.error(f"Error extracting images from {url}: {str(e)}")
            return []

    async def _analyze_image(self, image_data: str, query: str) -> Dict:
        """
        Analyze an image using Gemini Vision.
        """
        try:
            prompt = f"""
            Analyze this image in the context of {query} and provide insights in JSON format:
            
            Include these fields:
            - relevance_score: How relevant is this image to the query (0-1)
            - content_description: What's in the image
            - industry_insights: Any industry-specific insights
            - visual_elements: Key visual elements
            - confidence_score: How confident is the analysis (0-1)
            """
            
            response = await self._call_gemini(prompt, [image_data])
            
            try:
                analysis = json.loads(response)
                return analysis
            except json.JSONDecodeError:
                return {
                    'relevance_score': 0.0,
                    'content_description': response,
                    'industry_insights': [],
                    'visual_elements': [],
                    'confidence_score': 0.0
                }
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return None

    async def _should_analyze_image(self, image_analysis: Dict) -> bool:
        """
        Determine if an image should be included in the analysis based on relevance and confidence.
        """
        if not image_analysis:
            return False
        
        relevance_score = image_analysis.get('relevance_score', 0.0)
        confidence_score = image_analysis.get('confidence_score', 0.0)
        
        # Only include images that are both relevant and have high confidence
        return relevance_score > self.vision_threshold and confidence_score > self.vision_threshold

    @monitor_performance
    async def _perform_deep_search(self, query: str) -> List[Dict]:
        """
        Perform deep web search using multiple sources in parallel with fallback.
        """
        search_sources = {
            'Wikipedia': self._search_wikipedia,
            'Industry Sources': self._search_industry_sources,
            'AI Sources': self._search_ai_sources
        }
        
        # Create tasks for parallel execution
        tasks = []
        for source_name, search_func in search_sources.items():
            tasks.append(self._search_with_timeout(source_name, search_func, query))
        
        # Execute all searches in parallel
        results = []
        search_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for source_name, result in zip(search_sources.keys(), search_results):
            if isinstance(result, Exception):
                logger.error(f"Error searching {source_name}: {str(result)}")
                continue
            if result:
                results.extend(result)
                logger.info(f"Found {len(result)} results from {source_name}")
            else:
                logger.warning(f"No results found from {source_name}")

        # If no results found from primary sources, try fallback search
        if not results:
            logger.info("No results from primary sources, trying fallback search...")
            fallback_results = await self._perform_fallback_search(query)
            if fallback_results:
                results.extend(fallback_results)
                logger.info(f"Found {len(fallback_results)} results from fallback search")

        return results

    async def _perform_fallback_search(self, query: str) -> List[Dict]:
        """
        Perform fallback search when primary methods fail.
        """
        try:
            # Try to get basic information from Wikipedia summary
            summary = await asyncio.wait_for(
                asyncio.to_thread(wikipedia.summary, query, sentences=5),
                timeout=10
            )
            
            if summary:
                return [{
                    'source': 'Wikipedia Summary',
                    'title': query,
                    'content': summary,
                    'url': f'https://en.wikipedia.org/wiki/{query.replace(" ", "_")}',
                    'position': 0
                }]
            
            return []
            
        except Exception as e:
            logger.error(f"Fallback search failed: {str(e)}")
            return []

    async def _search_with_timeout(self, source_name: str, search_func, query: str) -> List[Dict]:
        """
        Execute a search function with timeout.
        """
        try:
            return await asyncio.wait_for(search_func(query), timeout=self.timeout)
        except asyncio.TimeoutError:
            logger.error(f"Timeout while searching {source_name}")
            return []
        except Exception as e:
            logger.error(f"Error in {source_name} search: {str(e)}")
            return []

    @monitor_performance
    async def _search_industry_sources(self, query: str) -> List[Dict]:
        """
        Search industry-specific sources in parallel.
        """
        results = []
        tasks = []
        
        for source_name, source_url in INDUSTRY_SOURCES.items():
            tasks.append(self._search_single_industry_source(source_name, source_url, query))
        
        # Execute all industry source searches in parallel
        source_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in source_results:
            if isinstance(result, Exception):
                continue
            if result:
                results.extend(result)

        return results

    async def _search_single_industry_source(self, source_name: str, source_url: str, query: str) -> List[Dict]:
        """
        Search a single industry source with improved error handling.
        """
        try:
            # Prepare search query
            search_query = f"site:{source_url} {query}"
            
            # Search using requests with timeout
            headers = {'User-Agent': self.ua.random}
            response = requests.get(
                f"https://www.google.com/search?q={search_query}",
                headers=headers,
                timeout=10  # 10 second timeout
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                search_results = soup.find_all('div', class_='g')
                
                results = []
                for idx, result in enumerate(search_results[:5]):
                    title_elem = result.find('h3')
                    link_elem = result.find('a')
                    snippet_elem = result.find('div', class_='VwiC3b')
                    
                    if title_elem and link_elem and snippet_elem:
                        try:
                            # Extract images from the result URL with timeout
                            images = await asyncio.wait_for(
                                self._extract_images_from_url(link_elem['href']),
                                timeout=5  # 5 second timeout for image extraction
                            )
                            image_analyses = []
                            
                            # Analyze each image with timeout
                            for image_data in images:
                                try:
                                    image_analysis = await asyncio.wait_for(
                                        self._analyze_image(image_data, query),
                                        timeout=5  # 5 second timeout for image analysis
                                    )
                                    if await self._should_analyze_image(image_analysis):
                                        image_analyses.append(image_analysis)
                                except asyncio.TimeoutError:
                                    logger.warning(f"Timeout analyzing image for {source_name}")
                                    continue
                                except Exception as e:
                                    logger.warning(f"Error analyzing image for {source_name}: {str(e)}")
                                    continue
                            
                            results.append({
                                'source': f'{source_name.capitalize()}',
                                'title': title_elem.text,
                                'content': snippet_elem.text,
                                'url': link_elem['href'],
                                'position': len(results) + idx,
                                'image_analyses': image_analyses
                            })
                        except asyncio.TimeoutError:
                            logger.warning(f"Timeout processing result for {source_name}")
                            continue
                        except Exception as e:
                            logger.warning(f"Error processing result for {source_name}: {str(e)}")
                            continue
                
                return results
            
            return []

        except requests.exceptions.Timeout:
            logger.error(f"Timeout while searching {source_name}")
            return []
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error while searching {source_name}")
            return []
        except Exception as e:
            logger.error(f"Error searching {source_name}: {str(e)}")
            return []

    @monitor_performance
    async def _search_ai_sources(self, query: str) -> List[Dict]:
        """
        Search AI-specific sources in parallel.
        """
        results = []
        tasks = []
        
        for source_name, source_url in AI_SOURCES.items():
            tasks.append(self._search_single_ai_source(source_name, source_url, query))
        
        # Execute all AI source searches in parallel
        source_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in source_results:
            if isinstance(result, Exception):
                continue
            if result:
                results.extend(result)

        return results

    async def _search_single_ai_source(self, source_name: str, source_url: str, query: str) -> List[Dict]:
        """
        Search a single AI source with improved error handling.
        """
        try:
            # Prepare search query
            search_query = f"site:{source_url} {query}"
            
            # Search using requests with timeout
            headers = {'User-Agent': self.ua.random}
            response = requests.get(
                f"https://www.google.com/search?q={search_query}",
                headers=headers,
                timeout=10  # 10 second timeout
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                search_results = soup.find_all('div', class_='g')
                
                results = []
                for idx, result in enumerate(search_results[:5]):
                    title_elem = result.find('h3')
                    link_elem = result.find('a')
                    snippet_elem = result.find('div', class_='VwiC3b')
                    
                    if title_elem and link_elem and snippet_elem:
                        results.append({
                            'source': f'{source_name.capitalize()}',
                            'title': title_elem.text,
                            'content': snippet_elem.text,
                            'url': link_elem['href'],
                            'position': len(results) + idx
                        })
                
                return results
            
            return []

        except requests.exceptions.Timeout:
            logger.error(f"Timeout while searching {source_name}")
            return []
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error while searching {source_name}")
            return []
        except Exception as e:
            logger.error(f"Error searching {source_name}: {str(e)}")
            return []

    async def _analyze_with_gemini(self, text: str, query: str, image_analyses: List[Dict] = None) -> Dict:
        """
        Analyze text and images using Gemini.
        """
        try:
            # Prepare prompt
            prompt = f"""
            Analyze the following text about {query} and provide a concise analysis in JSON format:
            
            {text}
            
            Include only these fields:
            - industry_overview: Brief overview
            - industry_segment: Industry segment (e.g., Automotive, Manufacturing, Finance, Retail, Healthcare)
            - key_players: List of top 3 players
            - market_trends: List of top 3 trends
            - challenges: List of top 3 challenges
            - opportunities: List of top 3 opportunities
            - future_outlook: Brief outlook
            - ai_applications: List of top 3 AI applications
            - digital_initiatives: List of top 3 initiatives
            - industry_insights: List of top 3 insights
            - market_size: Market size and growth
            - target_customers: Target customer segments
            - competitive_landscape: Brief overview of competition
            """
            
            # Add image analysis if available
            if image_analyses:
                prompt += "\n\nImage Analysis:\n"
                for analysis in image_analyses:
                    prompt += f"- {analysis['content_description']}\n"
                    prompt += f"  Relevance: {analysis['relevance_score']}\n"
                    prompt += f"  Insights: {', '.join(analysis['industry_insights'])}\n"
            
            # Call Gemini
            analysis = await self._call_gemini(prompt, image_analyses)
            
            try:
                # Parse JSON response
                analysis_dict = json.loads(analysis)
                return {
                    'analysis': analysis_dict,
                    'timestamp': datetime.now().isoformat()
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw analysis
                return {
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Gemini analysis error: {str(e)}")
            return {
                'analysis': "Analysis failed",
                'timestamp': datetime.now().isoformat()
            }

    async def _analyze_search_results(self, results: List[Dict], query: str) -> Dict:
        """
        Analyze search results using Gemini.
        """
        # Combine all search results
        combined_text = "\n\n".join([
            f"Source: {r['source']}\nTitle: {r['title']}\nContent: {r['content']}"
            for r in results
        ])

        # Collect all image analyses
        image_analyses = []
        for result in results:
            if 'image_analyses' in result:
                image_analyses.extend(result['image_analyses'])

        # Use Gemini for analysis
        analysis = await self._analyze_with_gemini(combined_text, query, image_analyses)
        
        return {
            'text': combined_text,
            'analysis': analysis
        }

    async def run(self, company_or_industry_name: str) -> Dict:
        """
        Main entry point for the research agent with strict error handling and fallback.
        """
        if not company_or_industry_name:
            raise ValueError("Company or industry name cannot be empty")
            
        self.performance_metrics['total_searches'] += 1
        cache_key = f"research:{company_or_industry_name.lower()}"
        
        try:
            # Try to get from cache first
            cached_result = await self._get_from_cache(cache_key)
            if cached_result:
                logger.info(f"Cache hit for {company_or_industry_name}")
                self.performance_metrics['cache_hits'] += 1
                return cached_result
            
            self.performance_metrics['cache_misses'] += 1
            logger.info(f"Cache miss for {company_or_industry_name}, performing fresh search")

            await self._create_session()
            
            # Perform deep web search and analysis
            search_results = await self._perform_deep_search(company_or_industry_name)
            if not search_results:
                # Try fallback search
                logger.warning("No results from primary search, trying fallback...")
                search_results = await self._perform_fallback_search(company_or_industry_name)
                if not search_results:
                    raise Exception("No search results found from any source")

            # Analyze the search results
            analysis = await self._analyze_search_results(search_results, company_or_industry_name)
            if not analysis:
                raise Exception("Failed to analyze search results")
            
            # Extract key information
            industry_info = await self._extract_industry_info(analysis)
            if not industry_info:
                raise Exception("Failed to extract industry information")
            
            # Perform competitor analysis
            competitors = await self._analyze_competitors(company_or_industry_name, industry_info)
            industry_info['competitors'] = competitors
            
            # Perform market trend analysis
            trends = await self._analyze_market_trends(company_or_industry_name, industry_info)
            industry_info['market_trends'] = trends
            
            # Cache the results
            await self._cache_result(cache_key, industry_info)
            
            self.performance_metrics['successful_searches'] += 1
            return industry_info

        except Exception as e:
            self.performance_metrics['failed_searches'] += 1
            logger.error(f"Critical error in research agent: {str(e)}")
            raise  # Re-raise the exception to stop execution
        finally:
            await self._close_session()

    async def _search_wikipedia(self, query: str) -> List[Dict]:
        """
        Search Wikipedia with improved error handling.
        """
        try:
            # Search for pages with timeout
            search_results = await asyncio.wait_for(
                asyncio.to_thread(wikipedia.search, query, results=5),
                timeout=10  # 10 second timeout
            )
            
            if not search_results:
                return []

            results = []
            for idx, title in enumerate(search_results):
                try:
                    # Get page content with timeout
                    page = await asyncio.wait_for(
                        asyncio.to_thread(wikipedia.page, title, auto_suggest=False),
                        timeout=10  # 10 second timeout
                    )
                    
                    results.append({
                        'source': 'Wikipedia',
                        'title': page.title,
                        'content': page.content,
                        'url': page.url,
                        'position': idx
                    })
                except wikipedia.exceptions.DisambiguationError:
                    continue
                except wikipedia.exceptions.PageError:
                    continue
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout processing Wikipedia page {title}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing Wikipedia page {title}: {str(e)}")
                    continue

            return results

        except asyncio.TimeoutError:
            logger.error("Timeout while searching Wikipedia")
            return []
        except Exception as e:
            logger.error(f"Wikipedia search error: {str(e)}")
            return []

    async def _extract_industry_info(self, analysis: Dict) -> Dict:
        """
        Extract industry information using Gemini.
        """
        try:
            # If analysis is already in JSON format, use it directly
            if isinstance(analysis.get('analysis'), dict):
                industry_info = {
                    'industry': analysis['analysis'].get('industry_overview', 'Unknown'),
                    'segment': analysis['analysis'].get('industry_segment', 'Unknown'),
                    'key_offerings': analysis['analysis'].get('ai_applications', []),
                    'strategic_focus': analysis['analysis'].get('digital_initiatives', []),
                    'vision': analysis['analysis'].get('future_outlook', 'Unknown'),
                    'products': analysis['analysis'].get('products', []),
                    'timestamp': datetime.now().isoformat()
                }
                return industry_info

            # If analysis is a string, try to parse it as JSON first
            if isinstance(analysis.get('analysis'), str):
                try:
                    parsed_analysis = json.loads(analysis['analysis'])
                    if isinstance(parsed_analysis, dict):
                        industry_info = {
                            'industry': parsed_analysis.get('industry_overview', 'Unknown'),
                            'segment': parsed_analysis.get('industry_segment', 'Unknown'),
                            'key_offerings': parsed_analysis.get('ai_applications', []),
                            'strategic_focus': parsed_analysis.get('digital_initiatives', []),
                            'vision': parsed_analysis.get('future_outlook', 'Unknown'),
                            'products': parsed_analysis.get('products', []),
                            'timestamp': datetime.now().isoformat()
                        }
                        return industry_info
                except json.JSONDecodeError:
                    pass  # If JSON parsing fails, continue with text analysis

            # Prepare prompt for text analysis
            prompt = f"""
            Extract key industry information from this analysis in JSON format:
            
            {analysis.get('analysis', '')}
            
            Include only these fields:
            - industry: Industry name
            - segment: Industry segment (e.g., Automotive, Manufacturing, Finance, Retail, Healthcare)
            - key_offerings: List of key offerings
            - strategic_focus: List of strategic focus areas
            - vision: Vision statement
            - products: List of products/services
            - market_size: Market size and growth
            - target_customers: Target customer segments
            - competitive_landscape: Brief overview of competition
            """
            
            # Call Gemini
            info_text = await self._call_gemini(prompt)
            
            # Parse JSON response
            try:
                industry_info = json.loads(info_text)
                industry_info['timestamp'] = datetime.now().isoformat()
                return industry_info
            except json.JSONDecodeError:
                raise Exception("Failed to parse industry info JSON")
        except Exception as e:
            logger.error(f"Error extracting industry info: {str(e)}")
            return self._get_fallback_industry_info()

    async def _analyze_competitors(self, query: str, industry_info: Dict) -> List[Dict]:
        """
        Analyze competitors using Gemini with robust JSON parsing.
        """
        try:
            # Prepare prompt
            prompt = f"""
            Analyze competitors for {query} in the {industry_info['industry']} industry.
            
            Industry Info:
            {json.dumps(industry_info, indent=2)}
            
            Identify and analyze the top 3 competitors in JSON format:
            [
                {{
                    "name": "Company name",
                    "description": "Brief description",
                    "strengths": ["Key strength 1", "Key strength 2"],
                    "position": "Market position",
                    "ai_capabilities": ["AI capability 1", "AI capability 2"],
                    "digital_initiatives": ["Initiative 1", "Initiative 2"],
                    "advantages": ["Advantage 1", "Advantage 2"],
                    "relevance_score": 0.8
                }}
            ]
            
            Ensure the response is valid JSON.
            """
            
            # Call Gemini
            competitors_text = await self._call_gemini(prompt)
            
            # Try to parse JSON response
            try:
                # First try direct JSON parsing
                competitors = json.loads(competitors_text)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the text
                try:
                    # Look for JSON array pattern
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', competitors_text, re.DOTALL)
                    if json_match:
                        competitors = json.loads(json_match.group())
                    else:
                        # If no JSON array found, try to parse as a list of objects
                        competitors = []
                        for line in competitors_text.split('\n'):
                            if line.strip().startswith('{') and line.strip().endswith('}'):
                                try:
                                    competitor = json.loads(line.strip())
                                    competitors.append(competitor)
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    logger.error(f"Failed to parse competitors JSON: {str(e)}")
                    return []

            # Validate and clean up the competitors data
            valid_competitors = []
            for competitor in competitors:
                if isinstance(competitor, dict):
                    # Ensure all required fields are present
                    competitor_data = {
                        'name': competitor.get('name', 'Unknown'),
                        'description': competitor.get('description', 'No description'),
                        'strengths': competitor.get('strengths', []),
                        'position': competitor.get('position', 'Unknown'),
                        'ai_capabilities': competitor.get('ai_capabilities', []),
                        'digital_initiatives': competitor.get('digital_initiatives', []),
                        'advantages': competitor.get('advantages', []),
                        'relevance_score': float(competitor.get('relevance_score', 0.0))
                    }
                    valid_competitors.append(competitor_data)

            return valid_competitors[:3]  # Return top 3 competitors

        except Exception as e:
            logger.error(f"Error analyzing competitors: {str(e)}")
            return []

    async def _analyze_market_trends(self, query: str, industry_info: Dict) -> List[Dict]:
        """
        Analyze market trends using Gemini with robust JSON parsing.
        """
        try:
            # Prepare prompt
            prompt = f"""
            Analyze market trends for {query} in the {industry_info['industry']} industry.
            
            Industry Info:
            {json.dumps(industry_info, indent=2)}
            
            Identify and analyze the top 3 market trends in JSON format:
            [
                {{
                    "name": "Trend name",
                    "description": "Brief description",
                    "impact": 0.8,
                    "horizon": "Short/Medium/Long term",
                    "drivers": ["Driver 1", "Driver 2"],
                    "ai_implications": ["Implication 1", "Implication 2"],
                    "digital_impact": ["Impact 1", "Impact 2"],
                    "considerations": ["Consideration 1", "Consideration 2"]
                }}
            ]
            
            Ensure the response is valid JSON.
            """
            
            # Call Gemini
            trends_text = await self._call_gemini(prompt)
            
            # Try to parse JSON response
            try:
                # First try direct JSON parsing
                trends = json.loads(trends_text)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the text
                try:
                    # Look for JSON array pattern
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', trends_text, re.DOTALL)
                    if json_match:
                        trends = json.loads(json_match.group())
                    else:
                        # If no JSON array found, try to parse as a list of objects
                        trends = []
                        for line in trends_text.split('\n'):
                            if line.strip().startswith('{') and line.strip().endswith('}'):
                                try:
                                    trend = json.loads(line.strip())
                                    trends.append(trend)
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    logger.error(f"Failed to parse trends JSON: {str(e)}")
                    return []

            # Validate and clean up the trends data
            valid_trends = []
            for trend in trends:
                if isinstance(trend, dict):
                    # Ensure all required fields are present
                    trend_data = {
                        'name': trend.get('name', 'Unknown'),
                        'description': trend.get('description', 'No description'),
                        'impact': float(trend.get('impact', 0.0)),
                        'horizon': trend.get('horizon', 'Unknown'),
                        'drivers': trend.get('drivers', []),
                        'ai_implications': trend.get('ai_implications', []),
                        'digital_impact': trend.get('digital_impact', []),
                        'considerations': trend.get('considerations', [])
                    }
                    valid_trends.append(trend_data)

            return valid_trends[:3]  # Return top 3 trends

        except Exception as e:
            logger.error(f"Error analyzing market trends: {str(e)}")
            return []

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

    def _get_fallback_industry_info(self) -> Dict:
        """
        Get fallback industry info.
        """
        return {
            'industry': 'Unknown',
            'segment': 'Unknown',
            'key_offerings': [],
            'strategic_focus': [],
            'vision': 'Unknown',
            'products': [],
            'market_size': 'Unknown',
            'target_customers': [],
            'competitive_landscape': 'Unknown',
            'timestamp': datetime.now().isoformat()
        }

    def get_performance_metrics(self) -> Dict:
        """
        Get current performance metrics.
        """
        return self.performance_metrics

research_agent = ResearchAgent() 