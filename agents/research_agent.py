import os
import json
import logging
import random
import requests
import redis
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv
import wikipedia
from sentence_transformers import SentenceTransformer, util
import asyncio
from fake_useragent import UserAgent
import re
from urllib.parse import urlparse
import ollama
#from ollama import Client
import aiohttp
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Initialize Redis with connection pool for better performance
redis_pool = redis.ConnectionPool.from_url(REDIS_URL)
r = redis.Redis(connection_pool=redis_pool)

# Initialize Ollama client
#ollama_client = Client(host='http://localhost:11434')

# Industry-specific sources
INDUSTRY_SOURCES = {
    'mckinsey': 'https://www.mckinsey.com/capabilities/quantumblack/our-insights',
    'deloitte': 'https://www2.deloitte.com/insights/us/en/topics/digital-transformation.html',
    'nexocode': 'https://nexocode.com/blog/',
    'gartner': 'https://www.gartner.com/en/insights',
    'forrester': 'https://www.forrester.com/insights/'
}

class ResearchAgent:
    def __init__(self):
        self.ua = UserAgent()
        self.cache_ttl = 86400  # 24 hours
        self.max_retries = 3
        self.timeout = 30
        self._initialize_models()
        self.session = None

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

    async def run(self, company_or_industry_name: str) -> Dict:
        """
        Main entry point for the research agent.
        """
        cache_key = f"research:{company_or_industry_name.lower()}"
        
        # Try to get from cache first
        cached_result = await self._get_from_cache(cache_key)
        if cached_result:
            logger.info(f"Retrieved cached research for {company_or_industry_name}")
            return cached_result

        try:
            await self._create_session()
            
            # Perform deep web search and analysis
            search_results = await self._perform_deep_search(company_or_industry_name)
            if not search_results:
                raise Exception("No search results found")

            # Analyze the search results
            analysis = await self._analyze_search_results(search_results, company_or_industry_name)
            
            # Extract key information
            industry_info = await self._extract_industry_info(analysis)
            
            # Perform competitor analysis
            competitors = await self._analyze_competitors(company_or_industry_name, industry_info)
            industry_info['competitors'] = competitors
            
            # Perform market trend analysis
            trends = await self._analyze_market_trends(company_or_industry_name, industry_info)
            industry_info['market_trends'] = trends
            
            # Cache the results
            await self._cache_result(cache_key, industry_info)
            
            return industry_info

        except Exception as e:
            logger.error(f"Error in research agent: {str(e)}")
            return await self._get_fallback_result(company_or_industry_name)
        finally:
            await self._close_session()

    async def _perform_deep_search(self, query: str) -> List[Dict]:
        """
        Perform deep web search using multiple sources.
        """
        results = []
        
        # Try Wikipedia for detailed information
        try:
            wiki_results = await self._search_wikipedia(query)
            if wiki_results:
                results.extend(wiki_results)
        except Exception as e:
            logger.warning(f"Wikipedia search failed: {str(e)}")

        # Try industry-specific sources
        try:
            industry_results = await self._search_industry_sources(query)
            if industry_results:
                results.extend(industry_results)
        except Exception as e:
            logger.warning(f"Industry source search failed: {str(e)}")

        # Try AI-specific sources
        try:
            ai_results = await self._search_ai_sources(query)
            if ai_results:
                results.extend(ai_results)
        except Exception as e:
            logger.warning(f"AI source search failed: {str(e)}")

        return results

    async def _search_wikipedia(self, query: str) -> List[Dict]:
        """
        Search Wikipedia for detailed information.
        """
        try:
            # Search for pages
            search_results = wikipedia.search(query, results=5)
            
            if not search_results:
                return []

            results = []
            for idx, title in enumerate(search_results):
                try:
                    # Get page content
                    page = wikipedia.page(title, auto_suggest=False)
                    
                    results.append({
                        'source': 'Wikipedia',
                        'title': page.title,
                        'content': page.content,
                        'url': page.url,
                        'position': idx
                    })
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation pages
                    continue
                except wikipedia.exceptions.PageError as e:
                    # Handle page not found
                    continue
                except Exception as e:
                    logger.warning(f"Error processing Wikipedia page {title}: {str(e)}")
                    continue

            return results

        except Exception as e:
            logger.error(f"Wikipedia search error: {str(e)}")
            return []

    async def _search_industry_sources(self, query: str) -> List[Dict]:
        """
        Search industry-specific sources.
        """
        results = []
        
        for source_name, source_url in INDUSTRY_SOURCES.items():
            try:
                # Prepare search query
                search_query = f"site:{source_url} {query}"
                
                # Search using requests
                headers = {'User-Agent': self.ua.random}
                response = requests.get(f"https://www.google.com/search?q={search_query}", headers=headers)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    search_results = soup.find_all('div', class_='g')
                    
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
                
                # Add delay to avoid rate limiting
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.warning(f"{source_name} search failed: {str(e)}")
                continue

        return results

    async def _search_ai_sources(self, query: str) -> List[Dict]:
        """
        Search AI-specific sources.
        """
        results = []
        
        # AI-specific sources
        ai_sources = {
            'huggingface': 'https://huggingface.co/blog',
            'arxiv': 'https://arxiv.org/search/',
            'paperswithcode': 'https://paperswithcode.com/',
            'aihub': 'https://aihub.org/',
            'deepmind': 'https://deepmind.com/blog'
        }
        
        for source_name, source_url in ai_sources.items():
            try:
                # Prepare search query
                search_query = f"site:{source_url} {query}"
                
                # Search using requests
                headers = {'User-Agent': self.ua.random}
                response = requests.get(f"https://www.google.com/search?q={search_query}", headers=headers)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    search_results = soup.find_all('div', class_='g')
                    
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
                
                # Add delay to avoid rate limiting
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.warning(f"{source_name} search failed: {str(e)}")
                continue

        return results

    async def _analyze_search_results(self, results: List[Dict], query: str) -> Dict:
        """
        Analyze search results using Ollama.
        """
        # Combine all search results
        combined_text = "\n\n".join([
            f"Source: {r['source']}\nTitle: {r['title']}\nContent: {r['content']}"
            for r in results
        ])

        # Generate embeddings for semantic search
        embeddings = self.sentence_model.encode(combined_text)
        
        # Use Ollama for analysis
        analysis = await self._analyze_with_ollama(combined_text, query)
        
        return {
            'text': combined_text,
            'embeddings': embeddings.tolist(),
            'analysis': analysis
        }

    async def _analyze_with_ollama(self, text: str, query: str) -> Dict:
        """
        Analyze text using Ollama.
        """
        try:
            # Prepare prompt
            prompt = f"""
            Analyze the following text about {query} and provide a concise analysis in JSON format:
            
            {text}
            
            Include only these fields:
            - industry_overview: Brief overview
            - key_players: List of top 3 players
            - market_trends: List of top 3 trends
            - challenges: List of top 3 challenges
            - opportunities: List of top 3 opportunities
            - future_outlook: Brief outlook
            - ai_applications: List of top 3 AI applications
            - digital_initiatives: List of top 3 initiatives
            - industry_insights: List of top 3 insights
            """
            
            # Call Ollama
            analysis = await self._call_ollama(prompt)
            
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
            logger.error(f"Ollama analysis error: {str(e)}")
            return {
                'analysis': "Analysis failed",
                'timestamp': datetime.now().isoformat()
            }

    async def _extract_industry_info(self, analysis: Dict) -> Dict:
        """
        Extract industry information using Ollama.
        """
        try:
            # If analysis is already in JSON format, use it directly
            if isinstance(analysis['analysis'], dict):
                industry_info = {
                    'industry': analysis['analysis'].get('industry_overview', 'Unknown'),
                    'segment': 'Unknown',
                    'key_offerings': analysis['analysis'].get('ai_applications', []),
                    'strategic_focus': analysis['analysis'].get('digital_initiatives', []),
                    'vision': analysis['analysis'].get('future_outlook', 'Unknown'),
                    'products': [],
                    'timestamp': datetime.now().isoformat()
                }
                return industry_info

            # Prepare prompt
            prompt = f"""
            Extract key industry information from this analysis in JSON format:
            
            {analysis['analysis']}
            
            Include only these fields:
            - industry: Industry name
            - segment: Industry segment
            - key_offerings: List of key offerings
            - strategic_focus: List of strategic focus areas
            - vision: Vision statement
            - products: List of products/services
            """
            
            # Call Ollama
            info_text = await self._call_ollama(prompt)
            
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
        Analyze competitors using Ollama.
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
            """
            
            # Call Ollama
            competitors_text = await self._call_ollama(prompt)
            
            # Parse JSON response
            try:
                competitors = json.loads(competitors_text)
                return competitors[:3]  # Return top 3 competitors
            except json.JSONDecodeError:
                raise Exception("Failed to parse competitors JSON")
        except Exception as e:
            logger.error(f"Error analyzing competitors: {str(e)}")
            return []

    async def _analyze_market_trends(self, query: str, industry_info: Dict) -> List[Dict]:
        """
        Analyze market trends using Ollama.
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
            """
            
            # Call Ollama
            trends_text = await self._call_ollama(prompt)
            
            # Parse JSON response
            try:
                trends = json.loads(trends_text)
                return trends[:3]  # Return top 3 trends
            except json.JSONDecodeError:
                raise Exception("Failed to parse trends JSON")
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

    async def _get_fallback_result(self, query: str = "") -> Dict:
        """
        Get fallback result when search fails.
        """
        return {
            'industry': query,
            'segment': 'Unknown',
            'key_offerings': [],
            'strategic_focus': [],
            'vision': 'Unknown',
            'products': [],
            'competitors': [],
            'market_trends': [],
            'timestamp': datetime.now().isoformat()
        }

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
            'timestamp': datetime.now().isoformat()
        }

research_agent = ResearchAgent() 