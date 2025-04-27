import os
import json
import logging
import redis
import requests
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import HfApi
from kaggle.api.kaggle_api_extended import KaggleApi
from github import Github
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
r = redis.Redis.from_url(REDIS_URL)

class ResourceAgent:
    def __init__(self):
        self.cache_ttl = 86400  # 24 hours
        self.max_retries = 3
        self.timeout = 30
        self.hf_api = HfApi()
        self.kaggle_api = None
        self.github_api = None
        self._initialize_apis()
        #self._initialize_models()

    def _initialize_apis(self):
        """
        Initialize API clients.
        """
        try:
            # Initialize Kaggle API
            self.kaggle_api = KaggleApi()
            self.kaggle_api.authenticate()
            logger.info("Kaggle API initialized successfully")
        except Exception as e:
            logger.warning(f"Kaggle API initialization failed: {str(e)}")
            self.kaggle_api = None

        try:
            # Initialize GitHub API
            self.github_api = Github(os.getenv('GITHUB_TOKEN'))
            logger.info("GitHub API initialized successfully")
        except Exception as e:
            logger.warning(f"GitHub API initialization failed: {str(e)}")
            self.github_api = None

    def _initialize_models(self):
        """
        Initialize ML models for resource validation.
        """
        try:
            # Initialize sentence transformer for semantic search
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize text classification model for relevance
            self.classifier = pipeline(
                "text-classification",
                model="facebook/bart-large-mnli",
                tokenizer="facebook/bart-large-mnli"
            )
            
            logger.info("ML models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ML models: {str(e)}")
            raise

    def run(self, usecases: List[Dict]) -> List[Dict]:
        """
        Collect resource assets for each use case.
        """
        for uc in usecases:
            cache_key = f"resources:{uc['use_case']}"
            
            # Try to get from cache first
            cached_resources = self._get_from_cache(cache_key)
            if cached_resources:
                logger.info(f"Retrieved cached resources for {uc['use_case']}")
                uc['resources'] = cached_resources
                continue

            try:
                # Search for resources
                resources = self._search_resources(uc)
                
                # Validate resources
                validated_resources = self._validate_resources(resources, uc)
                
                # Cache the results
                self._cache_result(cache_key, validated_resources)
                
                uc['resources'] = validated_resources

            except Exception as e:
                logger.error(f"Error collecting resources for {uc['use_case']}: {str(e)}")
                uc['resources'] = self._get_fallback_resources()

        return usecases

    def _search_resources(self, usecase: Dict) -> List[Dict]:
        """
        Search for resources across multiple platforms.
        """
        resources = []
        
        # Search HuggingFace
        try:
            hf_resources = self._search_huggingface(usecase)
            resources.extend(hf_resources)
        except Exception as e:
            logger.warning(f"HuggingFace search failed: {str(e)}")

        # Search Kaggle
        if self.kaggle_api:
            try:
                kaggle_resources = self._search_kaggle(usecase)
                resources.extend(kaggle_resources)
            except Exception as e:
                logger.warning(f"Kaggle search failed: {str(e)}")

        # Search GitHub
        if self.github_api:
            try:
                github_resources = self._search_github(usecase)
                resources.extend(github_resources)
            except Exception as e:
                logger.warning(f"GitHub search failed: {str(e)}")

        # Search academic sources
        try:
            academic_resources = self._search_academic_sources(usecase)
            resources.extend(academic_resources)
        except Exception as e:
            logger.warning(f"Academic source search failed: {str(e)}")

        return resources

    def _search_huggingface(self, usecase: Dict) -> List[Dict]:
        """
        Search for datasets on HuggingFace.
        """
        try:
            # Search for datasets
            datasets = list(self.hf_api.list_datasets(
                search=usecase['use_case'],
                limit=10
            ))
            
            # Search for models
            models = list(self.hf_api.list_models(
                search=usecase['use_case'],
                limit=10
            ))
            
            resources = []
            
            # Process datasets
            for dataset in datasets:
                resources.append({
                    'name': dataset.id,
                    'url': f'https://huggingface.co/datasets/{dataset.id}',
                    'description': dataset.description,
                    'source': 'HuggingFace',
                    'type': 'dataset',
                    'quality_score': self._calculate_quality_score(dataset)
                })
            
            # Process models
            for model in models:
                resources.append({
                    'name': model.id,
                    'url': f'https://huggingface.co/{model.id}',
                    'description': model.description,
                    'source': 'HuggingFace',
                    'type': 'model',
                    'quality_score': self._calculate_quality_score(model)
                })
            
            return resources

        except Exception as e:
            logger.error(f"HuggingFace search error: {str(e)}")
            return []

    def _search_kaggle(self, usecase: Dict) -> List[Dict]:
        """
        Search for datasets on Kaggle.
        """
        try:
            # Search for datasets
            datasets = self.kaggle_api.dataset_list(
                search=usecase['use_case'],
                max_size=10
            )
            
            # Search for notebooks
            notebooks = self.kaggle_api.kernels_list(
                search=usecase['use_case'],
                max_size=10
            )
            
            resources = []
            
            # Process datasets
            for dataset in datasets:
                resources.append({
                    'name': dataset.title,
                    'url': f'https://kaggle.com/datasets/{dataset.ref}',
                    'description': dataset.description,
                    'source': 'Kaggle',
                    'type': 'dataset',
                    'quality_score': self._calculate_quality_score(dataset)
                })
            
            # Process notebooks
            for notebook in notebooks:
                resources.append({
                    'name': notebook.title,
                    'url': f'https://kaggle.com/kernels/{notebook.ref}',
                    'description': notebook.description,
                    'source': 'Kaggle',
                    'type': 'notebook',
                    'quality_score': self._calculate_quality_score(notebook)
                })
            
            return resources

        except Exception as e:
            logger.error(f"Kaggle search error: {str(e)}")
            return []

    def _search_github(self, usecase: Dict) -> List[Dict]:
        """
        Search for repositories on GitHub.
        """
        try:
            # Search for repositories
            repos = self.github_api.search_repositories(
                query=f"{usecase['use_case']} dataset",
                sort="stars",
                order="desc"
            )
            
            resources = []
            for repo in repos[:10]:  # Get top 10 results
                resources.append({
                    'name': repo.name,
                    'url': repo.html_url,
                    'description': repo.description,
                    'source': 'GitHub',
                    'type': 'repository',
                    'quality_score': self._calculate_quality_score(repo)
                })
            
            return resources

        except Exception as e:
            logger.error(f"GitHub search error: {str(e)}")
            return []

    def _search_academic_sources(self, usecase: Dict) -> List[Dict]:
        """
        Search academic sources for relevant papers and datasets.
        """
        try:
            # Search arXiv
            arxiv_results = self._search_arxiv(usecase)
            
            # Search Papers with Code
            pwc_results = self._search_papers_with_code(usecase)
            
            return arxiv_results + pwc_results

        except Exception as e:
            logger.error(f"Academic source search error: {str(e)}")
            return []

    def _search_arxiv(self, usecase: Dict) -> List[Dict]:
        """
        Search arXiv for relevant papers.
        """
        try:
            url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f"all:{usecase['use_case']}",
                'start': 0,
                'max_results': 10
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            if response.status_code != 200:
                return []

            # Parse XML response
            from xml.etree import ElementTree
            root = ElementTree.fromstring(response.content)
            
            resources = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                resources.append({
                    'name': entry.find('{http://www.w3.org/2005/Atom}title').text,
                    'url': entry.find('{http://www.w3.org/2005/Atom}id').text,
                    'description': entry.find('{http://www.w3.org/2005/Atom}summary').text,
                    'source': 'arXiv',
                    'type': 'paper',
                    'quality_score': self._calculate_quality_score(entry)
                })
            
            return resources

        except Exception as e:
            logger.error(f"arXiv search error: {str(e)}")
            return []

    def _search_papers_with_code(self, usecase: Dict) -> List[Dict]:
        """
        Search Papers with Code for relevant papers and implementations.
        """
        try:
            url = "https://paperswithcode.com/api/v1/papers/"
            params = {
                'q': usecase['use_case'],
                'ordering': '-github_stars',
                'page_size': 10
            }
            
            response = requests.get(url, params=params, timeout=self.timeout)
            if response.status_code != 200:
                return []

            data = response.json()
            resources = []
            
            for result in data.get('results', []):
                resources.append({
                    'name': result['title'],
                    'url': result['url'],
                    'description': result['abstract'],
                    'source': 'Papers with Code',
                    'type': 'paper',
                    'quality_score': self._calculate_quality_score(result)
                })
                
                # Add implementation if available
                if result.get('github_link'):
                    resources.append({
                        'name': f"{result['title']} Implementation",
                        'url': result['github_link'],
                        'description': f"Implementation of {result['title']}",
                        'source': 'Papers with Code',
                        'type': 'implementation',
                        'quality_score': self._calculate_quality_score(result)
                    })
            
            return resources

        except Exception as e:
            logger.error(f"Papers with Code search error: {str(e)}")
            return []

    def _validate_resources(self, resources: List[Dict], usecase: Dict) -> List[Dict]:
        """
        Validate and filter resources.
        """
        validated_resources = []
        for resource in resources:
            try:
                # Check relevance
                if not self._is_relevant_resource(resource, usecase):
                    continue
                
                # Validate resource
                if self._is_valid_resource(resource):
                    # Add metadata
                    resource['timestamp'] = datetime.now().isoformat()
                    validated_resources.append(resource)
            except Exception as e:
                logger.error(f"Error validating resource: {str(e)}")
                continue
        
        return validated_resources

    def _is_relevant_resource(self, resource: Dict, usecase: Dict) -> bool:
        """
        Check if resource is relevant to use case.
        """
        try:
            # Prepare text for classification
            text = f"{resource['name']} {resource['description']}"
            label = f"relevant to {usecase['use_case']}"
            
            # Classify relevance
            result = self.classifier(
                text,
                candidate_labels=[label, "not relevant"],
                hypothesis_template="This resource is {}"
            )
            
            return result[0]['label'] == label and result[0]['score'] > 0.7
        except Exception as e:
            logger.error(f"Error checking resource relevance: {str(e)}")
            return True

    def _is_valid_resource(self, resource: Dict) -> bool:
        """
        Validate resource based on criteria.
        """
        # Check required fields
        required_fields = ['name', 'url', 'source', 'type']
        if not all(field in resource for field in required_fields):
            return False

        # Check quality score
        if resource.get('quality_score', 0) < 0.5:
            return False

        return True

    def _calculate_quality_score(self, resource: Dict) -> float:
        """
        Calculate quality score for a resource.
        """
        score = 0.0
        
        try:
            # Check description
            if resource.get('description'):
                score += 0.3
            
            # Check popularity (if available)
            if resource.get('stars') or resource.get('downloads'):
                score += 0.3
            
            # Check recency
            if resource.get('updated_at'):
                score += 0.2
            
            # Check documentation
            if resource.get('documentation'):
                score += 0.2
            
            return min(1.0, score)
        except Exception as e:
            logger.error(f"Error calculating quality score: {str(e)}")
            return 0.0

    def _get_from_cache(self, key: str) -> Optional[List[Dict]]:
        """
        Get results from cache.
        """
        try:
            cached = r.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {str(e)}")
        return None

    def _cache_result(self, key: str, result: List[Dict]) -> None:
        """
        Cache results.
        """
        try:
            r.set(key, json.dumps(result), ex=self.cache_ttl)
        except Exception as e:
            logger.warning(f"Cache storage error: {str(e)}")

    def _get_fallback_resources(self) -> List[Dict]:
        """
        Get fallback resources when primary methods fail.
        """
        return [{
            'name': 'Fallback Resource',
            'url': '',
            'description': 'Unable to retrieve resources',
            'source': 'Unknown',
            'type': 'unknown',
            'quality_score': 0.0,
            'timestamp': datetime.now().isoformat(),
            'error': 'Primary resource collection failed'
        }]

resource_agent = ResourceAgent() 