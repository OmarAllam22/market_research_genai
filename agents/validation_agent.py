import os
import json
import logging
import redis
import requests
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from slack_sdk.webhook import WebhookClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')
r = redis.Redis.from_url(REDIS_URL)

class ValidationAgent:
    def __init__(self):
        self.cache_ttl = 86400  # 24 hours
        self.max_retries = 3
        self.timeout = 30
        self.webhook = WebhookClient(SLACK_WEBHOOK_URL) if SLACK_WEBHOOK_URL else None
        self.industry_rules = self._load_industry_rules()

    def _load_industry_rules(self) -> Dict:
        """
        Load industry-specific validation rules.
        """
        return {
            'Software': {
                'min_creativity_score': 0.7,
                'min_feasibility_score': 0.6,
                'min_impact_score': 0.7,
                'min_innovation_score': 0.8,
                'required_technologies': ['AI', 'ML', 'Cloud']
            },
            'Manufacturing': {
                'min_creativity_score': 0.6,
                'min_feasibility_score': 0.7,
                'min_impact_score': 0.8,
                'min_innovation_score': 0.6,
                'required_technologies': ['IoT', 'Robotics', 'AI']
            },
            'Finance': {
                'min_creativity_score': 0.7,
                'min_feasibility_score': 0.8,
                'min_impact_score': 0.8,
                'min_innovation_score': 0.7,
                'required_technologies': ['AI', 'ML', 'Blockchain']
            },
            'Healthcare': {
                'min_creativity_score': 0.8,
                'min_feasibility_score': 0.7,
                'min_impact_score': 0.9,
                'min_innovation_score': 0.7,
                'required_technologies': ['AI', 'ML', 'IoT']
            },
            'Retail': {
                'min_creativity_score': 0.7,
                'min_feasibility_score': 0.7,
                'min_impact_score': 0.8,
                'min_innovation_score': 0.7,
                'required_technologies': ['AI', 'ML', 'Cloud']
            }
        }

    def run(self, usecases: List[Dict], resources: List[Dict], user_name: str = 'User') -> List[Dict]:
        """
        Validate and score use cases.
        """
        validated_usecases = []
        
        for uc in usecases:
            try:
                # Score use case
                scores = self._score_use_case(uc, resources)
                
                # Validate use case
                validation_result = self._validate_use_case(uc, scores)
                
                # Add validation results
                uc.update({
                    'scores': scores,
                    'validation': validation_result,
                    'timestamp': datetime.now().isoformat()
                })
                
                validated_usecases.append(uc)
                
            except Exception as e:
                logger.error(f"Error validating use case {uc.get('use_case')}: {str(e)}")
                uc.update({
                    'scores': self._get_fallback_scores(),
                    'validation': self._get_fallback_validation(),
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                })
                validated_usecases.append(uc)

        # Send notification
        self._notify_slack(user_name, validated_usecases)
        
        return validated_usecases

    def _score_use_case(self, usecase: Dict, resources: List[Dict]) -> Dict:
        """
        Score use case based on various criteria.
        """
        scores = {
            'creativity_score': self._calculate_creativity_score(usecase),
            'feasibility_score': self._calculate_feasibility_score(usecase),
            'impact_score': self._calculate_impact_score(usecase),
            'innovation_score': self._calculate_innovation_score(usecase),
            'resource_score': self._calculate_resource_score(usecase, resources)
        }
        
        # Calculate overall score
        scores['overall_score'] = sum(scores.values()) / len(scores)
        
        return scores

    def _calculate_creativity_score(self, usecase: Dict) -> float:
        """
        Calculate creativity score based on novelty and uniqueness.
        """
        score = 0.0
        
        # Check description length and quality
        if usecase.get('description'):
            score += 0.3
        
        # Check novelty
        if usecase.get('novelty'):
            score += 0.3
        
        # Check uniqueness
        if usecase.get('uniqueness'):
            score += 0.2
        
        # Check innovation level
        if usecase.get('innovation_level'):
            score += 0.2
        
        return min(1.0, score)

    def _calculate_feasibility_score(self, usecase: Dict) -> float:
        """
        Calculate feasibility score based on implementation challenges.
        """
        score = 0.0
        
        # Check implementation challenges
        if usecase.get('challenges'):
            score += 0.3
        
        # Check required technologies
        if usecase.get('technologies'):
            score += 0.3
        
        # Check resource availability
        if usecase.get('resources'):
            score += 0.2
        
        # Check timeline
        if usecase.get('timeline'):
            score += 0.2
        
        return min(1.0, score)

    def _calculate_impact_score(self, usecase: Dict) -> float:
        """
        Calculate impact score based on potential benefits.
        """
        score = 0.0
        
        # Check business impact
        if usecase.get('business_impact'):
            score += 0.3
        
        # Check customer impact
        if usecase.get('customer_impact'):
            score += 0.3
        
        # Check operational impact
        if usecase.get('operational_impact'):
            score += 0.2
        
        # Check strategic alignment
        if usecase.get('strategic_alignment'):
            score += 0.2
        
        return min(1.0, score)

    def _calculate_innovation_score(self, usecase: Dict) -> float:
        """
        Calculate innovation score based on novelty and potential.
        """
        score = 0.0
        
        # Check novelty
        if usecase.get('novelty'):
            score += 0.3
        
        # Check potential
        if usecase.get('potential'):
            score += 0.3
        
        # Check uniqueness
        if usecase.get('uniqueness'):
            score += 0.2
        
        # Check market readiness
        if usecase.get('market_readiness'):
            score += 0.2
        
        return min(1.0, score)

    def _calculate_resource_score(self, usecase: Dict, resources: List[Dict]) -> float:
        """
        Calculate resource score based on available resources.
        """
        score = 0.0
        
        # Check resource availability
        if resources:
            score += 0.3
        
        # Check resource quality
        if any(r.get('quality_score', 0) > 0.7 for r in resources):
            score += 0.3
        
        # Check resource diversity
        if len(set(r.get('source') for r in resources)) > 1:
            score += 0.2
        
        # Check resource relevance
        if any(r.get('relevance_score', 0) > 0.7 for r in resources):
            score += 0.2
        
        return min(1.0, score)

    def _validate_use_case(self, usecase: Dict, scores: Dict) -> Dict:
        """
        Validate use case based on industry rules and scores.
        """
        industry = usecase.get('industry', 'Unknown')
        rules = self.industry_rules.get(industry, self.industry_rules['Software'])
        
        validation = {
            'is_valid': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check minimum scores
        if scores['creativity_score'] < rules['min_creativity_score']:
            validation['is_valid'] = False
            validation['issues'].append('Creativity score below threshold')
            validation['recommendations'].append('Enhance novelty and uniqueness')
        
        if scores['feasibility_score'] < rules['min_feasibility_score']:
            validation['is_valid'] = False
            validation['issues'].append('Feasibility score below threshold')
            validation['recommendations'].append('Address implementation challenges')
        
        if scores['impact_score'] < rules['min_impact_score']:
            validation['is_valid'] = False
            validation['issues'].append('Impact score below threshold')
            validation['recommendations'].append('Strengthen business and customer impact')
        
        if scores['innovation_score'] < rules['min_innovation_score']:
            validation['is_valid'] = False
            validation['issues'].append('Innovation score below threshold')
            validation['recommendations'].append('Increase innovation level')
        
        # Check required technologies
        if usecase.get('technologies'):
            missing_tech = set(rules['required_technologies']) - set(usecase['technologies'])
            if missing_tech:
                validation['is_valid'] = False
                validation['issues'].append(f'Missing required technologies: {", ".join(missing_tech)}')
                validation['recommendations'].append(f'Consider adding: {", ".join(missing_tech)}')
        
        return validation

    def _notify_slack(self, user_name: str, usecases: List[Dict]) -> None:
        """
        Send notification to Slack.
        """
        if not self.webhook:
            return

        try:
            text = f"*{user_name}*'s GenAI Use Cases are ready!\n\n"
            
            for uc in usecases:
                scores = uc.get('scores', {})
                validation = uc.get('validation', {})
                
                text += f"• *{uc.get('use_case', '')}*\n"
                text += f"  - Overall Score: {scores.get('overall_score', 0):.2f}\n"
                text += f"  - Status: {'✅ Valid' if validation.get('is_valid') else '❌ Invalid'}\n"
                
                if validation.get('issues'):
                    text += f"  - Issues: {', '.join(validation['issues'])}\n"
                
                if validation.get('recommendations'):
                    text += f"  - Recommendations: {', '.join(validation['recommendations'])}\n"
                
                text += "\n"
            
            self.webhook.send(text=text)
            
        except Exception as e:
            logger.error(f"Slack notification failed: {str(e)}")

    def _get_fallback_scores(self) -> Dict:
        """
        Get fallback scores when primary methods fail.
        """
        return {
            'creativity_score': 0.0,
            'feasibility_score': 0.0,
            'impact_score': 0.0,
            'innovation_score': 0.0,
            'resource_score': 0.0,
            'overall_score': 0.0
        }

    def _get_fallback_validation(self) -> Dict:
        """
        Get fallback validation when primary methods fail.
        """
        return {
            'is_valid': False,
            'issues': ['Validation failed'],
            'recommendations': ['Review use case']
        }

validation_agent = ValidationAgent() 