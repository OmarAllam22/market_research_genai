import asyncio
import json
from agents.validation_agent import ValidationAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_validation_agent():
    """
    Test the validation agent with different scenarios.
    """
    test_cases = [
        {
            "name": "Software Industry Use Case",
            "use_case": {
                "use_case": "AI-Powered Code Review",
                "description": "Automated code review system using AI to detect bugs and suggest improvements",
                "novelty": "High",
                "uniqueness": "Medium",
                "innovation_level": "High",
                "challenges": ["Model accuracy", "Integration with existing tools"],
                "technologies": ["AI", "ML", "Cloud"],
                "resources": ["Pre-trained models", "Code datasets"],
                "timeline": "6 months",
                "business_impact": "Reduced review time by 50%",
                "customer_impact": "Improved code quality",
                "operational_impact": "Streamlined review process",
                "strategic_alignment": "Digital transformation"
            },
            "resources": [
                {
                    "name": "CodeBERT",
                    "type": "model",
                    "quality_score": 0.9
                },
                {
                    "name": "CodeSearchNet",
                    "type": "dataset",
                    "quality_score": 0.8
                }
            ]
        },
        {
            "name": "Healthcare Industry Use Case",
            "use_case": {
                "use_case": "AI-Powered Diagnosis Assistant",
                "description": "AI system to assist doctors in diagnosing diseases from medical images",
                "novelty": "High",
                "uniqueness": "High",
                "innovation_level": "High",
                "challenges": ["Regulatory compliance", "Model accuracy"],
                "technologies": ["AI", "ML", "IoT"],
                "resources": ["Medical image datasets", "Pre-trained models"],
                "timeline": "12 months",
                "business_impact": "Improved diagnosis accuracy",
                "customer_impact": "Better patient outcomes",
                "operational_impact": "Reduced diagnosis time",
                "strategic_alignment": "Healthcare innovation"
            },
            "resources": [
                {
                    "name": "MedMNIST",
                    "type": "dataset",
                    "quality_score": 0.9
                },
                {
                    "name": "CheXNet",
                    "type": "model",
                    "quality_score": 0.85
                }
            ]
        },
        {
            "name": "Manufacturing Industry Use Case",
            "use_case": {
                "use_case": "Predictive Maintenance System",
                "description": "IoT-based system for predicting equipment failures",
                "novelty": "Medium",
                "uniqueness": "Medium",
                "innovation_level": "High",
                "challenges": ["Sensor integration", "Data quality"],
                "technologies": ["IoT", "Robotics", "AI"],
                "resources": ["Sensor data", "ML models"],
                "timeline": "9 months",
                "business_impact": "Reduced downtime",
                "customer_impact": "Improved reliability",
                "operational_impact": "Optimized maintenance",
                "strategic_alignment": "Industry 4.0"
            },
            "resources": [
                {
                    "name": "Industrial IoT Dataset",
                    "type": "dataset",
                    "quality_score": 0.8
                },
                {
                    "name": "Predictive Maintenance Model",
                    "type": "model",
                    "quality_score": 0.85
                }
            ]
        }
    ]

    validation_agent = ValidationAgent()

    for test_case in test_cases:
        logger.info(f"\nTesting Validation Agent with: {test_case['name']}")
        try:
            # Run validation
            result = validation_agent.run([test_case['use_case']], test_case['resources'])
            
            # Print results
            logger.info(f"\nResults for {test_case['name']}:")
            
            # Print scores
            logger.info("\nScores:")
            for score_name, score_value in result[0].get('scores', {}).items():
                logger.info(f"{score_name}: {score_value}")
            
            # Print validation results
            logger.info("\nValidation Results:")
            for validation_name, validation_value in result[0].get('validation', {}).items():
                logger.info(f"{validation_name}: {validation_value}")
            
            # Save results to file
            with open(f"validation_results_{test_case['name'].lower().replace(' ', '_')}.json", 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"\nResults saved to validation_results_{test_case['name'].lower().replace(' ', '_')}.json")
            
        except Exception as e:
            logger.error(f"Error testing {test_case['name']}: {str(e)}")

async def main():
    """
    Main function to run all tests.
    """
    logger.info("Starting Validation Agent Tests")
    await test_validation_agent()
    logger.info("Validation Agent Tests Completed")

if __name__ == "__main__":
    asyncio.run(main()) 