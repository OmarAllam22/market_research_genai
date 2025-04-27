import asyncio
import json
from agents.resource_agent import ResourceAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_resource_agent():
    """
    Test the resource agent with different scenarios.
    """
    test_cases = [
        {
            "name": "AI Model Resources",
            "use_case": {
                "use_case": "Natural Language Processing",
                "description": "Building a text classification model",
                "requirements": ["Pre-trained models", "Training datasets", "Evaluation metrics"]
            }
        },
        {
            "name": "Dataset Resources",
            "use_case": {
                "use_case": "Computer Vision",
                "description": "Image classification project",
                "requirements": ["Image datasets", "Annotation tools", "Data augmentation"]
            }
        },
        {
            "name": "Research Paper Resources",
            "use_case": {
                "use_case": "Reinforcement Learning",
                "description": "Implementing a new RL algorithm",
                "requirements": ["Research papers", "Code implementations", "Benchmark datasets"]
            }
        }
    ]

    resource_agent = ResourceAgent()

    for test_case in test_cases:
        logger.info(f"\nTesting Resource Agent with: {test_case['name']}")
        try:
            # Run resource collection
            result = resource_agent.run([test_case['use_case']])
            
            # Print results
            logger.info(f"\nResults for {test_case['name']}:")
            
            # Print resources
            logger.info("\nResources:")
            for resource in result[0].get('resources', []):
                logger.info(f"\nName: {resource.get('name', 'Unknown')}")
                logger.info(f"URL: {resource.get('url', 'No URL')}")
                logger.info(f"Description: {resource.get('description', 'No description')}")
                logger.info(f"Source: {resource.get('source', 'Unknown')}")
                logger.info(f"Type: {resource.get('type', 'Unknown')}")
                logger.info(f"Quality Score: {resource.get('quality_score', 0.0)}")
            
            # Save results to file
            with open(f"resource_results_{test_case['name'].lower().replace(' ', '_')}.json", 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"\nResults saved to resource_results_{test_case['name'].lower().replace(' ', '_')}.json")
            
        except Exception as e:
            logger.error(f"Error testing {test_case['name']}: {str(e)}")

async def main():
    """
    Main function to run all tests.
    """
    logger.info("Starting Resource Agent Tests")
    await test_resource_agent()
    logger.info("Resource Agent Tests Completed")

if __name__ == "__main__":
    asyncio.run(main()) 