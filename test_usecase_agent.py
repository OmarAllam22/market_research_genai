import asyncio
import json
from agents.usecase_agent import usecase_agent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_usecase_agent():
    """
    Test the usecase agent with different scenarios.
    """
    test_cases = [
        {
            "name": "Tech Industry",
            "industry_info": {
                "industry": "Technology",
                "segment": "Software Development",
                "key_offerings": ["Cloud Computing", "AI Solutions", "Enterprise Software"],
                "strategic_focus": ["Digital Transformation", "AI/ML", "Cloud Services"],
                "vision": "Empowering digital transformation through innovative technology solutions",
                "products": ["Cloud Platform", "AI Services", "Enterprise Software"]
            }
        },
        {
            "name": "Healthcare Industry",
            "industry_info": {
                "industry": "Healthcare",
                "segment": "Digital Health",
                "key_offerings": ["Telemedicine", "Health Analytics", "Patient Care"],
                "strategic_focus": ["Digital Health", "Patient Experience", "Data Analytics"],
                "vision": "Transforming healthcare through digital innovation",
                "products": ["Telehealth Platform", "Health Analytics", "Patient Portal"]
            }
        },
        {
            "name": "Manufacturing Industry",
            "industry_info": {
                "industry": "Manufacturing",
                "segment": "Smart Manufacturing",
                "key_offerings": ["IoT Solutions", "Predictive Maintenance", "Process Automation"],
                "strategic_focus": ["Industry 4.0", "Automation", "Digital Twin"],
                "vision": "Leading the future of smart manufacturing",
                "products": ["IoT Platform", "Predictive Analytics", "Automation Solutions"]
            }
        }
    ]

    for test_case in test_cases:
        logger.info(f"\nTesting Usecase Agent with: {test_case['name']}")
        try:
            # Run usecase generation
            result = await usecase_agent.run(test_case['industry_info'])
            
            # Print results
            logger.info(f"\nResults for {test_case['name']}:")
            
            # Print use cases
            logger.info("\nUse Cases:")
            for use_case in result.get('use_cases', []):
                logger.info(f"\nTitle: {use_case.get('title', 'Unknown')}")
                logger.info(f"Description: {use_case.get('description', 'No description')}")
                logger.info(f"Business Value: {use_case.get('business_value', 'No value')}")
                logger.info(f"Technical Approach: {use_case.get('technical_approach', 'No approach')}")
                logger.info(f"Innovation Potential: {use_case.get('innovation_potential', 'No potential')}")
                logger.info(f"Overall Score: {use_case.get('overall_score', 0.0)}")
            
            # Save results to file
            with open(f"usecase_results_{test_case['name'].lower().replace(' ', '_')}.json", 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"\nResults saved to usecase_results_{test_case['name'].lower().replace(' ', '_')}.json")
            
        except Exception as e:
            logger.error(f"Error testing {test_case['name']}: {str(e)}")

async def main():
    """
    Main function to run all tests.
    """
    logger.info("Starting Usecase Agent Tests")
    await test_usecase_agent()
    logger.info("Usecase Agent Tests Completed")

if __name__ == "__main__":
    asyncio.run(main()) 