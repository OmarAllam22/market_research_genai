import asyncio
import json
from agents.research_agent import research_agent
import logging
import time
from datetime import datetime
import sys
import os
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_previous_results():
    """
    Clean up previous test results and logs.
    """
    # Clean up result files
    result_files = glob.glob("research_results_*.json")
    for file in result_files:
        try:
            os.remove(file)
            logger.info(f"Removed previous result file: {file}")
        except Exception as e:
            logger.warning(f"Failed to remove {file}: {str(e)}")

    # Clean up test summary
    if os.path.exists("test_summary.json"):
        try:
            os.remove("test_summary.json")
            logger.info("Removed previous test summary")
        except Exception as e:
            logger.warning(f"Failed to remove test summary: {str(e)}")

    # Clean up log file
    if os.path.exists("research_agent.log"):
        try:
            os.remove("research_agent.log")
            logger.info("Removed previous log file")
        except Exception as e:
            logger.warning(f"Failed to remove log file: {str(e)}")

async def test_research_agent():
    """
    Test the research agent with different scenarios.
    """
    # Clean up previous results
    cleanup_previous_results()

    test_cases = [
        {
            "name": "Tech Company",
            "query": "Microsoft"
        },
        {
            "name": "Manufacturing Industry",
            "query": "Automotive Manufacturing"
        },
        {
            "name": "Healthcare Industry",
            "query": "Digital Health"
        }
    ]

    start_time = time.time()
    total_results = []
    critical_errors = []

    for test_case in test_cases:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing Research Agent with: {test_case['name']}")
        logger.info(f"{'='*50}")
        
        try:
            # Run research
            result = await research_agent.run(test_case['query'])
            
            # Print results
            logger.info(f"\nResults for {test_case['name']}:")
            logger.info(f"Industry: {result.get('industry', 'Unknown')}")
            logger.info(f"Segment: {result.get('segment', 'Unknown')}")
            logger.info(f"Key Offerings: {', '.join(result.get('key_offerings', []))}")
            logger.info(f"Strategic Focus: {', '.join(result.get('strategic_focus', []))}")
            logger.info(f"Vision: {result.get('vision', 'Unknown')}")
            
            # Print competitors
            logger.info("\nCompetitors:")
            for competitor in result.get('competitors', []):
                logger.info(f"- {competitor.get('name', 'Unknown')}: {competitor.get('description', 'No description')}")
            
            # Print market trends
            logger.info("\nMarket Trends:")
            for trend in result.get('market_trends', []):
                logger.info(f"- {trend.get('name', 'Unknown')}: {trend.get('description', 'No description')}")
            
            # Save results to file
            filename = f"research_results_{test_case['name'].lower().replace(' ', '_')}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"\nResults saved to {filename}")
            
            total_results.append({
                'test_case': test_case['name'],
                'success': True,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Critical error testing {test_case['name']}: {str(e)}")
            critical_errors.append({
                'test_case': test_case['name'],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            # Stop execution on critical errors
            break

    # Print performance metrics
    execution_time = time.time() - start_time
    metrics = research_agent.get_performance_metrics()
    key_stats = research_agent.load_balancer.get_key_stats()
    
    logger.info(f"\n{'='*50}")
    logger.info("Performance Metrics:")
    logger.info(f"{'='*50}")
    logger.info(f"Total Execution Time: {execution_time:.2f} seconds")
    logger.info(f"Total Searches: {metrics['total_searches']}")
    logger.info(f"Successful Searches: {metrics['successful_searches']}")
    logger.info(f"Failed Searches: {metrics['failed_searches']}")
    logger.info(f"Cache Hits: {metrics['cache_hits']}")
    logger.info(f"Cache Misses: {metrics['cache_misses']}")
    logger.info(f"Success Rate: {(metrics['successful_searches'] / metrics['total_searches'] * 100):.2f}%")
    logger.info(f"Cache Hit Rate: {(metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses']) * 100):.2f}%")
    
    logger.info(f"\nKey Load Balancing Statistics:")
    logger.info(f"Total Keys: {key_stats['total_keys']}")
    logger.info(f"Available Keys: {key_stats['available_keys']}")
    logger.info(f"Failed Keys: {key_stats['failed_keys']}")
    logger.info(f"Key Errors: {key_stats['key_errors']}")
    logger.info(f"Current Usage: {key_stats['current_usage']}")
    
    # Save test summary
    summary = {
        'execution_time': execution_time,
        'performance_metrics': metrics,
        'key_load_balancing': key_stats,
        'test_results': total_results,
        'critical_errors': critical_errors,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nTest summary saved to test_summary.json")
    
    # Exit with error code if there were critical errors
    if critical_errors:
        logger.error("Test failed due to critical errors")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_research_agent()) 