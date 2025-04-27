import os
import json
import logging
import argparse
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from agents import research_agent, usecase_agent, resource_agent, validation_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('market_research.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def save_to_markdown(results: List[Dict], output_file: str = 'market_research_report.md') -> None:
    """
    Save results to a markdown file.
    """
    try:
        with open(output_file, 'w') as f:
            f.write('# Market Research & Use Case Generation Report\n\n')
            f.write(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            
            # Write industry information
            f.write('## Industry Analysis\n\n')
            industry_info = results[0].get('industry_info', {})
            f.write(f'### Industry: {industry_info.get("industry", "Unknown")}\n')
            f.write(f'### Segment: {industry_info.get("segment", "Unknown")}\n\n')
            f.write('#### Key Offerings:\n')
            for offering in industry_info.get('key_offerings', []):
                f.write(f'- {offering}\n')
            f.write('\n')
            
            f.write('#### Strategic Focus:\n')
            for focus in industry_info.get('strategic_focus', []):
                f.write(f'- {focus}\n')
            f.write('\n')
            
            f.write(f'#### Vision: {industry_info.get("vision", "Not available")}\n\n')
            
            # Write use cases
            f.write('## Generated Use Cases\n\n')
            for uc in results:
                f.write(f'### {uc.get("use_case", "Unknown Use Case")}\n\n')
                f.write(f'**Description:** {uc.get("description", "No description available")}\n\n')
                
                # Write scores
                scores = uc.get('scores', {})
                f.write('#### Scores:\n')
                f.write(f'- Overall Score: {scores.get("overall_score", 0):.2f}\n')
                f.write(f'- Creativity: {scores.get("creativity_score", 0):.2f}\n')
                f.write(f'- Feasibility: {scores.get("feasibility_score", 0):.2f}\n')
                f.write(f'- Impact: {scores.get("impact_score", 0):.2f}\n')
                f.write(f'- Innovation: {scores.get("innovation_score", 0):.2f}\n')
                f.write(f'- Resource: {scores.get("resource_score", 0):.2f}\n\n')
                
                # Write validation results
                validation = uc.get('validation', {})
                f.write('#### Validation:\n')
                f.write(f'- Status: {"✅ Valid" if validation.get("is_valid") else "❌ Invalid"}\n')
                if validation.get('issues'):
                    f.write('- Issues:\n')
                    for issue in validation['issues']:
                        f.write(f'  - {issue}\n')
                if validation.get('recommendations'):
                    f.write('- Recommendations:\n')
                    for rec in validation['recommendations']:
                        f.write(f'  - {rec}\n')
                f.write('\n')
                
                # Write resources
                f.write('#### Resources:\n')
                for resource in uc.get('resources', []):
                    f.write(f'- [{resource.get("name", "Unknown")}]({resource.get("url", "#")})\n')
                    f.write(f'  - Source: {resource.get("source", "Unknown")}\n')
                    f.write(f'  - Type: {resource.get("type", "Unknown")}\n')
                    f.write(f'  - Quality Score: {resource.get("quality_score", 0):.2f}\n\n')
                
                f.write('---\n\n')
            
            logger.info(f"Report saved to {output_file}")
            
    except Exception as e:
        logger.error(f"Error saving report: {str(e)}")
        raise

def main(company_or_industry_name: str, user_name: str = 'User', output_file: str = 'market_research_report.md') -> None:
    """
    Main entry point for the multi-agent workflow.
    """
    try:
        logger.info(f"Starting market research for {company_or_industry_name}")
        
        # 1. Research the company or industry
        logger.info("Starting industry research...")
        industry_info = research_agent.run(company_or_industry_name)
        if not industry_info or 'error' in industry_info:
            raise Exception(f"Industry research failed: {industry_info.get('error', 'Unknown error')}")
        logger.info("Industry research completed successfully")
        
        # 2. Generate use cases based on research
        logger.info("Generating use cases...")
        usecases = usecase_agent.run(industry_info)
        if not usecases:
            raise Exception("Use case generation failed")
        logger.info(f"Generated {len(usecases)} use cases")
        
        # 3. Collect resource assets for each use case
        logger.info("Collecting resources...")
        resources = resource_agent.run(usecases)
        if not resources:
            raise Exception("Resource collection failed")
        logger.info("Resource collection completed successfully")
        
        # 4. Validate and score use cases
        logger.info("Validating use cases...")
        validated = validation_agent.run(usecases, resources, user_name=user_name)
        if not validated:
            raise Exception("Use case validation failed")
        logger.info("Use case validation completed successfully")
        
        # 5. Save results
        logger.info("Saving results...")
        save_to_markdown(validated, output_file)
        logger.info("Results saved successfully")
        
        logger.info("Market research process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in market research process: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Market Research & Use Case Generation')
    parser.add_argument('company_or_industry', help='Company or industry name to research')
    parser.add_argument('--user', default='User', help='User name for notifications')
    parser.add_argument('--output', default='market_research_report.md', help='Output file path')
    
    args = parser.parse_args()
    
    try:
        main(args.company_or_industry, args.user, args.output)
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        exit(1)
