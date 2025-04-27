import asyncio
from agents.usecase_agent import usecase_agent

async def main():
    # Prepare industry research data
    industry_research = {
        'industry': 'Technology',
        'segment': 'AI/ML',
        'key_offerings': ['AI Solutions', 'ML Services'],
        'strategic_focus': ['Innovation', 'Digital Transformation'],
        'vision': 'Leading AI solutions provider',
        'products': ['AI Platform', 'ML Tools']
    }
    
    # Run the agent
    result = await usecase_agent.run(industry_research)
    
    # Process the results
    print("Generated Use Cases:")
    for use_case in result['use_cases']:
        print(f"\nTitle: {use_case['title']}")
        print(f"Description: {use_case['description']}")
        print(f"Business Value: {use_case['business_value']}")
        print(f"Technical Approach: {use_case['technical_approach']}")
        print(f"Innovation Potential: {use_case['innovation_potential']}")
        print(f"Overall Score: {use_case['overall_score']:.2f}")

if __name__ == "__main__":
    asyncio.run(main()) 