# Market Research & Use Case Generation Agent

A multi-agent system that generates relevant AI and Generative AI (GenAI) use cases for companies and industries. The system conducts market research, understands the industry and product, and provides resource assets for AI/ML solutions, focusing on enhancing operations and customer experiences.

## Features

- **Industry Research Agent**: Researches and analyzes companies and industries using multiple data sources
- **Use Case Generation Agent**: Generates relevant AI/GenAI use cases based on industry research
- **Resource Collection Agent**: Collects and validates relevant datasets and resources
- **Validation Agent**: Evaluates and scores use cases based on industry-specific criteria

## Architecture

The system uses a multi-agent architecture with the following components:

1. **Research Agent**
   - Web search and analysis
   - Industry segmentation
   - Company analysis
   - Caching for faster results

2. **Use Case Agent**
   - LLM-powered use case generation
   - Industry-specific use cases
   - Validation and scoring
   - Reference tracking

3. **Resource Agent**
   - Dataset search and validation
   - Resource quality scoring
   - Multiple platform integration
   - Caching for faster results

4. **Validation Agent**
   - Industry-specific validation rules
   - Comprehensive scoring system
   - Feedback mechanism
   - Slack notifications

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/market-research-genai.git
cd market-research-genai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Usage

### Command Line Interface

```bash
python main.py "Company Name" --user "Your Name" --output "report.md"
```

### Python API

```python
from agents import research_agent, usecase_agent, resource_agent, validation_agent

# Research the company/industry
industry_info = research_agent.run("Company Name")

# Generate use cases
usecases = usecase_agent.run(industry_info)

# Collect resources
resources = resource_agent.run(usecases)

# Validate use cases
validated = validation_agent.run(usecases, resources, user_name="Your Name")
```

## Configuration

The system can be configured through environment variables:

- `REDIS_URL`: Redis connection URL for caching
- `HUGGINGFACE_API_KEY`: HuggingFace API key for LLM access
- `GOOGLE_API_KEY`: Google Custom Search API key
- `GOOGLE_SEARCH_ENGINE_ID`: Google Custom Search Engine ID
- `GITHUB_TOKEN`: GitHub API token
- `SLACK_WEBHOOK_URL`: Slack webhook URL for notifications

## Output

The system generates a comprehensive markdown report containing:

1. Industry Analysis
   - Industry and segment information
   - Key offerings and strategic focus
   - Vision and mission

2. Generated Use Cases
   - Detailed descriptions
   - Implementation challenges
   - Required technologies
   - Impact assessment

3. Resource Assets
   - Dataset links
   - Quality scores
   - Source information
   - Relevance assessment

4. Validation Results
   - Overall scores
   - Industry-specific validation
   - Recommendations
   - Issues and solutions

## Development

### Running Tests

```bash
pytest
```

### Code Style

```bash
black .
isort .
flake8
mypy .
```

### Documentation

```bash
cd docs
make html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- HuggingFace for providing free LLM APIs
- Redis for caching support
- Various open-source libraries and tools

## Contact

For questions and support, please open an issue in the GitHub repository.
