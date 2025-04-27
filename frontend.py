import streamlit as st
import requests
import os
from typing import Dict, List, Optional

API_URL = os.getenv('API_URL', 'http://localhost:8000')

# Configure page
st.set_page_config(
    page_title="GenAI Use Case Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'user_name' not in st.session_state:
    st.session_state['user_name'] = ''
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'errors' not in st.session_state:
    st.session_state['errors'] = None
if 'loading' not in st.session_state:
    st.session_state['loading'] = False

# Title and description
st.title("🚀 Market Research & GenAI Use Case Generator")
st.markdown("""
This tool helps you research companies and industries, generate AI use cases, and find relevant resources.
Enter a company name or industry to get started.
""")

# Sidebar
with st.sidebar:
    st.header("User Info")
    st.session_state['user_name'] = st.text_input(
        "Enter your name",
        st.session_state['user_name']
    )
    company = st.text_input("Company or Industry", "Cadence")
    
    if st.button("Run Research & Generate Use Cases"):
        st.session_state['results'] = None
        st.session_state['errors'] = None
        st.session_state['loading'] = True
        
        try:
            with st.spinner("Running full pipeline..."):
                resp = requests.post(
                    f"{API_URL}/full_pipeline",
                    json={
                        "company_or_industry_name": company,
                        "user_name": st.session_state['user_name'] or 'User'
                    },
                    timeout=300
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state['results'] = data.get('results', {})
                    st.session_state['errors'] = data.get('errors')
                    st.success("Results ready! Slack notification sent if configured.")
                else:
                    st.error(f"API error: {resp.text}")
                    
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Please check if the server is running.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            st.session_state['loading'] = False

# Display results
if st.session_state['loading']:
    st.info("Processing your request... Please wait.")

elif st.session_state['results']:
    res = st.session_state['results']
    errors = st.session_state['errors']
    
    # Display any errors
    if errors:
        st.warning("Some operations encountered errors:")
        for operation, error in errors.items():
            st.error(f"{operation}: {error}")
    
    # Research Summary
    st.header("1. Research Summary")
    if res.get('industry_info'):
        st.write(res['industry_info'].get('summary', 'No summary available'))
        st.markdown(f"**Industry:** {res['industry_info'].get('industry', 'Unknown')}")
        st.markdown(f"**Segment:** {res['industry_info'].get('segment', 'Unknown')}")
        st.markdown(f"**Key Offerings:** {', '.join(res['industry_info'].get('key_offerings', ['None']))}")
        st.markdown(f"**Strategic Focus:** {', '.join(res['industry_info'].get('strategic_focus', ['None']))}")
        st.markdown(f"**Vision:** {res['industry_info'].get('vision', 'Unknown')}")
        st.markdown(f"**Products:** {', '.join(res['industry_info'].get('products', ['None']))}")
    else:
        st.warning("No industry information available")

    # AI/GenAI Use Cases
    st.header("2. AI/GenAI Use Cases")
    if res.get('usecases'):
        for uc in res['usecases']:
            with st.expander(uc.get('use_case', 'Unknown Use Case')):
                st.write(uc.get('description', 'No description available'))
                if uc.get('reference'):
                    st.markdown(f"[Reference]({uc.get('reference')})")
    else:
        st.warning("No use cases available")

    # Resource Assets
    st.header("3. Resource Assets")
    if res.get('resources'):
        for uc in res['resources']:
            with st.expander(uc.get('use_case', 'Unknown Use Case')):
                for r in uc.get('resources', []):
                    if isinstance(r, dict) and 'url' in r:
                        st.markdown(f"- [{r.get('name', 'Unknown')}]({r.get('url')})")
                    else:
                        st.markdown(f"- {r}")
    else:
        st.warning("No resources available")

    # Validation & Scores
    st.header("4. Validation & Scores")
    if res.get('validated'):
        for uc in res['validated']:
            with st.expander(uc.get('use_case', 'Unknown Use Case')):
                st.markdown(f"**Creativity:** {uc.get('creativity_score', 'N/A')}")
                st.markdown(f"**Feasibility:** {uc.get('feasibility_score', 'N/A')}")
                st.markdown(f"**Reference Quality:** {uc.get('reference_quality', 'N/A')}")
    else:
        st.warning("No validation results available")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and FastAPI") 