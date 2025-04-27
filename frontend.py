import streamlit as st
import requests
import os

API_URL = os.getenv('API_URL', 'http://localhost:8000')

st.set_page_config(page_title="GenAI Use Case Generator", layout="wide")
st.title("🚀 Market Research & GenAI Use Case Generator")

# User session state
if 'user_name' not in st.session_state:
    st.session_state['user_name'] = ''
if 'results' not in st.session_state:
    st.session_state['results'] = None

st.sidebar.header("User Info")
st.session_state['user_name'] = st.sidebar.text_input("Enter your name", st.session_state['user_name'])
company = st.sidebar.text_input("Company or Industry", "Cadence")

if st.sidebar.button("Run Research & Generate Use Cases"):
    st.session_state['results'] = None
    with st.spinner("Running full pipeline..."):
        try:
            resp = requests.post(f"{API_URL}/full_pipeline", json={
                "company_or_industry_name": company,
                "user_name": st.session_state['user_name'] or 'User'
            }, timeout=120)
            if resp.status_code == 200:
                st.session_state['results'] = resp.json()
                st.success("Results ready! Slack notification sent if configured.")
            else:
                st.error(f"API error: {resp.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")

if st.session_state['results']:
    res = st.session_state['results']
    st.header("1. Research Summary")
    st.write(res['industry_info'].get('summary', ''))
    st.markdown(f"**Industry:** {res['industry_info'].get('industry','')}  ")
    st.markdown(f"**Segment:** {res['industry_info'].get('segment','')}  ")
    st.markdown(f"**Key Offerings:** {', '.join(res['industry_info'].get('key_offerings', []))}")
    st.markdown(f"**Strategic Focus:** {', '.join(res['industry_info'].get('strategic_focus', []))}")
    st.markdown(f"**Vision:** {res['industry_info'].get('vision','')}")
    st.markdown(f"**Products:** {', '.join(res['industry_info'].get('products', []))}")

    st.header("2. AI/GenAI Use Cases")
    for uc in res['usecases']:
        st.subheader(uc.get('use_case',''))
        st.write(uc.get('description',''))
        st.markdown(f"[Reference]({uc.get('reference','')})")

    st.header("3. Resource Assets")
    for uc in res['resources']:
        st.markdown(f"**{uc.get('use_case','')}**")
        for r in uc.get('resources', []):
            if 'url' in r:
                st.markdown(f"- [{r.get('name','')}]({r.get('url','')})")
            else:
                st.markdown(f"- {r}")

    st.header("4. Validation & Scores")
    for uc in res['validated']:
        st.markdown(f"**{uc.get('use_case','')}**: Creativity {uc.get('creativity_score','')}, Feasibility {uc.get('feasibility_score','')}, Reference Quality {uc.get('reference_quality','')}") 