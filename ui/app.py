import streamlit as st
import requests

# Page configurations
st.set_page_config(
    page_title="Legal Clause Fairness Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# App Title & Headers
st.title("⚖️ Legal Clause Fairness Analysis Engine")
st.markdown("""
This production dashboard evaluates Terms of Service (ToS) clauses for potential unfairness 
using a hybrid pipeline featuring a **Legal-BERT SVC-RBF Classifier** integrated with a 
**Bayesian Semantic Neighborhood post-processor**.
""")

st.sidebar.header("Configuration")
FASTAPI_URL = st.sidebar.text_input("API Endpoint URL", value="http://localhost:8000/analyze")
top_k = st.sidebar.slider("Semantic Neighbors (Top K)", min_value=1, max_value=10, value=5)

st.markdown("---")

# Main user input block
user_clause = st.text_area(
    "Enter a contract clause / provision to evaluate:",
    placeholder="e.g., The company reserves the right to modify these terms at any time without prior notice to the user.",
    height=100
)

if st.button("Run Compliance Analysis", type="primary"):
    if not user_clause.strip():
        st.warning("Please enter a valid text clause to analyze.")
    else:
        with st.spinner("Analyzing semantics and legal risk topology..."):
            try:
                # Prepare payload matching your AnalysisRequest schema
                payload = {
                    "sentence": user_clause,
                    "top_k": top_k
                }
                
                # Query your FastAPI server
                response = requests.post(FASTAPI_URL, json=payload, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 1. Display Classification Verdict Header
                    is_unfair = data.get("is_unfair", False)
                    status_label = data.get("status_label", "UNKNOWN")
                    
                    if is_unfair:
                        st.error(f"🚨 **Verdict: {status_label}**")
                    else:
                        st.success(f"✅ **Verdict: {status_label}**")
                        
                    # Layout columns for detailed breakdown
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("### Clause Metrics")
                        st.metric(
                            label="Target Status", 
                            value="Flagged / High Risk" if is_unfair else "Compliant / Fair",
                            delta="Action Required" if is_unfair else "Safe Base-rate"
                        )
                        st.caption("Classification verified via out-of-fold calibrated probability boundaries.")
                        
                    with col2:
                        st.markdown("### Near Semantic Matches (Historical Case Law)")
                        matches = data.get("top_matches", [])
                        
                        if not matches:
                            st.info("No close semantic matches found in the historical vector database.")
                        else:
                            for match in matches:
                                rank = match.get("rank")
                                score = match.get("similarity_score", 0.0)
                                company = match.get("company", "Unknown")
                                text = match.get("text", "")
                                truth = match.get("historical_ground_truth", "Fair")
                                
                                # Visual styling tag for historical ground truth match
                                badge = "🔴 [Unfair]" if truth == "Unfair" else "🟢 [Fair]"
                                
                                with st.expander(f"Rank {rank}: Cosine Sim {score:.4f} | {company} {badge}"):
                                    st.markdown(f"*{text}*")
                                    
                else:
                    st.error(f"API returned an error code: {response.status_code}")
                    st.json(response.json())
                    
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to the FastAPI server at `{FASTAPI_URL}`. Make sure your Uvicorn app is running!")
            except Exception as e:
                st.error(f"An unexpected UI handling error occurred: {str(e)}")