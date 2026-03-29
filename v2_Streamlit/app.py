import streamlit as st

st.set_page_config(
    page_title="MH Chatbot Safety Evaluator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from pages_content import prompt_eval, model_compare, api_settings
from utils.styles import inject_styles

inject_styles()

NAV_OPTIONS = ["⚙️ API Settings", "🔬 Prompt Evaluation", "📊 Model Comparison"]

# Handle nav button redirects from the home page
if "_nav_target" in st.session_state:
    target = st.session_state.pop("_nav_target")
    if target in NAV_OPTIONS:
        st.session_state["_current_page"] = target

if "_current_page" not in st.session_state:
    st.session_state["_current_page"] = "⚙️ API Settings"

with st.sidebar:
    st.markdown("## 🧠 MH Safety Evaluator")
    st.caption("Mental Health Chatbot Safety Evaluation Tool")
    st.divider()
    page = st.radio(
        "nav",
        NAV_OPTIONS,
        index=NAV_OPTIONS.index(st.session_state["_current_page"]),
        label_visibility="collapsed",
    )
    st.session_state["_current_page"] = page
    st.divider()
    st.caption("AI Safety Research")

if page == "⚙️ API Settings":
    api_settings.render()
elif page == "🔬 Prompt Evaluation":
    prompt_eval.render()
elif page == "📊 Model Comparison":
    model_compare.render()
