import streamlit as st

st.set_page_config(
    page_title="MH Chatbot Safety Evaluator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from pages_content import prompt_eval, model_compare, api_settings, direct_eval
from utils.styles import inject_styles

inject_styles()

NAV_OPTIONS = ["⚙️ API Settings", "🔬 Prompt Evaluation", "📊 Model Comparison", "📋 Direct Evaluation"]

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
    st.divider()
    st.caption("AI Safety Research")

_page_changed = page != st.session_state.get("_prev_page")
if _page_changed:
    st.session_state["_prev_page"] = page

st.session_state["_current_page"] = page

# Reset tab state when navigating away from each page
if _page_changed:
    if page != "📋 Direct Evaluation":
        st.session_state["de_reset_tab"] = True
    if page != "📊 Model Comparison":
        st.session_state["mc_reset_tab"] = True
    if page != "🔬 Prompt Evaluation":
        st.session_state["pe_reset_tab"] = True

if page == "⚙️ API Settings":
    api_settings.render()
elif page == "🔬 Prompt Evaluation":
    prompt_eval.render()
elif page == "📊 Model Comparison":
    model_compare.render()
elif page == "📋 Direct Evaluation":
    direct_eval.render()

# Inject scroll AFTER page content renders — must be last
if _page_changed:
    import streamlit.components.v1 as _cv1
    _scroll_js = """<script>
setTimeout(function() {
  var candidates = [
    document.querySelector('section.main'),
    window.parent ? window.parent.document.querySelector('section.main') : null,
    window.parent ? window.parent.document.querySelector('[data-testid="stMain"]') : null,
    window.parent ? window.parent.document.querySelector('[data-testid="stAppViewContainer"]') : null
  ];
  for (var i = 0; i < candidates.length; i++) {
    if (candidates[i]) { candidates[i].scrollTo({top: 0, behavior: 'instant'}); break; }
  }
}, 80);
</script>"""
    _cv1.html(_scroll_js, height=1)
