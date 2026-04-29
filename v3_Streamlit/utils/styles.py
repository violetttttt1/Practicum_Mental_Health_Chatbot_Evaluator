import streamlit as st


def inject_styles():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background-color: #ffffff; }
section[data-testid="stSidebar"] { background-color: #f7f7f7; }

.score-card {
    background: #fafafa;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 6px;
    border-left: 4px solid #ccc;
}
.score-high { border-left-color: #27ae60; }
.score-mid  { border-left-color: #e67e22; }
.score-low  { border-left-color: #e74c3c; }

.bubble-user {
    background: #f0f0f0;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 14px;
    margin: 6px 0;
    max-width: 85%;
    margin-left: auto;
    font-size: 14px;
}
.bubble-bot {
    background: #ffffff;
    border-radius: 16px 16px 16px 4px;
    padding: 10px 14px;
    margin: 6px 0;
    max-width: 85%;
    border: 1px solid #e0e0e0;
    font-size: 14px;
}

.verdict-box {
    background: #eaf3fb;
    border: 1px solid #93c5e8;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.verdict-winner {
    background: #edfaf1;
    border: 1px solid #27ae60;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
}
.verdict-concern {
    background: #fef9e7;
    border: 1px solid #e67e22;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 12px;
}

hr { border: none; border-top: 1px solid #eee; margin: 12px 0; }

/* Make selectbox dropdown options wrap to full text */
[data-baseweb="select"] [role="option"] {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    padding: 10px 14px !important;
    line-height: 1.5 !important;
}
[data-baseweb="popover"] li {
    white-space: normal !important;
    line-height: 1.5 !important;
    padding: 8px 14px !important;
}
</style>
""", unsafe_allow_html=True)

