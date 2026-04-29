import streamlit as st
from datetime import datetime


def _init(key="version_history"):
    if key not in st.session_state:
        st.session_state[key] = []


def save_version(record: dict, key="version_history"):
    _init(key)
    record["timestamp"] = datetime.now().strftime("%H:%M:%S")
    record["id"] = len(st.session_state[key]) + 1
    st.session_state[key].append(record)


def get_history(key="version_history"):
    _init(key)
    return st.session_state[key]


def avg_score(scores: dict) -> float:
    if not scores:
        return 0.0
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def score_color(score: float) -> str:
    if score >= 4:
        return "#2ecc71"
    elif score >= 3:
        return "#f39c12"
    return "#e74c3c"
