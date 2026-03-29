import streamlit as st
import plotly.graph_objects as go
from utils.history import score_color

DIM_LABELS = {
    "therapeutic_approach": "Therapeutic Approach",
    "monitoring_and_risk": "Monitoring & Risk",
    "harm_non_reinforcement": "Harm / Non-reinforce",
    "therapeutic_alliance": "Therapeutic Alliance",
    "instruction_following": "Instruction Following",
    "implicit_risk_detection": "Implicit Risk Detection",
    "default_safety_behavior": "Default Safety",
    "hallucination": "Hallucination Avoid.",
    "consistency": "Consistency",
}


def dim_label(key: str) -> str:
    return DIM_LABELS.get(key, key.replace("_", " ").title())


def render_score_card(dim_key: str, score: int, reason: str = ""):
    color = score_color(score)
    tier = "score-high" if score >= 4 else ("score-mid" if score >= 3 else "score-low")
    label = dim_label(dim_key)
    st.markdown(
        f"""<div class="score-card {tier}">
        <strong>{label}</strong>
        <span style="float:right;color:{color};font-weight:700;font-size:18px">{score}/5</span>
        </div>""",
        unsafe_allow_html=True,
    )
    if reason:
        st.caption(reason)


def render_radar(scores: dict, title: str = "", color: str = "#c4705a"):
    if not scores:
        return
    dims = list(scores.keys())
    vals = [scores[d] for d in dims]
    labels = [dim_label(d) for d in dims]
    vals_closed = vals + [vals[0]]
    labels_closed = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor=f"rgba({_hex_rgb(color)},0.2)",
        line=dict(color=color, width=2),
        name=title,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=bool(title),
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_radar_overlay(versions: list, colors: list):
    """Overlay multiple radar charts. versions = list of {label, scores}."""
    fig = go.Figure()
    # Render in reverse so first item is on top
    for idx, (v, color) in enumerate(zip(versions, colors)):
        scores = v.get("scores", {})
        if not scores:
            continue
        dims = list(scores.keys())
        vals = [scores[d] for d in dims]
        labels = [dim_label(d) for d in dims]
        vals_closed = vals + [vals[0]]
        labels_closed = labels + [labels[0]]
        # Vary opacity so overlapping shapes don't hide each other
        opacity = 0.25 - (idx * 0.04)
        opacity = max(0.08, opacity)
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor=f"rgba({_hex_rgb(color)},{opacity})",
            line=dict(color=color, width=3),
            marker=dict(size=6, color=color),
            name=v.get("label", ""),
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True,
        legend=dict(orientation="v", x=1.05),
        margin=dict(l=60, r=120, t=40, b=40),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_chat_bubble(role: str, text: str):
    css_class = "bubble-user" if role == "user" else "bubble-bot"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def render_suggestion_box(suggestion: str):
    st.markdown(
        f'<div style="background:#fff8f5;border-left:4px solid #c4705a;'
        f'padding:12px 16px;border-radius:8px;margin-top:8px;">'
        f'<strong>💡 Suggestion</strong><br>{suggestion}</div>',
        unsafe_allow_html=True,
    )


def no_judge_warning():
    st.warning("No API key available for the judge. Add at least one key in **API Settings**.")


def no_model_warning():
    st.warning("No models available. Go to **API Settings** to add your API keys.")


def _hex_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"
