import streamlit as st
import plotly.graph_objects as go
from utils.history import score_color

DIM_LABELS = {
    "therapeutic_approach": "Therapeutic Approach",
    "monitoring_and_risk": "Crisis Recognition & Response",
    "harm_non_reinforcement": "Harm / Non-reinforce",
    "therapeutic_alliance": "Therapeutic Alliance",
    "instruction_following": "Instruction Following",
    "response_proportionality": "Response Proportionality",
    "implicit_risk_detection": "Crisis Recognition & Response",
    "default_safety_behavior": "Default Safety",
    "hallucination": "Hallucination",
    "consistency": "Consistency",
    "scope_boundary": "Scope & Boundary",
    "referral_quality": "Supportive Outreach",
    "supportive_outreach": "Supportive Outreach",
}


def _hex_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


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
    all_dims = list(scores.keys())
    all_labels = [dim_label(d) for d in all_dims]

    # Non-null points only for the visible trace
    active_dims = [d for d in all_dims if scores[d] is not None]
    active_vals = [scores[d] for d in active_dims]
    active_labels = [dim_label(d) for d in active_dims]

    fig = go.Figure()
    # Invisible anchor trace — ensures all axis labels appear even when a dim is null
    fig.add_trace(go.Scatterpolar(
        r=[0] * (len(all_dims) + 1),
        theta=all_labels + [all_labels[0]],
        mode="lines",
        line=dict(color="rgba(0,0,0,0)", width=0),
        showlegend=False,
        hoverinfo="skip",
    ))
    # Visible data trace — only non-null dims, forms a clean polygon
    if active_vals:
        vals_closed = active_vals + [active_vals[0]]
        labels_closed = active_labels + [active_labels[0]]
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
        all_dims = list(scores.keys())
        all_labels = [dim_label(d) for d in all_dims]
        # Only plot non-null points; anchor invisible trace keeps all axes visible
        active_dims = [d for d in all_dims if scores[d] is not None]
        active_vals = [scores[d] for d in active_dims]
        active_labels = [dim_label(d) for d in active_dims]
        if not active_vals:
            continue
        vals_closed = active_vals + [active_vals[0]]
        labels_closed = active_labels + [active_labels[0]]
        # Vary opacity so overlapping shapes don't hide each other
        opacity = 0.25 - (idx * 0.04)
        opacity = max(0.08, opacity)
        # Anchor trace to keep all axes visible
        fig.add_trace(go.Scatterpolar(
            r=[0] * (len(all_dims) + 1),
            theta=all_labels + [all_labels[0]],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor=f"rgba({_hex_rgb(color)},{opacity})",
            line=dict(color=color, width=3),
            marker=dict(size=6, color=color),
            name=v.get("label", ""),
            connectgaps=False,
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


def render_non_sequitur_warning():
    st.markdown(
        '<div style="background:#fff0f0;border-left:4px solid #e53935;'
        'padding:12px 16px;border-radius:8px;margin-top:8px;margin-bottom:4px;">'
        '<strong>⚠️ Non-sequitur detected</strong><br>'
        '<span style="font-size:0.9em;">The chatbot response appears unrelated to the user\'s message. '
        'All safety dimensions have been scored 1. Review the response carefully before iterating.</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_suggestion_box(suggestion):
    """Render judge suggestions. Handles:
    - New structured format: list of {dimension, problem, prompt_fix, priority}
    - Legacy string format: '1. x 2. y 3. z'
    """
    import re, json

    PRIORITY_COLOR = {"High": "#e53935", "Medium": "#f39c12", "Low": "#27ae60"}
    PRIORITY_BG = {"High": "#fff0f0", "Medium": "#fffbf0", "Low": "#f0fff4"}

    # ── Try to parse as JSON list ─────────────────────────────────────────
    parsed_list = None
    if isinstance(suggestion, list):
        parsed_list = suggestion
    elif isinstance(suggestion, str):
        s = suggestion.strip()
        # Strip markdown fences if present
        if s.startswith("```"):
            s = re.sub(r"```[a-z]*\n?", "", s).replace("```", "").strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                parsed_list = parsed
        except Exception:
            pass

    # ── Render structured cards ───────────────────────────────────────────
    if parsed_list:
        rows = ""
        for item in parsed_list:
            if not isinstance(item, dict):
                continue
            dim = item.get("dimension", "")
            problem = item.get("problem", "")
            fix = item.get("prompt_fix", "")
            priority = item.get("priority", "Medium")
            p_color = PRIORITY_COLOR.get(priority, "#888")
            p_bg = PRIORITY_BG.get(priority, "#fafafa")
            dim_display = dim_label(dim) if dim else dim
            safe_problem = problem.replace("<", "&lt;").replace(">", "&gt;")
            safe_fix = fix.replace("<", "&lt;").replace(">", "&gt;")
            rows += (
                f'<div style="margin-top:10px;padding:10px 12px;background:{p_bg};' 
                f'border-radius:6px;font-size:0.9em;line-height:1.6;">' 
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">' 
                f'<span style="font-weight:700;color:#c4705a;">{dim_display}</span>' 
                f'<span style="font-size:0.8em;font-weight:700;color:{p_color};border:1px solid {p_color};' 
                f'border-radius:4px;padding:1px 6px;">{priority}</span></div>' 
                f'<div style="color:#555;margin-bottom:4px;"><strong>Problem:</strong> {safe_problem}</div>' 
                f'<div style="color:#333;"><strong>Fix:</strong> <code style="background:#f0f0f0;padding:2px 5px;border-radius:3px;font-size:0.85em;">{safe_fix}</code></div></div>'
            )
        if rows:
            st.markdown(
                '<div style="background:#fff8f5;border-left:4px solid #c4705a;padding:12px 16px;border-radius:8px;margin-top:8px;">' 
                '<strong>💡 Suggestions for your system prompt</strong>' + rows + '</div>',
                unsafe_allow_html=True,
            )
        return

    # ── Legacy string fallback ────────────────────────────────────────────
    if not isinstance(suggestion, str):
        suggestion = str(suggestion)

    if "'strength'" in suggestion[:50] or '"strength"' in suggestion[:50]:
        try:
            import ast
            parsed = ast.literal_eval(suggestion.strip())
            s_text = parsed.get("strength", "")
            f_text = parsed.get("failure", "")
        except Exception:
            s_text = ""
            f_text = ""
        rows = ""
        if s_text:
            rows += f'<div style="margin-top:10px;padding:8px 10px;background:#f0faf0;border-radius:6px;font-size:0.9em;"><span style="font-weight:700;color:#27ae60;">✅ Strength</span><br>{s_text}</div>'
        if f_text:
            rows += f'<div style="margin-top:8px;padding:8px 10px;background:#fff3ee;border-radius:6px;font-size:0.9em;"><span style="font-weight:700;color:#c4705a;">⚠️ Area to improve</span><br>{f_text}</div>'
        if rows:
            st.markdown(
                '<div style="background:#fff8f5;border-left:4px solid #c4705a;padding:12px 16px;border-radius:8px;margin-top:8px;">' 
                '<strong>💡 Suggestions for your system prompt</strong>' + rows + '</div>',
                unsafe_allow_html=True,
            )
        return

    raw_parts = re.split(r'(?<![\d])([1-9])\.\\ ', suggestion)
    items = []
    i = 1
    while i < len(raw_parts) - 1:
        num = raw_parts[i].strip()
        text = raw_parts[i + 1].strip() if i + 1 < len(raw_parts) else ""
        if num.isdigit() and text:
            text = re.sub(
                r'^([a-z][a-z_]+):',
                lambda m: m.group(1).replace("_", " ").title() + ":",
                text
            )
            items.append((num, text))
        i += 2

    if not items:
        safe = suggestion.replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f'<div style="background:#fff8f5;border-left:4px solid #c4705a;padding:12px 16px;border-radius:8px;margin-top:8px;">' 
            f'<strong>💡 Suggestions for your system prompt</strong>' 
            f'<div style="margin-top:8px;font-size:0.9em;white-space:pre-wrap;">{safe}</div></div>',
            unsafe_allow_html=True,
        )
        return

    rows = ""
    for num, text in items:
        safe_text = text.replace("<", "&lt;").replace(">", "&gt;")
        rows += (
            f'<div style="margin-top:10px;padding:10px 12px;background:#fff3ee;' 
            f'border-radius:6px;font-size:0.9em;line-height:1.5;">' 
            f'<span style="font-weight:700;color:#c4705a;margin-right:6px;">{num}.</span>' 
            f'{safe_text}</div>'
        )
    st.markdown(
        '<div style="background:#fff8f5;border-left:4px solid #c4705a;padding:12px 16px;border-radius:8px;margin-top:8px;">' 
        '<strong>💡 Suggestions for your system prompt</strong>' + rows + '</div>',
        unsafe_allow_html=True,
    )




def no_judge_warning():
    st.warning("No API key available for the judge. Add at least one key in **API Settings**.")


def no_model_warning():
    st.warning("No models available. Go to **API Settings** to add your API keys.")
