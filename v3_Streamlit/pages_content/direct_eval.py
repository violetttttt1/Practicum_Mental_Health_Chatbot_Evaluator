import streamlit as st
from utils.api_utils import run_judge, get_judge
from utils.history import avg_score, score_color, save_version, get_history
from utils.components import (
    render_score_card, render_radar, render_radar_overlay,
    render_suggestion_box, render_non_sequitur_warning, no_judge_warning, dim_label,
)

DIRECT_DIMENSIONS = [
    "therapeutic_approach",
    "monitoring_and_risk",
    "harm_non_reinforcement",
    "therapeutic_alliance",
    "instruction_following",
    "default_safety_behavior",
    "hallucination",
    "scope_boundary",
    "supportive_outreach",
]

VERSION_COLORS = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad"]


def _init_state():
    defaults = {
        "de_system_prompt": "",
        "de_user_msg": "",
        "de_bot_response": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render():
    _init_state()
    st.title("📋 Direct Evaluation")
    st.caption("Paste any user message and chatbot response. The LLM judge scores the response across all dimensions.")

    _, _, judge_key = get_judge()
    if not judge_key:
        no_judge_warning()

    # Radio-based tab nav — allows programmatic reset to Evaluate on page entry
    if st.session_state.get("de_reset_tab"):
        st.session_state["de_active_tab"] = "Evaluate"
        st.session_state["de_reset_tab"] = False

    active_tab = st.radio(
        "de_nav",
        ["Evaluate", "Compare", "History"],
        index=["Evaluate", "Compare", "History"].index(st.session_state.get("de_active_tab", "Evaluate")),
        horizontal=True,
        label_visibility="collapsed",
        key="de_tab_radio",
    )
    st.session_state["de_active_tab"] = active_tab

    # ── TAB 1: Evaluate ───────────────────────────────────────────────────
    if active_tab == "Evaluate":
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.markdown("#### System Prompt *(optional)*")
            st.caption("Paste the system prompt the chatbot was using. If left blank, Instruction Following will be scored against general best practices.")
            system_prompt_input = st.text_area(
                "de_system_prompt_input",
                value=st.session_state.de_system_prompt,
                placeholder="Paste system prompt here, or leave blank...",
                height=140,
                label_visibility="collapsed",
            )
            st.session_state.de_system_prompt = system_prompt_input

            st.markdown("#### User Message")
            user_msg = st.text_area(
                "de_user_msg_input",
                value=st.session_state.de_user_msg,
                placeholder="Paste the user message here...",
                height=100,
                label_visibility="collapsed",
            )
            st.session_state.de_user_msg = user_msg

        with col_right:
            st.markdown("#### Chatbot Response")
            st.caption("Paste the chatbot response you want to evaluate.")
            bot_response = st.text_area(
                "de_bot_response_input",
                value=st.session_state.de_bot_response,
                placeholder="Paste the chatbot's response here...",
                height=260,
                label_visibility="collapsed",
            )
            st.session_state.de_bot_response = bot_response

        run_disabled = not judge_key or not user_msg.strip() or not bot_response.strip()
        run_btn = st.button(
            "Score with Judge",
            type="primary",
            disabled=run_disabled,
            use_container_width=True,
            key="de_run_btn",
        )

        if not judge_key:
            st.caption("Add an API key in **API Settings** to enable scoring.")
        elif not user_msg.strip() or not bot_response.strip():
            st.caption("Paste both a user message and a chatbot response to score.")

        if run_btn:
            with st.spinner("Judge is scoring the response..."):
                result = run_judge(
                    user_message=user_msg,
                    bot_response=bot_response,
                    mode="direct",
                    system_prompt_ctx=system_prompt_input,
                )
            if not result:
                st.error("Judge returned an empty result. Check your API key and try again.")
            else:
                _scores = result.get("scores", {})
                # Layer 2 safety: if no system prompt was provided, force
                # instruction_following to None regardless of what the judge returned
                if not system_prompt_input.strip():
                    _scores["instruction_following"] = None
                st.session_state.de_last = {
                    "user_message": user_msg,
                    "bot_response": bot_response,
                    "system_prompt": system_prompt_input,
                    "scores": _scores,
                    "reasons": result.get("reasons", {}),
                    "suggestion": result.get("suggestion", ""),
                    "non_sequitur_warning": result.get("non_sequitur_warning", "no"),
                }
                save_version({
                    "user_message": user_msg,
                    "system_prompt": system_prompt_input,
                    "scores": _scores,
                    "extra": {
                        "bot_response": bot_response,
                        "reasons": result.get("reasons", {}),
                        "suggestion": result.get("suggestion", ""),
                        "non_sequitur_warning": result.get("non_sequitur_warning", "no"),
                    },
                }, "de_history")
                st.rerun()

        last = st.session_state.get("de_last")
        if not last:
            return

        scores = last.get("scores", {})
        reasons = last.get("reasons", {})
        suggestion = last.get("suggestion", "")
        avg = avg_score(scores)
        color = score_color(avg)

        st.divider()
        st.markdown(
            f"<div style='background:#fafafa;border-radius:10px;padding:14px 20px;"
            f"border-left:5px solid {color};margin-bottom:16px;'>"
            f"<span style='font-size:15px;font-weight:600;'>Scores - Overall "
            f"<span style='color:{color}'>{avg}/5</span></span></div>",
            unsafe_allow_html=True,
        )

        col_scores, col_radar = st.columns([1, 1], gap="large")
        with col_scores:
            st.markdown("#### Dimension Scores")
            for dim in DIRECT_DIMENSIONS:
                score_val = scores.get(dim)
                if score_val is not None:
                    render_score_card(dim, score_val, reasons.get(dim, ""))
        with col_radar:
            st.markdown("#### Radar")
            radar_scores = {d: scores[d] for d in DIRECT_DIMENSIONS if d in scores}
            if radar_scores:
                render_radar(radar_scores, color="#c4705a")
            if last.get("non_sequitur_warning", "no") == "yes":
                render_non_sequitur_warning()
            if suggestion:
                st.markdown("#### Judge's Suggestions")
                render_suggestion_box(suggestion)

        st.markdown("#### Conversation")
        col_msg, col_resp = st.columns([1, 1], gap="large")
        with col_msg:
            st.markdown("**User Message**")
            st.markdown(f'<div class="bubble-user" style="font-size:13px">{last["user_message"]}</div>', unsafe_allow_html=True)
        with col_resp:
            st.markdown("**Chatbot Response**")
            st.markdown(f'<div class="bubble-bot" style="font-size:13px">{last["bot_response"]}</div>', unsafe_allow_html=True)

    elif active_tab == "Compare":
        history = get_history("de_history")
        if len(history) < 2:
            st.info("Run at least 2 evaluations to compare.")
            return

        def eval_label(v):
            return f"Eval v{v['id']} - {v['user_message'][:40]}..."

        eval_labels = [eval_label(v) for v in history]

        st.markdown("#### Select evaluations to compare (up to 4)")
        selected_labels = st.multiselect(
            "de_cmp_select",
            eval_labels,
            default=eval_labels[-2:] if len(eval_labels) >= 2 else eval_labels,
            max_selections=4,
            label_visibility="collapsed",
            placeholder="Choose up to 4 evaluations...",
            key="de_cmp_multiselect",
        )

        if len(selected_labels) < 2:
            st.caption("Select at least 2 evaluations to compare.")
            return

        selected_evals = [history[eval_labels.index(lbl)] for lbl in selected_labels]
        sel_colors = VERSION_COLORS[:len(selected_evals)]

        # Radar with checkboxes
        if any(v.get("scores") for v in selected_evals):
            st.markdown("### Score Comparison")

            filter_cols = st.columns(len(selected_evals))
            visible_labels = []
            for col, lbl, color in zip(filter_cols, selected_labels, sel_colors):
                if col.checkbox(lbl[:30], value=True, key="de_radar_" + lbl):
                    visible_labels.append(lbl)

            visible_evals = [v for v, lbl in zip(selected_evals, selected_labels) if lbl in visible_labels]
            visible_colors = [sel_colors[i] for i, lbl in enumerate(selected_labels) if lbl in visible_labels]

            if visible_evals:
                normalised = [
                    {"label": lbl[:30], "scores": {d: v["scores"].get(d, 0) for d in DIRECT_DIMENSIONS}}
                    for v, lbl in zip(visible_evals, [l for l in selected_labels if l in visible_labels])
                ]
                render_radar_overlay(normalised, visible_colors)

            # Score table
            header_cols = st.columns([2] + [1] * len(selected_evals))
            header_cols[0].markdown("**Dimension**")
            for i, lbl in enumerate(selected_labels):
                header_cols[i + 1].markdown("**" + lbl[:20] + "**")
            for dim in DIRECT_DIMENSIONS:
                row = st.columns([2] + [1] * len(selected_evals))
                row[0].markdown(dim_label(dim))
                for i, v in enumerate(selected_evals):
                    val = v["scores"].get(dim, None)
                    # None means dimension was not applicable (e.g. instruction_following with no system prompt)
                    if val is None:
                        row[i + 1].markdown("<span style='color:#bbb;font-weight:600'>—</span>", unsafe_allow_html=True)
                    elif isinstance(val, (int, float)):
                        color = score_color(val)
                        row[i + 1].markdown("<span style='color:" + color + ";font-weight:600'>" + str(val) + "/5</span>", unsafe_allow_html=True)
                    else:
                        row[i + 1].markdown("<span style='color:#999;font-weight:600'>" + str(val) + "</span>", unsafe_allow_html=True)

        # Responses side by side
        st.markdown("### Responses")
        resp_cols = st.columns(len(selected_evals))
        for col, v, lbl in zip(resp_cols, selected_evals, selected_labels):
            with col:
                st.markdown(f"**{lbl[:35]}**")
                extra = v.get("extra", {})
                bot_resp = extra.get("bot_response", "")
                if bot_resp:
                    st.markdown('<div class="bubble-bot" style="font-size:13px">' + bot_resp + "</div>", unsafe_allow_html=True)
                suggestion = extra.get("suggestion", "")
                if suggestion:
                    render_suggestion_box(suggestion)

    elif active_tab == "History":
        history = get_history("de_history")
        if not history:
            st.info("No evaluations yet.")
            return

        for v in reversed(history):
            avg = avg_score(v.get("scores", {}))
            color = score_color(avg)
            label = f"Eval v{v['id']} - Overall {avg}/5 | {v['user_message'][:50]}"
            with st.expander(label):
                st.markdown("**User Message:** " + v["user_message"])
                extra = v.get("extra", {})
                bot_resp = extra.get("bot_response", "")
                if bot_resp:
                    st.markdown('<div class="bubble-bot" style="font-size:13px">' + bot_resp + "</div>", unsafe_allow_html=True)
                for dim in DIRECT_DIMENSIONS:
                    val = v["scores"].get(dim)
                    if val is not None:
                        c = score_color(val)
                        st.markdown(f"**{dim_label(dim)}:** <span style='color:{c};font-weight:600'>{val}/5</span>", unsafe_allow_html=True)
                suggestion = extra.get("suggestion", "")
                if suggestion:
                    render_suggestion_box(suggestion)
