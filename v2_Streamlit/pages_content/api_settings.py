import streamlit as st
from utils.api_utils import get_all_keys, add_key, remove_key, all_available_models, get_judge

PROVIDER_ICONS = {"openai": "🟢", "anthropic": "🟠", "google": "🔵"}
PROVIDER_NAMES = {"openai": "OpenAI", "anthropic": "Anthropic", "google": "Google Gemini"}


def render():
    # ── Project intro ─────────────────────────────────────────────────────
    st.title("🧠 Mental Health Chatbot Safety Evaluator")
    st.markdown(
        "This tool helps developers test and improve system prompts for mental health AI chatbots. "
        "You write a system prompt, send a simulated user message, and an LLM judge automatically evaluates "
        "the chatbot's response across safety dimensions — therapeutic approach, crisis detection, harm avoidance. "
        "Iterate on your prompt, compare versions, and see exactly what changed."
    )

    st.markdown("#### Two features:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            '<div style="background:#f0f4ff;border:1px solid #c5d3f0;border-radius:10px;padding:16px 20px;margin-bottom:8px">'
            '<div style="font-size:15px;font-weight:600;margin-bottom:6px">🔬 Prompt Evaluation</div>'
            '<div style="font-size:13px;color:#444">Fix one model, vary the system prompt. Track how each edit affects scores and read the judge\'s analysis across versions.</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div style="background:#f0f4ff;border:1px solid #c5d3f0;border-radius:10px;padding:16px 20px;margin-bottom:8px">'
            '<div style="font-size:15px;font-weight:600;margin-bottom:6px">📊 Model Comparison</div>'
            '<div style="font-size:13px;color:#444">Fix one prompt, run it across multiple models. The judge writes a comparative verdict: which model performed best and what to fix.</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── API key input ─────────────────────────────────────────────────────
    st.subheader("API Settings")
    st.caption("Add at least one API key to get started. Keys are stored only in your browser session.")

    new_key = st.text_input(
        "Paste your API key",
        type="password",
        placeholder="sk-...  or  sk-ant-...  or  AIza...",
    )

    if st.button("Add Key", type="primary", use_container_width=True):
        if not new_key.strip():
            st.error("Please enter a key.")
        else:
            with st.spinner("Detecting which models this key can access..."):
                ok, result = add_key(new_key.strip(), label="")
            if ok:
                st.success("Key added.")
                st.rerun()
            else:
                st.error(result)

    st.divider()

    # ── Stored keys ───────────────────────────────────────────────────────
    keys = get_all_keys()

    if not keys:
        st.caption("No keys added yet.")
        return

    st.subheader(f"Your Keys ({len(keys)})")

    for i, entry in enumerate(keys):
        caps = entry["capabilities"]
        providers_found = list(caps.keys())

        col_info, col_remove = st.columns([5, 1])
        with col_info:
            masked = entry["key_str"][:8] + "..." + entry["key_str"][-4:]
            st.markdown(f"`{masked}`")
            badges = " &nbsp; ".join(
                f"{PROVIDER_ICONS.get(p, '⚪')} **{PROVIDER_NAMES.get(p, p)}** ({len(caps[p])} models)"
                for p in providers_found
            )
            st.markdown(badges, unsafe_allow_html=True)
            for p in providers_found:
                model_list = ", ".join(f"`{m}`" for m in caps[p])
                st.caption(f"{PROVIDER_NAMES.get(p, p)}: {model_list}")
        with col_remove:
            if st.button("Remove", key=f"remove_{i}", use_container_width=True):
                remove_key(i)
                st.rerun()

        st.divider()

    # ── Judge selector ────────────────────────────────────────────────────
    st.subheader("Judge Model")
    st.caption("The judge scores chatbot responses. Defaults to the best available model, but you can override it.")

    all_models = all_available_models()
    if not all_models:
        st.warning("No models available yet.")
        return

    default_provider, default_model, _ = get_judge()
    default_label = f"{PROVIDER_NAMES.get(default_provider, default_provider)} · {default_model}" if default_provider else None

    model_labels = [f"{PROVIDER_NAMES.get(m['provider'], m['provider'])} · {m['model']}" for m in all_models]
    default_idx = model_labels.index(default_label) if default_label in model_labels else 0

    selected_judge_label = st.selectbox(
        "Select judge model",
        model_labels,
        index=default_idx,
        label_visibility="collapsed",
    )
    selected_judge = all_models[model_labels.index(selected_judge_label)]

    if default_label and selected_judge_label == default_label:
        st.success(f"✅ Recommended judge: **{selected_judge_label}**")
    else:
        st.info(f"Judge set to: **{selected_judge_label}**")

    # Store override in session state so other pages can read it
    st.session_state["judge_override"] = {
        "provider": selected_judge["provider"],
        "model": selected_judge["model"],
        "key_str": selected_judge["key_str"],
    }
