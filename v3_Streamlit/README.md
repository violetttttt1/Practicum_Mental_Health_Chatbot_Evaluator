# Mental Health Chatbot Safety Evaluator — v3

**Live demo:** [https://mh-safety-evaluator-v3.streamlit.app/](https://mh-safety-evaluator-v3.streamlit.app/)

A developer tool for testing and iterating on AI chatbot system prompts in mental health contexts, using LLM-as-a-Judge evaluation.

## What's new in v3

- **Flexible API key system** — add any key without labeling it by provider. The tool auto-detects which models (OpenAI, Anthropic, Google Gemini) each key can access and shows only what's available to you.
- **Dynamic model list** — no hardcoded models. OpenAI models are fetched live from the API; Anthropic and Google use a curated list verified against your key.
- **Selectable judge model** — defaults to the best available model, but you can override it in API Settings.
- **Structured comparative verdict** — Model Comparison now produces a four-part analysis: overall summary, safety gap analysis, prompt fix recommendations, and per-model critique placed directly under each model's response.
- **Radar chart filter** — checkboxes to show/hide individual models on the overlay radar, so overlapping lines are no longer a problem.
- **Analysis-first layout** — written verdicts and judge analysis are the primary output; scores are secondary and collapsed by default.

## Features

### Prompt Evaluation
- Fix one model, vary the system prompt
- Each run saves a version labeled with model name and score (e.g. `v1 · gpt-4o · 3.2/5`)
- Compare any two versions: judge notes side by side, score delta metrics, full prompt diff, response comparison

### Model Comparison
- Fix one prompt, run it across up to 4 models simultaneously
- Optional therapy framework injection (CBT, ACT, MI, Person-Centered)
- Sample test messages including implicit crisis scenarios (no explicit keywords)
- Judge writes a structured comparative analysis with prompt fix recommendations

## Evaluation Dimensions

**Prompt Evaluation:** Therapeutic Approach, Monitoring & Risk, Harm/Non-reinforcement, Therapeutic Alliance, Instruction Following

**Model Comparison:** same as above, plus Implicit Risk Detection and Default Safety Behavior

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Go to **API Settings**, paste your API key, click **Add Key**. The tool will detect which models your key can access.

## Evaluation Framework

Based on CAPE-II (5 response-evaluable sections) + supplementary literature:
- arXiv 2501.15599 — CBT timing, toxic positivity
- PMC 12643404 — 9 therapeutic communication techniques
- arXiv 2509.24857 + 2510.27521 — crisis taxonomy, LLM-as-judge for crisis
- arXiv 2601.18630 — cognitive-affective gap
