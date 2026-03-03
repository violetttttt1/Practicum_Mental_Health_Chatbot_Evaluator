# Mental Health Chatbot Safety Evaluator

A developer tool for testing and iterating on AI chatbot system prompts in mental health contexts. Built with Streamlit and LLM-as-a-Judge evaluation.

## What it does

- Send simulated user messages to a chatbot and get automatic safety scores
- Evaluate responses across 5 dimensions: Safety, Empathy, Inclusivity, Non-Reinforcement, and Crisis Referral
- Edit the system prompt and re-run tests to see how scores change
- Compare any two versions side-by-side with radar charts

## Setup

### Step 1 — Add your OpenAI API key

Create a file called `.env` in the project folder and add:

```
OPENAI_API_KEY=sk-your-key-here
```

> **Note for Mac users:** Files starting with `.` are hidden by default in Finder. Press `Cmd + Shift + .` to toggle hidden files on/off.

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Project structure

```
├── app.py               # Main application
├── requirements.txt     # Python dependencies
├── .env                 # Your API key (do NOT commit this)
├── .env.example         # Template for .env
└── .gitignore           # Keeps .env out of GitHub
```
