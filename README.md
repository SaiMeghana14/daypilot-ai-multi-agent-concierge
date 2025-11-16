# 🌟 DayPilot AI - Multi-Agent Personal Workflow Concierge #

**🚀 Overview**

DayPilot AI is an intelligent multi-agent system designed to help users plan their day with clarity, structure, and optimized workflow decisions.

Built as a capstone project for the Google AI Agents Intensive, it showcases real-world agent orchestration using multiple collaborating agents, tool routing, memory, multi-cycle refinement, and a production-quality UI.

DayPilot AI transforms a simple user goal like:

“I have 6 hours today. Help me study AI and IoT.”

into a fully optimized daily schedule using AI reasoning and looped improvements.

**🎯 Features**
💡 Multi-Agent Collaboration

Planner → Executor → Summarizer → Reflection loop

Automatic tool routing (search, code execution)

A2A structured messaging

Multi-cycle refinement with Loop Orchestrator

🧠 Memory Systems

Session Memory (current session preferences)

Long-Term Memory (persistent JSON storage)

Context compaction for efficient prompting

🛠️ Tools & Reasoning

Search Tool (live or simulated)

Code Execution Tool (safe AST evaluation)

Offline fallback with deterministic LLM simulation

🎨 Modern Streamlit UI

Gradient banner header

Theme mode (Light / Dark)

Badges for profile, model, mode, status

Animated typing effect

Lottie animations (customizable)

Step-by-step agent status animation

Profile avatars (🎓 💻 🎨 🔬)

Quick-start templates

Tabs: Home • Agents • Analytics • Memory

Download final plan as TXT

📊 Analytics & Observability

Logs panel

Agent sequence tracking

Execution runtime

Average cycles

Evaluation score

Long-term memory snapshot

**🧩 Architecture**
User Input
     │
     ▼
Planner Agent ──→ Tool Router ──→ Execution Agent
     │                               │
     ▼                               ▼
Summarizer Agent  ◀──────────────  Results
     │
     ▼
Reflection Agent
     │
     ▼
Loop Orchestrator (multi-cycle refinement)


Message passing follows a structured A2A protocol:

{
  "timestamp": "2025-01-01 UTC",
  "sender": "PlannerAgent",
  "receiver": "ExecutorAgent",
  "payload": {
    "plan_text": "..."
  }
}

**📦 Project Structure**
```
📁 project/
│── app.py
│── ui_style.py
│── requirements.txt
│── README.md
│──  lottie/
│     └──welcome.json 
```
**🛠️ Installation**
1️⃣ Clone repository
git clone https://github.com/username/daypilot-ai.git
cd daypilot-ai

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Add your API key in Streamlit secrets

Create:

.streamlit/secrets.toml


Add:

GOOGLE_API_KEY = "your-key-here"
LLM_MODEL = "gemini-pro"
OFFLINE_MODE = "false"

4️⃣ Run app
streamlit run app.py

**🧪 Usage**

1. Open the Home tab

2. Pick quick-start templates OR write your own input

3. Customize preferences (Wake time, profile, work style)

4. Go to Agents tab

5. Click Run Agents

6. Watch step animations as each agent completes

7. Download your final plan as .txt

**🧠 Example Query**
I have 6 hours today. Help me study AI and IoT with structured blocks.

Output (example)
1) AI — 90 mins (theory + coding example)
2) Break — 10 mins
3) IoT — 90 mins (ESP32 + sensor lab)
4) Break — 10 mins
5) AI — 60 mins (training pipeline)
6) Review — 20 mins

Reflection Improvements:
– Add checkpoints after each block
– Include “If stuck” instructions
– Prioritize hands-on tasks

**🔍 Core Concepts Demonstrated (Google Agents Requirements)**
- Concept	Status
- Multi-agent system	✅
- Parallel / Sequential agents	✅
- Loop agent (multi-cycle refinement)	✅
- LLM-powered agents	✅
- Tools (search, code execution)	✅
- Context compaction	✅
- Observability (logging, metrics)	✅
- Memory (session + long-term)	✅
- A2A protocol	✅
- Agent evaluation	✅
- Deployment-ready Streamlit app	✅
**📥 Downloadable Outputs**

1. Final schedule (txt)

2. Execution logs

3. Long-term memory snapshot

4. Evaluation score card

**🎯 Why DayPilot AI?**

This project demonstrates not just LLM prompting but true agentic reasoning:

- Multi-stage planning

- Adaptive improvements

- Personalized context retention

- Dynamic tool usage

- Smooth, user-friendly UI

It is both a technical demonstration and a practical daily tool.

**🛡️ Offline Mode**

If API key is missing or invalid:

- The system runs in deterministic “offline LLM simulation” mode

- Agents still collaborate and produce realistic plans

- Extremely helpful for demo reliability

**🔮 Future Enhancements**

- Calendar API integration

- WhatsApp / Email reminders

- Multi-day planning

- Adaptive habits-based scheduling

- Team planning mode

- MCP tool integration

**✨ Credits**

Built by Sai Meghana
For Google AI Agents Intensive – Capstone Project
Powered by: Gemini Pro, Streamlit, Python, Multi-Agent Architecture
