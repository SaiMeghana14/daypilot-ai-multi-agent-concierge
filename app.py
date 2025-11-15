import os
import time
import traceback
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime
from typing import Dict, Optional

# Try to import google.generativeai but do not crash if unavailable
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# ---------- Load secrets ----------
API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
PREFERRED_MODEL = st.secrets.get("LLM_MODEL", "gemini-pro")
FORCE_OFFLINE = str(st.secrets.get("OFFLINE_MODE", "false")).lower() == "true"

# ---------- App config ----------
st.set_page_config(page_title="DayPilot AI — Multi-Agent Concierge", layout="wide")

# ---------- Utilities ----------
def now_ts():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def safe_log(log_list: list, stage: str, msg: str):
    log_list.append(f"[{now_ts()}] [{stage}] {msg}")

# ---------- Memory (InMemorySessionService simulation) ----------
class SessionMemory:
    def __init__(self):
        self.store = {
            "preferred_wakeup": "7:00 AM",
            "work_style": "Pomodoro",
            "planning_format": "list",
            "last_plan": None,
        }

    def get(self, key, default=None):
        return self.store.get(key, default)

    def set(self, key, value):
        self.store[key] = value

    def dump(self):
        return dict(self.store)

session_memory = SessionMemory()

# ---------- LLM Wrapper (Hybrid behavior) ----------
class LLMEngine:
    def __init__(self, api_key: str, preferred_model: str, force_offline: bool = False):
        self.api_key = api_key
        self.preferred_model = preferred_model
        self.force_offline = force_offline or (not GENAI_AVAILABLE) or (not api_key)
        self.active_model = None
        if not self.force_offline:
            # configure genai
            try:
                genai.configure(api_key=self.api_key)
                # try to use preferred_model; may raise during generate if unsupported
                self.active_model = self.preferred_model
            except Exception as e:
                # fallback to offline
                self.force_offline = True

    def generate(self, prompt: str, tools: Optional[list] = None) -> str:
        """
        Try to use the configured model. If model call fails, attempt fallback models.
        If everything fails, fall back to fake LLM (deterministic).
        """
        if self.force_offline:
            return self.fake_llm(prompt)

        # first attempt - preferred model
        try:
            model_name = self.active_model or self.preferred_model
            # For older SDKs, use "models/text-bison-001" or similar naming; wrap call
            # We attempt to call generate_content and handle exceptions
            response = genai.GenerativeModel(model_name).generate_content(prompt)
            if hasattr(response, "text"):
                return response.text
            # sometimes response comes with .result[0].content or similar — fallback
            return str(response)
        except Exception as e:
            # try a known fallback model
            try:
                fallback = "models/text-bison-001"
                response = genai.GenerativeModel(fallback).generate_content(prompt)
                if hasattr(response, "text"):
                    self.active_model = fallback
                    return response.text
                return str(response)
            except Exception as e2:
                # ultimate fallback: offline
                self.force_offline = True
                return self.fake_llm(prompt)

    @staticmethod
    def fake_llm(prompt: str) -> str:
        # Simple deterministic simulation — expand as needed for demo.
        # We'll look for agent-role hints in prompt to return structured outputs.
        text = prompt.lower()
        if "planner agent" in text or "break the user's" in text or "plan" in text[:200]:
            return (
                "Subtasks:\n"
                "- Split 6 hours into three blocks (90, 90, 60) + short breaks\n"
                "- Prioritize AI fundamentals (theory + small examples) and hands-on IoT labs\n"
                "- Use Pomodoro cycles (50/10) due to user's preference\n"
                "Needs_Search: No\nNeeds_Code: No\n"
            )
        if "execution agent" in text or "execute" in text:
            return (
                "Execution Results:\n"
                "- AI Modules suggested: ML basics, neural networks, training loop resources\n"
                "- IoT Modules suggested: sensors, microcontroller (ESP32), basic cloud integration\n"
                "- Productivity recommendation: 50-min focus blocks, 10-min breaks\n"
            )
        if "summarizer" in text or "final" in text:
            return (
                "FINAL PLAN:\n"
                "1) AI — 90 mins: neural networks (theory + 1 worked example)\n"
                "2) Break — 10 mins\n"
                "3) IoT — 90 mins: sensors + ESP32 hands-on lab\n"
                "4) Break — 10 mins\n"
                "5) AI — 90 mins: training pipelines + code walkthrough\n"
                "6) IoT — 60 mins: cloud basics + deployment notes\n"
                "7) Review — 20 mins\n"
            )
        # default
        return "I'm running in offline/simulated mode. No live LLM available."

# Initialize engine
llm_engine = LLMEngine(api_key=API_KEY, preferred_model=PREFERRED_MODEL, force_offline=FORCE_OFFLINE)

# ---------- Agent definitions ----------
def planner_agent(user_input: str, logs: list) -> str:
    safe_log(logs, "Planner", "Activated")
    prompt = f"""Planner Agent
You are the Planner Agent. Read user input, break tasks into subtasks, and decide whether to use search or code execution.
USER: {user_input}
Return a clear structured plan and whether tools are needed."""
    safe_log(logs, "Planner", "Calling LLM")
    out = llm_engine.generate(prompt)
    safe_log(logs, "Planner", "Completed")
    return out

def executor_agent(plan_text: str, logs: list) -> str:
    safe_log(logs, "Executor", "Activated")
    # simple heuristic: look for 'needs_search' or 'needs_code' tokens
    needs_search = any(tok in plan_text.lower() for tok in ["needs_search: yes", "needs_search: true"])
    needs_code  = any(tok in plan_text.lower() for tok in ["needs_code: yes", "needs_code: true"])
    results = []
    if needs_search:
        safe_log(logs, "Executor", "Invoking search tool (via LLM tool or simulated)")
        # If LLM is live and supports tools, it would use google_search; we simulate a search prompt
        search_prompt = f"Execution Agent\nSearch for high-quality learning resources and quick tips for the plan:\nPlan:\n{plan_text}"
        search_out = llm_engine.generate(search_prompt)
        results.append(f"Search results:\n{search_out}")
    if needs_code:
        safe_log(logs, "Executor", "Invoking code tool (simulated)")
        # simulate code execution result
        results.append("CodeTool result: (simulated) Sample calculation or time parsing.")
    if not (needs_search or needs_code):
        results.append("No external tools required. Using curated internal knowledge.")
    safe_log(logs, "Executor", "Completed")
    return "\n".join(results)

def summarizer_agent(user_input: str, plan_text: str, exec_output: str, logs: list) -> str:
    safe_log(logs, "Summarizer", "Activated")
    prompt = f"""Summarizer Agent
You are the Summarizer Agent. Combine the USER input, the PLAN, and the EXECUTION OUTPUT into a final user-facing schedule. Keep it short, actionable, and in the user's preferred style ({session_memory.get('work_style')}).
USER: {user_input}
PLAN: {plan_text}
EXECUTION_RESULTS: {exec_output}
Return a formatted final plan."""
    safe_log(logs, "Summarizer", "Calling LLM")
    out = llm_engine.generate(prompt)
    # persist last plan
    session_memory.set("last_plan", out)
    safe_log(logs, "Summarizer", "Completed and saved to memory")
    return out

# ---------- Streamlit UI layout ----------
st.title("🛟 DayPilot AI — Multi-Agent Personal Workflow Concierge")
st.markdown("Hybrid mode: **Gemini Pro preferred**; fallback to PaLM or offline simulator. (Set keys in `.env`)")

# left: inputs and controls
col1, col2, col3 = st.columns([3, 3, 4])

with col1:
    st.header("User Input")
    user_input = st.text_area("Describe your goal for today (brief):",
                              value="I have 6 hours today. Help me plan a productive study schedule for AI and IoT.")
    run_btn = st.button("Run Agents")
    st.markdown("---")
    st.subheader("Preferences (session memory)")
    wake = st.text_input("Preferred wakeup time", value=session_memory.get("preferred_wakeup"))
    style = st.selectbox("Work style", options=["Pomodoro", "Continuous", "Custom"], index=0)
    if st.button("Save Preferences"):
        session_memory.set("preferred_wakeup", wake)
        session_memory.set("work_style", style)
        st.success("Preferences saved to session memory.")

with col2:
    st.header("Agent Outputs")
    planner_out = st.empty()
    executor_out = st.empty()
    summarizer_out = st.empty()

with col3:
    st.header("Observability & Memory")
    log_box = st.empty()
    st.subheader("Session Memory")
    mem_box = st.empty()
    st.subheader("Environment")
    env_info = {
        "GENAI_AVAILABLE": GENAI_AVAILABLE,
        "API_KEY_PROVIDED": bool(API_KEY),
        "PREFERRED_MODEL": PREFERRED_MODEL,
        "FORCE_OFFLINE": FORCE_OFFLINE,
    }
    st.json(env_info)

# Log list stored in session state for interactivity
if "dp_logs" not in st.session_state:
    st.session_state.dp_logs = []
if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = None

# Run the pipeline
if run_btn:
    st.session_state.last_run_time = now_ts()
    st.session_state.dp_logs.clear()
    try:
        safe_log(st.session_state.dp_logs, "Main", "Starting workflow")
        p_out = planner_agent(user_input, st.session_state.dp_logs)
        planner_out.markdown(f"**Planner output:**\n```\n{p_out}\n```")

        e_out = executor_agent(p_out, st.session_state.dp_logs)
        executor_out.markdown(f"**Executor output:**\n```\n{e_out}\n```")

        s_out = summarizer_agent(user_input, p_out, e_out, st.session_state.dp_logs)
        summarizer_out.markdown(f"**Final Plan:**\n```\n{s_out}\n```")

        # show memory and logs
        mem_box.text(session_memory.dump())

    except Exception as ex:
        safe_log(st.session_state.dp_logs, "Main", f"Error: {ex}")
        summarizer_out.error("Agent run failed. See logs.")
        traceback.print_exc()

    # always update logs panel
    log_box.text_area("Agent logs (newest last)", value="\n".join(st.session_state.dp_logs), height=400)

# display memory and logs even when not running
if not run_btn:
    mem_box.text(session_memory.dump())
    log_box.text_area("Agent logs (newest last)", value="\n".join(st.session_state.dp_logs), height=400)

st.markdown("---")
st.caption("Developed for a Google AI Agents Intensive capstone. Hybrid Streamlit demo: live Gemini Pro (if configured) or offline fallback.")
