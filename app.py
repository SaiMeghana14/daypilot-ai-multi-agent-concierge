import os
import json
import time
import traceback
from datetime import datetime
from typing import Dict, Optional, Any, List
from pathlib import Path
from ui_style import render_header, theme_toggle, show_badges, icons, animated_text, inject_css

import streamlit as st

# Try to import google.generativeai but don't crash if unavailable
try:
    import google.generativeai as genai  # type: ignore
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# ---------- Config & Secrets ----------
st.set_page_config(page_title="DayPilot AI — Enhanced Multi-Agent Concierge", layout="wide")

# Apply UI styling (CSS + Header + Theme)
inject_css()
render_header()

# Theme toggle must come early
mode = theme_toggle()

# Show badges (profile, model, mode)
show_badges(session_memory.get("profile"), PREFERRED_MODEL, FORCE_OFFLINE)

SECRETS = st.secrets if hasattr(st, "secrets") else {}
API_KEY = SECRETS.get("GOOGLE_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
PREFERRED_MODEL = SECRETS.get("LLM_MODEL", "gemini-pro") or os.getenv("LLM_MODEL", "gemini-pro")
FORCE_OFFLINE = str(SECRETS.get("OFFLINE_MODE", "false")).lower() == "true" or os.getenv("OFFLINE_MODE", "false").lower() == "true"

# paths
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
LONG_TERM_MEM_FILE = DATA_DIR / "long_term_memory.json"

# ---------- Utilities ----------
def now_ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def safe_log(log_list: list, stage: str, msg: str):
    """Append a timestamped message to a log list (Session-state friendly)."""
    log_list.append(f"[{now_ts()}] [{stage}] {msg}")

def save_long_term_memory(store: dict):
    try:
        with open(LONG_TERM_MEM_FILE, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)
    except Exception as e:
        print("Failed to save long-term memory:", e)

def load_long_term_memory() -> dict:
    if LONG_TERM_MEM_FILE.exists():
        try:
            with open(LONG_TERM_MEM_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

# ---------- Memory (Session + Long-Term) ----------
class SessionMemory:
    def __init__(self):
        self.store = {
            "preferred_wakeup": "7:00 AM",
            "work_style": "Pomodoro",
            "planning_format": "list",
            "last_plan": None,
            "profile": "Student",
        }
    def get(self, key: str, default=None):
        return self.store.get(key, default)
    def set(self, key: str, value):
        self.store[key] = value
    def dump(self):
        return dict(self.store)

session_memory = SessionMemory()
long_term_memory = load_long_term_memory()  # dict persisted to disk

# ---------- Observability metrics ----------
metrics = {
    "runs": 0,
    "avg_cycles": 0.0,
    "last_run_duration_s": 0.0,
    "last_score": None,
}

# ---------- Simple A2A message format ----------
def make_a2a(sender: str, receiver: str, payload: Any) -> dict:
    return {
        "timestamp": now_ts(),
        "sender": sender,
        "receiver": receiver,
        "payload": payload
    }

# ---------- LLM Engine (Hybrid) ----------
class LLMEngine:
    def __init__(self, api_key: str, preferred_model: str, force_offline: bool = False):
        self.api_key = api_key
        self.preferred_model = preferred_model
        self.force_offline = force_offline or (not GENAI_AVAILABLE) or (not api_key)
        self.active_model = None
        if not self.force_offline:
            try:
                genai.configure(api_key=self.api_key)
                self.active_model = self.preferred_model
            except Exception:
                self.force_offline = True

    def generate(self, prompt: str, tools: Optional[list] = None) -> str:
        if self.force_offline:
            return self.fake_llm(prompt)
        try:
            # Primary attempt
            model_name = self.active_model or self.preferred_model
            response = genai.GenerativeModel(model_name).generate_content(prompt)
            if hasattr(response, "text"):
                return response.text
            return str(response)
        except Exception:
            # fallback
            try:
                fallback = "models/text-bison-001"
                response = genai.GenerativeModel(fallback).generate_content(prompt)
                if hasattr(response, "text"):
                    self.active_model = fallback
                    return response.text
                return str(response)
            except Exception:
                self.force_offline = True
                return self.fake_llm(prompt)

    # A slightly more capable fake LLM to simulate responses across agents
    @staticmethod
    def fake_llm(prompt: str) -> str:
        text = prompt.lower()
        if "planner agent" in text or "break the user's" in text or "plan" in text[:200]:
            return (
                "Subtasks:\n"
                "- Split available time into focused blocks + breaks\n"
                "- Prioritize topics by difficulty\n"
                "- Suggest resources if needed\n"
                "Needs_Search: False\nNeeds_Code: False\n"
            )
        if "execution agent" in text or "search for" in text:
            return (
                "Execution Results:\n"
                "- Suggested AI resources: short tutorial, sample notebook\n"
                "- Suggested IoT resources: ESP32 basic guide, sensor wiring\n"
                "- Productivity suggestion: 50-min focus + 10-min break\n"
            )
        if "summarizer agent" in text or "final plan" in text:
            return (
                "FINAL PLAN (simulated):\n"
                "1) AI — 90 mins: theory + small coding example\n"
                "2) Break — 10 mins\n"
                "3) IoT — 90 mins: hands-on sensor tutorial\n"
                "4) Break — 10 mins\n"
                "5) AI — 60 mins: training pipeline overview\n"
                "6) Review — 20 mins\n"
            )
        if "reflection agent" in text or "improve" in text:
            return "Reflection: remove less essential parts, add checkpoints and 'what to do if stuck' tips."
        if "compact context" in text or "compaction" in text:
            return "CompactContext: user likes pomodoro, prefers hands-on labs, prefers 6-hour sessions."
        return "SIMULATED_LLM: no specific match for prompt."

llm_engine = LLMEngine(api_key=API_KEY, preferred_model=PREFERRED_MODEL, force_offline=FORCE_OFFLINE)

# ---------- Tools (Search & Code) - safe simulators; implement real ones if desired ----------
def search_tool(query: str, logs: list) -> str:
    safe_log(logs, "SearchTool", f"Querying: {query}")
    if llm_engine.force_offline:
        return f"FAKE SEARCH: top resources for '{query}': quick tutorial, official docs, example notebook."
    # if LLM is live, we can call LLM to behave like a search responder
    prompt = f"Execution Agent: act as a web search summarizer for: {query}\nReturn short resource list."
    return llm_engine.generate(prompt)

def code_tool(expression: str, logs: list) -> str:
    safe_log(logs, "CodeTool", f"Eval request: {expression}")
    # safe limited evaluation: arithmetic and simple parsing
    try:
        # literal_eval for safety
        import ast
        val = ast.literal_eval(expression)
        return f"CodeTool result: {val}"
    except Exception:
        # last resort: do not execute arbitrary code; return simulated result
        return "CodeTool: unable to evaluate expression safely (demo mode)."

# ---------- Context Compaction Agent ----------
def context_compaction(history: List[str], logs: list) -> str:
    safe_log(logs, "ContextCompaction", "Activated")
    # basic compaction: keep only last N entries and summarize via LLM or simulated method
    relevant = "\n".join(history[-6:])  # keep last 6 turns
    prompt = f"Context Compaction\nCompact these lines into compact facts:\n{relevant}"
    compacted = llm_engine.generate(prompt)
    safe_log(logs, "ContextCompaction", "Completed")
    return compacted

# ---------- Tool Router Agent ----------
def tool_router(plan_text: str, logs: list) -> Dict[str, bool]:
    safe_log(logs, "ToolRouter", "Deciding tools to use")
    plan_lower = plan_text.lower()
    uses = {
        "search": ("needs_search: true" in plan_lower) or ("needs_search: yes" in plan_lower),
        "code": ("needs_code: true" in plan_lower) or ("needs_code: yes" in plan_lower)
    }
    # heuristics: if plan mentions "resource" or "tutorial" => search
    if ("resource" in plan_lower or "tutorial" in plan_lower) and not uses["search"]:
        uses["search"] = True
    safe_log(logs, "ToolRouter", f"Decided tools: {uses}")
    return uses

# ---------- Agent Evaluation (simple scoring) ----------
def evaluate_plan(plan_text: str, exec_output: str, logs: list) -> dict:
    safe_log(logs, "Evaluator", "Scoring plan")
    # heuristics for scoring: length, presence of steps, tool usage
    score = 0
    # clarity: presence of numeric steps or bullets
    if any(tok in plan_text.lower() for tok in ["1)", "-", "step"]):
        score += 40
    # relevance: mentions topics from user input (estimated via length here)
    if len(plan_text) < 800:
        score += 20
    # execution usefulness:
    if "Search results:" in exec_output or "Suggested" in exec_output:
        score += 30
    else:
        score += 10
    # bonus for memory persistence usage
    score += 0
    final = min(score, 100)
    safe_log(logs, "Evaluator", f"Score computed: {final}")
    return {"score": final, "components": {"clarity": 40, "conciseness": 20, "usefulness": 40}}

# ---------- Agents: Planner, Executor, Summarizer, Reflection, Loop Orchestrator ----------
def planner_agent(user_input: str, logs: list, compacted_context: Optional[str] = None) -> dict:
    safe_log(logs, "Planner", "Activated")
    prompt = f"""Planner Agent
You are Planner Agent. User input:
{user_input}
Context facts (compacted): {compacted_context}
Task: Break user's goal into subtasks, durations (minutes), priorities. Indicate Needs_Search and Needs_Code as True/False.
Return structured plan as plain text with clear subtasks."""
    out = llm_engine.generate(prompt)
    msg = make_a2a("PlannerAgent", "ExecutorAgent", {"plan_text": out})
    safe_log(logs, "Planner", "Completed")
    return {"a2a": msg, "plan_text": out}

def executor_agent(plan_text: str, logs: list) -> dict:
    safe_log(logs, "Executor", "Activated")
    router = tool_router(plan_text, logs)
    results = []
    if router.get("search"):
        res = search_tool("best resources for plan: " + plan_text[:200], logs)
        results.append("Search results:\n" + res)
    if router.get("code"):
        # simple simulated code execution e.g., time parsing
        res = code_tool("2+2", logs)
        results.append(res)
    if not results:
        results.append("No external tools required. Using internal curated knowledge.")
    out = "\n".join(results)
    msg = make_a2a("ExecutorAgent", "SummarizerAgent", {"exec_output": out})
    safe_log(logs, "Executor", "Completed")
    return {"a2a": msg, "exec_output": out}

def summarizer_agent(user_input: str, plan_text: str, exec_output: str, logs: list) -> dict:
    safe_log(logs, "Summarizer", "Activated")
    prompt = f"""Summarizer Agent
Combine user input, plan, and execution results into a short actionable schedule tailored to user's style: {session_memory.get('work_style')}
User: {user_input}
Plan: {plan_text}
ExecResults: {exec_output}
Return a final plan text."""
    out = llm_engine.generate(prompt)
    msg = make_a2a("SummarizerAgent", "ReflectionAgent", {"summary": out})
    session_memory.set("last_plan", out)
    safe_log(logs, "Summarizer", "Completed and saved to session memory")
    return {"a2a": msg, "summary": out}

def reflection_agent(summary_text: str, logs: list) -> dict:
    safe_log(logs, "Reflection", "Activated")
    prompt = f"""Reflection Agent
Analyze the plan and summary. Propose at most 3 improvements, add a contingency step 'If stuck' and short checkpoints.
Summary: {summary_text}"""
    out = llm_engine.generate(prompt)
    msg = make_a2a("ReflectionAgent", "User", {"reflection": out})
    # optionally update long-term memory: store improvements
    long_term_memory.setdefault("improvements", []).append({"timestamp": now_ts(), "text": out})
    save_long_term_memory(long_term_memory)
    safe_log(logs, "Reflection", "Completed and saved suggestions to long-term memory")
    return {"a2a": msg, "reflection": out}

# ---------- Loop Agent (autonomous multi-cycle refinement) ----------
def loop_orchestrator(user_input: str, cycles: int, logs: list) -> dict:
    safe_log(logs, "Loop", f"Starting loop with cycles={cycles}")
    start = time.time()
    history = []
    compact_ctx = None
    plan_text, exec_output, summary_text, final_reflection = "", "", "", ""
    for i in range(max(1, cycles)):
        safe_log(logs, "Loop", f"Cycle {i+1} begin")
        # compact context from history
        if history:
            compact_ctx = context_compaction(history, logs)
        planner_out = planner_agent(user_input, logs, compact_ctx)
        plan_text = planner_out["plan_text"]
        history.append(plan_text)
        exec_out = executor_agent(plan_text, logs)
        exec_output = exec_out["exec_output"]
        history.append(exec_output)
        summed = summarizer_agent(user_input, plan_text, exec_output, logs)
        summary_text = summed["summary"]
        history.append(summary_text)
        # small improvement step: reflection after each cycle (could be only after last)
        ref = reflection_agent(summary_text, logs)
        final_reflection = ref["reflection"]
        history.append(final_reflection)
        safe_log(logs, "Loop", f"Cycle {i+1} end")
        # optionally adapt user_input or session_memory based on reflection — minimal here
        # small delay to simulate processing
        time.sleep(0.2)
    duration = time.time() - start
    metrics["runs"] += 1
    metrics["last_run_duration_s"] = duration
    metrics["avg_cycles"] = ((metrics.get("avg_cycles", 0.0) * (metrics["runs"] - 1)) + cycles) / metrics["runs"]
    safe_log(logs, "Loop", f"Completed {cycles} cycles in {duration:.2f}s")
    # Evaluate the final plan
    eval_res = evaluate_plan(summary_text, exec_output, logs)
    metrics["last_score"] = eval_res["score"]
    return {
        "plan": plan_text,
        "execution": exec_output,
        "summary": summary_text,
        "reflection": final_reflection,
        "evaluation": eval_res
    }

# ---------- Streamlit UI ----------
# layout: left inputs, center outputs, right observability
col1, col2, col3 = st.columns([3, 4, 3])

with col1:
    st.markdown("## 🧭 User Input")
    user_input = st.text_area("Describe your goal for today (brief):",
                              value="I have 6 hours today. Help me plan a productive study schedule for AI and IoT.")
    st.markdown("### Profile & Preferences")
    profile = st.selectbox("Profile", options=["Student", "Developer", "Researcher", "Designer"], index=0)
    wake = st.text_input("Preferred wakeup time", value=session_memory.get("preferred_wakeup"))
    style = st.selectbox("Work style", options=["Pomodoro", "Continuous", "Custom"], index=0)
    cycles = st.slider("Refinement cycles (loop agent)", min_value=1, max_value=4, value=2)
    run_btn = st.button("Run Agents")

    if st.button("Save Preferences"):
        session_memory.set("preferred_wakeup", wake)
        session_memory.set("work_style", style)
        session_memory.set("profile", profile)
        st.success("Preferences saved to session memory.")
    st.markdown("---")
    st.markdown("### Long-term memory controls")
    if st.button("Clear Long-Term Memory"):
        long_term_memory.clear()
        save_long_term_memory(long_term_memory)
        st.warning("Long-term memory cleared.")

with col2:
    st.markdown("## 🧠 Agent Outputs")
    planner_exp = st.expander(f"{icons['planner']} Planner Output", expanded=True)
    exec_exp = st.expander(f"{icons['executor']} Executor Output", expanded=False)
    summ_exp = st.expander(f"{icons['summarizer']} Summarizer Output", expanded=False)
    reflect_exp = st.expander(f"{icons['reflection']} Reflection Output", expanded=False)
    eval_exp = st.expander(f"{icons['evaluation']} Evaluation", expanded=False)

    with planner_exp:
        planner_text = st.empty()
    with exec_exp:
        exec_text = st.empty()
    with summ_exp:
        summ_text = st.empty()
    with reflect_exp:
        reflect_text = st.empty()
    with eval_exp:
        eval_text = st.empty()

with col3:
    st.markdown("## 📊 Observability & Memory")
    logs_box = st.empty()
    st.subheader("Session Memory")
    mem_box = st.empty()
    st.subheader("Long-Term Memory Snapshot")
    ltm_box = st.empty()
    st.subheader("Metrics")
    metrics_box = st.empty()
    st.markdown("---")
    st.caption("A2A messaging, loop cycles, and evaluations are visible in the logs.")

# prepare session logs
if "dp_logs" not in st.session_state:
    st.session_state.dp_logs = []
if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = None

# Run pipeline when clicked
if run_btn:
    st.session_state.last_run_time = now_ts()
    st.session_state.dp_logs.clear()
    try:
        # store preferences
        session_memory.set("preferred_wakeup", wake)
        session_memory.set("work_style", style)
        session_memory.set("profile", profile)

        safe_log(st.session_state.dp_logs, "Main", f"Starting pipeline for profile={profile}, cycles={cycles}")
        result = loop_orchestrator(user_input, cycles, st.session_state.dp_logs)

        # display outputs
        animated_text(result["plan"], planner_text)
        animated_text(result["execution"], exec_text)
        animated_text(result["summary"], summ_text)
        animated_text(result["reflection"], reflect_text)
        eval_text.markdown(f"**Evaluation score:** {result['evaluation']['score']}\n\nComponents:\n```\n{json.dumps(result['evaluation']['components'], indent=2)}\n```")

        # update memory and LTM boxes
        mem_box.json(session_memory.dump())
        ltm_box.text(json.dumps(long_term_memory, indent=2)[:2000])  # limit displayed chars
        metrics_box.json(metrics)

    except Exception as ex:
        safe_log(st.session_state.dp_logs, "Main", f"Error: {ex}")
        traceback.print_exc()
        st.error("Agent pipeline failed. See logs.")
    # always update logs
    logs_box.text_area("Agent logs (newest last)", value="\n".join(st.session_state.dp_logs), height=400)

# display memory & logs when not running
if not run_btn:
    mem_box.json(session_memory.dump())
    ltm_box.text(json.dumps(long_term_memory, indent=2)[:2000])
    metrics_box.json(metrics)
    logs_box.text_area("Agent logs (newest last)", value="\n".join(st.session_state.dp_logs), height=400)

# Display small "how to demo" & notes
st.markdown("---")
st.markdown("### Demo tips")
st.markdown(
    "- Save preferences and run a few cycles (2-3) to show loop behavior.\n"
    "- Open expanders to show each agent output.\n"
    "- Show logs and A2A messages to demonstrate agent orchestration.\n"
    "- Toggle `OFFLINE_MODE` in Streamlit secrets to demo fallback behavior."
)

st.caption("Developed for Google AI Agents Intensive — Enhanced capstone demo.")
