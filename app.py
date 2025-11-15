import os
import json
import time
import traceback
from datetime import datetime
from typing import Dict, Optional, Any, List
from pathlib import Path

# import UI helpers you created
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
        # Load from long-term memory first
        ltm = load_long_term_memory()

        self.store = {
            "preferred_wakeup": ltm.get("preferred_wakeup", ""),
            "work_style": ltm.get("work_style", ""),
            "planning_format": ltm.get("planning_format", ""),
            "last_plan": ltm.get("last_plan", None),
            "profile": ltm.get("profile", ""),
        }

    def get(self, key, default=None):
        return self.store.get(key, default)

    def set(self, key, value):
        self.store[key] = value
        # save to long-term memory immediately
        ltm = load_long_term_memory()
        ltm[key] = value
        save_long_term_memory(ltm)

    def dump(self):
        return dict(self.store)

session_memory = SessionMemory()

# Apply UI styling and header (must be after session memory exists)
inject_css()
render_header()

# Theme toggle and badges
mode = theme_toggle()
show_badges(session_memory.get("profile"), PREFERRED_MODEL, FORCE_OFFLINE)

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
            model_name = self.active_model or self.preferred_model
            response = genai.GenerativeModel(model_name).generate_content(prompt)
            if hasattr(response, "text"):
                return response.text
            return str(response)
        except Exception:
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
    prompt = f"Execution Agent: act as a web search summarizer for: {query}\nReturn short resource list."
    return llm_engine.generate(prompt)

def code_tool(expression: str, logs: list) -> str:
    safe_log(logs, "CodeTool", f"Eval request: {expression}")
    try:
        import ast
        val = ast.literal_eval(expression)
        return f"CodeTool result: {val}"
    except Exception:
        return "CodeTool: unable to evaluate expression safely (demo mode)."

# ---------- Context Compaction Agent ----------
def context_compaction(history: List[str], logs: list) -> str:
    safe_log(logs, "ContextCompaction", "Activated")
    relevant = "\n".join(history[-6:])
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
    if ("resource" in plan_lower or "tutorial" in plan_lower) and not uses["search"]:
        uses["search"] = True
    safe_log(logs, "ToolRouter", f"Decided tools: {uses}")
    return uses

# ---------- Agent Evaluation (simple scoring) ----------
def evaluate_plan(plan_text: str, exec_output: str, logs: list) -> dict:
    safe_log(logs, "Evaluator", "Scoring plan")
    score = 0
    if any(tok in plan_text.lower() for tok in ["1)", "-", "step"]):
        score += 40
    if len(plan_text) < 800:
        score += 20
    if "Search results:" in exec_output or "Suggested" in exec_output:
        score += 30
    else:
        score += 10
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
    long_term_memory.setdefault("improvements", []).append({"timestamp": now_ts(), "text": out})
    save_long_term_memory(long_term_memory)
    safe_log(logs, "Reflection", "Completed and saved suggestions to long-term memory")
    return {"a2a": msg, "reflection": out}

def loop_orchestrator(user_input: str, cycles: int, logs: list) -> dict:
    safe_log(logs, "Loop", f"Starting loop with cycles={cycles}")
    start = time.time()
    history = []
    compact_ctx = None
    plan_text, exec_output, summary_text, final_reflection = "", "", "", ""
    for i in range(max(1, cycles)):
        safe_log(logs, "Loop", f"Cycle {i+1} begin")
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
        ref = reflection_agent(summary_text, logs)
        final_reflection = ref["reflection"]
        history.append(final_reflection)
        safe_log(logs, "Loop", f"Cycle {i+1} end")
        time.sleep(0.2)
    duration = time.time() - start
    metrics["runs"] += 1
    metrics["last_run_duration_s"] = duration
    metrics["avg_cycles"] = ((metrics.get("avg_cycles", 0.0) * (metrics["runs"] - 1)) + cycles) / metrics["runs"]
    safe_log(logs, "Loop", f"Completed {cycles} cycles in {duration:.2f}s")
    eval_res = evaluate_plan(summary_text, exec_output, logs)
    metrics["last_score"] = eval_res["score"]
    return {
        "plan": plan_text,
        "execution": exec_output,
        "summary": summary_text,
        "reflection": final_reflection,
        "evaluation": eval_res
    }

# ------------------ NEW UI: Onboarding & Tabs & Sidebar ------------------
# Onboarding
if "show_intro" not in st.session_state:
    st.session_state.show_intro = True

# Sidebar tips & controls
with st.sidebar:
    st.markdown("## 💡 Quick Tips")
    st.markdown("- Use templates for a fast start\n- Run 2–3 cycles for best results\n- Save preferences to personalize\n")
    if st.button("🔄 Reset App (clear session)"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
    st.markdown("---")
    st.markdown("## ⚙️ Quick Controls")
    if st.button("🎯 Example: Study 6h (AI + IoT)"):
        st.session_state.user_template = "I have 6 hours today. Help me plan a productive study schedule for AI and IoT."
    if st.button("💼 Example: Workday Focus (8h)"):
        st.session_state.user_template = "I have 8 hours. Help me plan a focused workday: deep work blocks, meetings, and breaks."

# Tabs layout
tabs = st.tabs(["🏠 Home", "🤖 Agents", "📊 Analytics", "🧠 Memory"])

# Home tab: onboarding + lottie + quick templates
with tabs[0]:
    if st.session_state.show_intro:
        st.markdown("## 👋 Welcome to DayPilot AI")
        st.markdown(
            "A multi-agent personal concierge that plans productive days. "
            "It demonstrates Planner, Executor, Summarizer, Reflection agents, loop refinement, tools, and memory."
        )
        st.markdown("### Quick-start templates")
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        if "user_template" not in st.session_state:
            st.session_state.user_template = ""
        with col_a:
            if st.button("📌 Study Mode"):
                st.session_state.user_template = "I have 6 hours today. Help me study AI and IoT with a structured plan."
        with col_b:
            if st.button("🏃 Productivity Sprint"):
                st.session_state.user_template = "I want to complete 3 tasks in 4 hours with Pomodoro style."
        with col_c:
            if st.button("🛠️ Project Work"):
                st.session_state.user_template = "I need a 5-hour schedule to develop a small IoT prototype and test it."
        with col_d:
            if st.button("👩🏻‍💻 Coding Bootcamp"):
                st.session_state.user_template = "Give me a 5-hour coding bootcamp plan."
        with col_e:
            if st.button("🎯 Hackathon Prep"):
                st.session_state.user_template = "Help me prepare for a hackathon in 4 hours."
        st.markdown("---")
        if st.session_state.user_template:
            st.info("Template loaded into the input field. Go to Agents tab to run.")
            st.write(st.session_state.user_template)
        if st.button("🚀 Start Planning"):
            st.session_state.show_intro = False
            st.rerun()
    else:
        st.success("You're ready — open the Agents tab and run the pipeline.")

# Agents tab: main app (inputs + agents)
with tabs[1]:
    col1, col2, col3 = st.columns([3,4,3])
    with col1:
        st.markdown("## 🧭 Input & Preferences")
        user_input = st.text_area(
        "Describe your goal:",
        value=st.session_state.get("user_template", "")
    )
        st.markdown("### 🎤 Voice Input (Optional)")
        audio_bytes = st.audio_input("Speak your goal")
        
        if audio_bytes:
            try:
                transcript = "Voice transcription disabled in offline mode."
                if GENAI_AVAILABLE and not FORCE_OFFLINE:
                    # Gemini Speech-to-Text
                    model = genai.GenerativeModel("gemini-pro")
                    transcript = model.generate_content(
                        ["Transcribe this audio:", audio_bytes]
                    ).text
                st.success("Voice detected!")
                st.write("**Transcribed:**", transcript)
                user_input = transcript   # auto-fill
            except:
                st.error("Unable to transcribe audio.")

        st.markdown("### Profile & Preferences")
        # Load saved profile (persistent)
        saved_profile = session_memory.get("profile", "Student")
        
        profile = st.selectbox(
            "Profile",
            options=["Student", "Developer", "Researcher", "Designer"],
            index=["Student", "Developer", "Researcher", "Designer"].index(saved_profile)
        )
        
        # Avatar icons
        avatars = {
            "Student": "🎓",
            "Developer": "💻",
            "Designer": "🎨",
            "Researcher": "🔬"
        }
        
        st.markdown(f"### {avatars.get(profile, '🙂')} {profile} Mode Activated")
        
        # Wakeup time (loaded from persistent memory)
        wake = st.text_input("Preferred wakeup time", value=session_memory.get("preferred_wakeup", ""))
        
        # Work style (loaded from memory)
        saved_style = session_memory.get("work_style", "Pomodoro")
        style = st.selectbox(
            "Work style",
            options=["Pomodoro", "Continuous", "Custom"],
            index=["Pomodoro", "Continuous", "Custom"].index(saved_style)
        )
        
        # Loop cycles
        cycles = st.slider("Refinement cycles (loop agent)", min_value=1, max_value=4, value=2)
        
        # Run button
        run_btn = st.button("Run Agents")
        
        # Save button
        if st.button("Save Preferences"):
            session_memory.set("preferred_wakeup", wake)
            session_memory.set("work_style", style)
            session_memory.set("profile", profile)
            st.success("Preferences saved to session + long-term memory.")

        st.markdown("---")
        if st.button("🎉 Celebrate last run"):
            st.balloons()

    with col2:
        st.markdown("## 🧠 Agent Outputs")
        planner_exp = st.expander("🧭 Planner Output", expanded=True)
        exec_exp = st.expander("⚙️ Executor Output", expanded=False)
        summ_exp = st.expander("📝 Summarizer Output", expanded=False)
        reflect_exp = st.expander("💡 Reflection Output", expanded=False)
        eval_exp = st.expander("📊 Evaluation", expanded=False)

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
        st.caption("A2A messages and cycle logs appear below.")

    # prepare session logs
    if "dp_logs" not in st.session_state:
        st.session_state.dp_logs = []
    if "last_run_time" not in st.session_state:
        st.session_state.last_run_time = None

    # Run pipeline when clicked
    if run_btn:
        # --- Sliding Step-by-Step Agent Animation ---
        with st.status("🤖 Running Multi-Agent Pipeline...", expanded=True) as status:
            st.write("🧭 Planner Agent starting...")
            time.sleep(0.5)
        
            st.write("⚙️ Executor Agent starting...")
            time.sleep(0.5)
        
            st.write("📝 Summarizer Agent preparing output...")
            time.sleep(0.5)
        
            st.write("💡 Reflection Agent analyzing improvements...")
            time.sleep(0.5)
        
            status.update(label="✨ All agents completed successfully!", state="complete")

        st.session_state.last_run_time = now_ts()
        st.session_state.dp_logs.clear()
        try:
            session_memory.set("preferred_wakeup", wake)
            session_memory.set("work_style", style)
            session_memory.set("profile", profile)
            safe_log(st.session_state.dp_logs, "Main", f"Starting pipeline for profile={profile}, cycles={cycles}")

            # Animated stepper UI
            step_placeholder = st.empty()
            step_placeholder.info("Step 1/4 — Planner running...")
            time.sleep(0.4)

            result = loop_orchestrator(user_input, cycles, st.session_state.dp_logs)

            # show results with animated typing
            step_placeholder.success("Planner ✓ Executor ✓ Summarizer ✓ Reflection ✓")
            animated_text(result["plan"], planner_text)
            animated_text(result["execution"], exec_text)
            animated_text(result["summary"], summ_text)
            # --- Text to Speech Output ---
            try:
                if GENAI_AVAILABLE and not FORCE_OFFLINE:
                    tts_model = genai.GenerativeModel("gemini-pro")
                    audio_data = tts_model.generate_content(
                        f"Convert the following text to speech: {result['summary']}",
                        audio=True,
                    )
                    st.audio(audio_data, format="audio/wav")
            except:
                st.warning("Text-to-speech unavailable in offline mode.")
            animated_text(result["reflection"], reflect_text)
            eval_text.markdown(f"**Evaluation score:** {result['evaluation']['score']}\n\nComponents:\n```\n{json.dumps(result['evaluation']['components'], indent=2)}\n```")

            # Download button for final plan
            st.download_button("📥 Download Final Plan", result["summary"], file_name="daypilot_plan.txt")

            # Celebration if score above threshold
            if result["evaluation"]["score"] >= 70:
                st.balloons()

            mem_box.json(session_memory.dump())
            ltm_box.text(json.dumps(long_term_memory, indent=2)[:2000])
            metrics_box.json(metrics)

        except Exception as ex:
            safe_log(st.session_state.dp_logs, "Main", f"Error: {ex}")
            traceback.print_exc()
            st.error("Agent pipeline failed. See logs.")
        logs_box.text_area("Agent logs (newest last)", value="\n".join(st.session_state.dp_logs), height=350)

    # show when not running
    if not run_btn:
        mem_box.json(session_memory.dump())
        ltm_box.text(json.dumps(long_term_memory, indent=2)[:2000])
        metrics_box.json(metrics)
        logs_box.text_area("Agent logs (newest last)", value="\n".join(st.session_state.dp_logs), height=350)

# Analytics tab
with tabs[2]:
    st.markdown("## 📈 Analytics & Visualization")
    st.markdown("### Agent Network")
    st.graphviz_chart(
        '''
        digraph {
            User -> Planner;
            Planner -> Executor;
            Executor -> Summarizer;
            Summarizer -> Reflection;
            Reflection -> User;
        }
        '''
    )
    st.markdown("### Metrics")
    st.json(metrics)
    st.markdown("### Recent Logs")
    st.code("\n".join(st.session_state.get("dp_logs", [])[-20:]))

# Memory tab
with tabs[3]:
    st.markdown("## 🧠 Memory Inspector")
    st.markdown("### Session Memory")
    st.json(session_memory.dump())
    st.markdown("### Long-Term Memory (sample)")
    st.text(json.dumps(long_term_memory, indent=2)[:5000])
    if st.button("Export Long-Term Memory"):
        st.download_button("Export LTM JSON", json.dumps(long_term_memory, indent=2), file_name="long_term_memory.json")

# Footer & demo tips
st.markdown("---")
st.markdown("### Demo tips")
st.markdown(
    "- Save preferences and run 2–3 cycles for best refinement.\n"
    "- Use the quick templates on the Home tab to load sample prompts.\n"
    "- Use the Download button to get the final plan as a text file.\n"
    "- Toggle OFFLINE_MODE in Streamlit secrets to demo fallback behavior."
)

st.markdown("""
<hr>
<center>
Built with ❤️ by Sai Meghana for Google’s AI Intensive.<br>
<small>Powered by Gemini + Streamlit.</small>
</center>
""", unsafe_allow_html=True)
