import streamlit as st
import time

# =============================
# Gradient Header Banner
# =============================
def render_header():
    st.markdown(
        """
        <div style='padding:18px; text-align:center; color:white; font-size:30px;
        background: linear-gradient(to right, #6a11cb, #2575fc); border-radius:12px;'>
            ✨ DayPilot AI — Intelligent Multi-Agent Concierge ✨
        </div>
        <br>
        """,
        unsafe_allow_html=True,
    )

# =============================
# Theme Toggle
# =============================
def theme_toggle():
    mode = st.radio("Theme Mode", ["Light", "Dark"], horizontal=True)
    if mode == "Dark":
        st.markdown(
            """
            <style>
            body, .css-18e3th9, .css-1d391kg {
                background-color: #111 !important;
                color: #f0f0f0 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    return mode

# =============================
# Markdown Badges
# =============================
def show_badges(profile, model, offline):
    st.markdown(
        f"""
        **Profile:** ![](https://img.shields.io/badge/Profile-{profile}-blue)
        &nbsp;&nbsp; **Model:** ![](https://img.shields.io/badge/LLM-{model}-purple)
        &nbsp;&nbsp; **Mode:** ![](https://img.shields.io/badge/Mode-{'Offline' if offline else 'Online'}-{'red' if offline else 'green'})
        &nbsp;&nbsp; **Status:** ![](https://img.shields.io/badge/Status-Active-brightgreen)
        """,
        unsafe_allow_html=True,
    )

# =============================
# Custom Icons for Expanders
# =============================
icons = {
    "planner": "🧭",
    "executor": "⚙️",
    "summarizer": "📝",
    "reflection": "💡",
    "evaluation": "📊",
}

# =============================
# Animated Typing Effect
# =============================
def animated_text(text, placeholder):
    placeholder.empty()
    buffer = ""
    for ch in text:
        buffer += ch
        placeholder.markdown(f"```\n{buffer}\n```")
        time.sleep(0.004)

# =============================
# CSS Styling Injection
# =============================
def inject_css():
    st.markdown(
        """
        <style>

        /* Planner */
        .planner-block {
            padding: 15px;
            background-color: #e3f2fd;
            border-left: 6px solid #2196f3;
            border-radius: 12px;
        }

        /* Executor */
        .executor-block {
            padding: 15px;
            background-color: #e8f5e9;
            border-left: 6px solid #4caf50;
            border-radius: 12px;
        }

        /* Summarizer */
        .summarizer-block {
            padding: 15px;
            background-color: #fff3e0;
            border-left: 6px solid #fb8c00;
            border-radius: 12px;
        }

        /* Reflection */
        .reflection-block {
            padding: 15px;
            background-color: #fce4ec;
            border-left: 6px solid #d81b60;
            border-radius: 12px;
        }

        /* Evaluation */
        .evaluation-block {
            padding: 15px;
            background-color: #ede7f6;
            border-left: 6px solid #5e35b1;
            border-radius: 12px;
        }

        /* Logs console */
        #logs-area textarea {
            background-color: #000;
            color: #0f0;
            font-family: 'Consolas', monospace;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
