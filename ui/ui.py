import streamlit as st
import requests
import re

FASTAPI_URL = "http://localhost:8000"  # or your external URL if deployed

st.set_page_config(page_title="Startup Evaluator", layout="wide")
st.title("🚀 Startup Evaluator using LLaMA-3 + LoRA")

# -------- STEP 1: Startup Info Form --------
with st.form("startup_info_form"):
    st.subheader("📋 Step 1: Enter Your Startup Details")

    name = st.text_input("Startup Name")
    industry = st.text_input("Industry")
    pitch = st.text_area("Pitch", height=150)
    year = st.text_input("Founding Year (e.g., 2021)")
    funding = st.text_input("Funding Raised (e.g., $1.5M)")

    submitted = st.form_submit_button("Generate Questions")

if submitted:
    if not all([name, industry, pitch, year, funding]):
        st.error("❌ Please fill in all fields.")
    else:
        startup_data = {
            "name": name,
            "industry": industry,
            "pitch": pitch,
            "year": year,
            "funding": funding
        }

        with st.spinner("🔄 Generating Questions..."):
            try:
                res = requests.post(f"{FASTAPI_URL}/generate-questions", json=startup_data)
                res.raise_for_status()
                full_output = res.json()["questions"]

                # Extract only numbered questions (1. ... to 10. ...)
                questions = re.findall(r"\d+\.\s.+", full_output.strip())
                if len(questions) > 10:
                    questions = questions[:10]

                st.session_state["startup_info"] = startup_data
                st.session_state["questions"] = questions
                st.success("✅ Questions generated successfully!")

            except Exception as e:
                st.error(f"❌ Error generating questions: {e}")

# -------- STEP 2: Answer Questions --------
if "questions" in st.session_state:
    st.subheader("✍️ Step 2: Answer the Questions")

    answers = []
    with st.form("answer_form"):
        for i, question in enumerate(st.session_state["questions"]):
            answer = st.text_area(f"{question}", key=f"answer_{i}", height=80)
            answers.append(answer)

        eval_button = st.form_submit_button("Get Evaluation")

    if eval_button:
        if not all(answers):
            st.error("❌ Please answer all questions.")
        else:
            with st.spinner("📊 Evaluating your startup..."):
                payload = {
                    "startup_info": st.session_state["startup_info"],
                    "founder_answers": answers
                }

                try:
                    eval_res = requests.post(f"{FASTAPI_URL}/evaluate-startup", json=payload)
                    eval_res.raise_for_status()
                    full_eval = eval_res.json()["evaluation"]

                    # -------- Parse Evaluation --------
                    pattern = r"(\d+)\.\s(.+?):\s(\d+(?:\.\d+)?)\s[—-]\s(.+?)(?:\.|\n)"
                    matches = re.findall(pattern, full_eval)

                    overall = ""
                    if "Overall Evaluation:" in full_eval:
                        overall = full_eval.split("Overall Evaluation:")[-1].strip()

                    st.success("✅ Evaluation completed!")
                    st.subheader("📄 Evaluation Summary")

                    # -------- TABLE OF 21 METRICS --------
                    if matches:
                        import pandas as pd
                        data = []
                        for num, metric, score, comment in matches:
                            data.append({
                                "Metric": metric.strip(),
                                "Score": float(score),
                                "Insight": comment.strip()
                            })

                        df = pd.DataFrame(data)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("⚠️ Could not extract table info from the evaluation.")

                    # -------- SHOW OVERALL PARAGRAPH --------
                    if overall:
                        st.subheader("🧠 Overall Evaluation")
                        st.markdown(overall)

                except Exception as e:
                    st.error(f"❌ Evaluation request failed: {e}")
