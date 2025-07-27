    import streamlit as st
    import requests

    FASTAPI_URL = "http://localhost:8000"  # Change to your deployed URL when live

    st.set_page_config(page_title="Startup Evaluator", layout="wide")

    st.title("🚀 Startup Evaluator using LLaMA-3 + LoRA")

    # --- STEP 1: Collect Startup Info ---
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
            st.error("❌ Please fill in all startup fields before proceeding.")
        else:
            with st.spinner("🔄 Generating VC & Industry Questions..."):
                startup_data = {
                    "name": name,
                    "industry": industry,
                    "pitch": pitch,
                    "year": year,
                    "funding": funding
                }

                try:
                    res = requests.post(f"{FASTAPI_URL}/generate-questions", json=startup_data)
                    res.raise_for_status()
                    questions_text = res.json()["questions"]
                    questions = [q.strip() for q in questions_text.split("\n") if q.strip() and q[0].isdigit()]
                    st.session_state["startup_info"] = startup_data
                    st.session_state["questions"] = questions
                except Exception as e:
                    st.error(f"❌ Failed to generate questions: {e}")

    # --- STEP 2: Show Questions & Collect Answers ---
    if "questions" in st.session_state:
        st.subheader("✍️ Step 2: Answer the Questions")

        answers = []
        with st.form("answer_form"):
            for i, q in enumerate(st.session_state["questions"]):
                answer = st.text_area(f"{q}", key=f"answer_{i}", height=80)
                answers.append(answer)

            eval_button = st.form_submit_button("Get Evaluation")

        if eval_button:
            if not all(answers):
                st.error("❌ Please answer all 10 questions before proceeding.")
            else:
                with st.spinner("📊 Evaluating your startup..."):
                    payload = {
                        "startup_info": st.session_state["startup_info"],
                        "founder_answers": answers
                    }

                    try:
                        eval_res = requests.post(f"{FASTAPI_URL}/evaluate-startup", json=payload)
                        eval_res.raise_for_status()
                        evaluation = eval_res.json()["evaluation"]
                        st.success("✅ Evaluation complete!")
                        st.subheader("📄 Startup Evaluation:")
                        st.markdown(evaluation)
                    except Exception as e:
                        st.error(f"❌ Evaluation failed: {e}")
