import plotly.express as px
import os
import pandas as pd
from datetime import datetime
from src.gemini_helper import get_gemini_response
from src.fusion_classifier import HybridEmotionClassifier

import streamlit as st

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Emotion Detection & Learning Support Engine",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# Load Hybrid Classifier
# =====================================================

@st.cache_resource
def load_classifier():
    return HybridEmotionClassifier()

classifier = load_classifier()
def save_to_csv(field, problem, emotion, confidence, response):

    filename = "emotion_response_examples.csv"

    new_row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "field": field,
        "problem": problem,
        "emotion": emotion,
        "confidence": round(confidence, 4),
        "response": response
    }])

    if os.path.exists(filename):
        old = pd.read_csv(filename)
        df = pd.concat([old, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(filename, index=False)
# =====================================================
# Session History
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# Sidebar
# =====================================================

with st.sidebar:

    st.header("⚙️ Settings")

    use_ai = st.checkbox(
        "Use AI Response (Gemini)",
        value=True
    )

    save_data = st.checkbox(
        "Save to CSV for Learning",
        value=True
    )

    show_details = st.checkbox(
        "Show Analysis Details",
        value=False
    )

    st.divider()

    st.subheader("📊 Dashboard")
    st.write("Model : Hybrid (BERT + BiLSTM)")
    st.write("Status : Ready")

if os.path.exists("emotion_response_examples.csv"):
    examples = pd.read_csv("emotion_response_examples.csv")
    st.write(f"CSV Examples : {len(examples)}")
else:
    st.write("CSV Examples : 0")
    st.divider()

st.subheader("🕘 Recent History")

if len(st.session_state.history) == 0:

    st.info("No interactions yet.")

else:

    for item in reversed(st.session_state.history[-5:]):

        st.markdown(
            f"""
**{item['emotion']}**
- {item['field']}
"""
        )
    if st.button("🗑 Clear History"):

     st.session_state.history = []

    st.rerun()

# =====================================================
# Title
# =====================================================

st.title("🧠 Emotion Detection & Learning Support Engine")

st.markdown(
"""
This system analyzes a student's emotional state
and provides AI-powered learning guidance.
"""
)

st.divider()

# =====================================================
# Student Information
# =====================================================

st.subheader("🎓 Student Information")

field = st.selectbox(
    "What field are you studying?",
    [
        "Computer Science",
        "Mathematics",
        "Physics",
        "Chemistry",
        "Biology",
        "Engineering",
        "Business",
        "Literature",
        "History",
        "Psychology",
        "Other"
    ],
    help="Select your study field for personalized responses."
)

problem = st.text_area(
    "Describe your problem or challenge",
    placeholder=f"Example: I'm struggling with concepts in {field}.",
    height=150
)

# =====================================================
# Analyze Button
# =====================================================

if st.button("🧠 Get AI Learning Help", use_container_width=True):

    if not problem.strip():
        st.warning("Please describe your problem first.")

    else:

        with st.spinner("Analyzing learning state..."):

            # Hybrid Prediction
            emotion, confidence, scores = classifier.predict(problem)

            # Gemini Response
            if use_ai:
                ai_response = get_gemini_response(
                    field,
                    problem,
                    emotion,
                    confidence
                )
            else:
                ai_response = (
                    "AI response is disabled. Enable Gemini in the sidebar."
                )

        st.success("Analysis Complete!")

        # Save to history
        st.session_state.history.append({
            "field": field,
            "problem": problem,
            "emotion": emotion,
            "confidence": confidence,
            "response": ai_response
        })
        if save_data:
         save_to_csv(
        field,
        problem,
        emotion,
        confidence,
        ai_response
        )
        # Emotion Analysis
    st.subheader("📊 Emotion Scores")

     # Sort scores from highest to lowest
    sorted_scores = sorted(
    scores.items(),
    key=lambda x: x[1],
    reverse=True
    )

    # Detect mixed emotions
    mixed_emotions = [
    (emotion_name, score)
    for emotion_name, score in sorted_scores
    if score >= 0.15
    ]

    # Display mixed emotions
    if len(mixed_emotions) > 1:

     st.info("🧠 Mixed Emotional State Detected")

    emojis = {
        "Confused": "😕",
        "Frustrated": "😣",
        "Curious": "🤔",
        "Confident": "😎",
        "Bored": "😴",
        "Neutral": "😐",
        "Happy": "😊",
        "Sad": "😔",
        "Angry": "😠",
        "Fear": "😨",
        "Surprise": "😲"
    }

    cols = st.columns(len(mixed_emotions))

    for col, (emotion_name, score) in zip(cols, mixed_emotions):

        with col:

            st.metric(
                label=f"{emojis.get(emotion_name,'🧠')} {emotion_name}",
                value=f"{score*100:.1f}%"
            )

    st.divider()

    # Show all emotion scores
    for emotion_name, score in sorted_scores:

     st.write(f"**{emotion_name}** : {score*100:.2f}%")

    st.progress(float(score))

    st.divider()

    st.subheader("🤖 AI Learning Guidance")
    st.write(ai_response)

    if show_details:
            st.divider()
            st.subheader("Model Details")
            st.json(scores)
            st.write("Field:", field)
            st.write("Problem:", problem)
# =====================================================
# Learning Analytics Dashboard
# =====================================================

if len(st.session_state.history) > 0:

    st.divider()
    st.header("📈 Learning Analytics")

    df = pd.DataFrame(st.session_state.history)

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Emotions",
            "📚 Fields",
            "📋 Summary"
        ]
    )

    # ------------------------------------
    # TAB 1
    # ------------------------------------

    with tab1:

        col1, col2 = st.columns(2)

        with col1:

            emotion_counts = df["emotion"].value_counts()

            fig = px.pie(
                values=emotion_counts.values,
                names=emotion_counts.index,
                title="Emotion Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig2 = px.bar(
                df,
                x="emotion",
                title="Emotion Frequency",
                color="emotion"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

    # ------------------------------------
    # TAB 2
    # ------------------------------------

    with tab2:

        field_df = (
            df.groupby(["field", "emotion"])
              .size()
              .reset_index(name="count")
        )

        fig3 = px.bar(
            field_df,
            x="field",
            y="count",
            color="emotion",
            title="Emotions by Study Field"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    # ------------------------------------
    # TAB 3
    # ------------------------------------

    with tab3:

        st.metric(
            "Total Sessions",
            len(df)
        )

        st.metric(
            "Most Common Emotion",
            df["emotion"].mode()[0]
        )

        st.metric(
            "Average Confidence",
            f"{df['confidence'].mean()*100:.2f}%"
        )

        st.dataframe(df)