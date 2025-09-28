import streamlit as st
from transformers import pipeline
import time
import numpy as np

# --- MODEL AND LABEL CONFIGURATION ---

# Model IDs based on the notebook logic
FINAL_SUMMARY_MODEL = "valhalla/distilbart-mnli-12-3"
CUE_DIMENSION_MODEL = "facebook/bart-large-mnli"

# Labels for Employee Expression Summary (Single Label Classification)
CANDIDATE_LABELS = [
    "complaint about project scope and documentation",
    "feedback on outdated software/technology",
    "complaint about compensation and career growth",
    "complaint about poor network connection/IT infrastructure",
    "inquiry about upcoming salary hike timing",
    "disappointed about low hike and missing bonus",
    "Queries about low arrears and appraisal process",
    "Frustration over lack of promotion despite good performance",
    "Request for flexible working hours",
    "positive feedback on work-life balance/flexibility",
    "positive feedback on management/team collaboration",
]

# Labels for Meta Data (Multi-Label Classification)
META_CATEGORIES = [
    "promotion delay", "performance ratings", "career growth", "policy clarification",
    "leadership development", "arrears", "appraisal letter", "compensation guidelines",
    "HR clarification", "low increment", "missing bonus", "dissatisfied", "rewards concern",
    "salary hike", "increment cycle", "next hike", "clarification",
]

# Labels for Cue Dimension (High-Level Classification)
CUE_DIMENSIONS = [
    "Salary & Compensation Issues", 
    "HR Communication Issues", 
    "Hike / Salary Increment Related", 
    "Promotion / Career Growth"
]

# --- MODEL CACHING FUNCTIONS ---

@st.cache_resource
def load_classifier(model_name):
    """Loads a Hugging Face zero-shot classification pipeline and caches it."""
    try:
        # device=-1 ensures CPU usage, which is safest for general deployment
        return pipeline("zero-shot-classification", model=model_name, device=-1) 
    except Exception as e:
        st.error(f"Error loading model {model_name}. Please ensure PyTorch/TensorFlow are installed correctly. Error: {e}")
        return None

# Load the required models only once
classifier_summary = load_classifier(FINAL_SUMMARY_MODEL)
classifier_cue = load_classifier(CUE_DIMENSION_MODEL)

# --- CORE ANALYSIS FUNCTION ---

@st.cache_data
def analyze_feedback(text):
    """
    Performs the three classification tasks based on the provided notebook logic.
    """
    if not classifier_summary or not classifier_cue:
        return "Model Loading Failed", "Model Loading Failed", "Model Loading Failed"

    # Handle empty/non-string input gracefully
    if not isinstance(text, str) or not text.strip():
        return "No Input Provided", "No Input Provided", "No Meta Tags Found"
        
    # 1. Employee Expression Summary (Single-label)
    summary_result = classifier_summary(text, CANDIDATE_LABELS, multi_label=False)
    predicted_summary = summary_result['labels'][0]

    # 2. Cue Dimension Prediction (Single-label)
    cue_result = classifier_cue(text, CUE_DIMENSIONS, multi_label=False)
    predicted_cue = cue_result['labels'][0]

    # 3. Meta Data Prediction (Multi-label)
    meta_result = classifier_summary(text, META_CATEGORIES, multi_label=True)
    
    # Filter for score > 0.7 and limit to top 8 (as per notebook logic)
    matched_meta = [
        label for label, score in zip(meta_result['labels'], meta_result['scores']) 
        if score > 0.7
    ][:8]
    
    # Format the output string
    predicted_meta = ", ".join(matched_meta) if matched_meta else "No strong meta tags found (Below threshold 0.7)"

    return predicted_summary, predicted_cue, predicted_meta

# --- STREAMLIT UI ---

# CRITICAL CHANGE 1: Set wide layout for max screen width
st.set_page_config(layout="wide", page_title="Employee Feedback Analyzer", initial_sidebar_state="expanded")

st.title("💡 AI Employee Feedback Tagger")
st.markdown("### Zero-Shot Classification for Compensation & HR Issues")

if not classifier_summary or not classifier_cue:
    st.error("Cannot proceed. One or both Transformer models failed to load. Check console for PyTorch/dependency errors.")
else:
    st.markdown("---")
    
    default_text = "I have a query related to salary increment. I did not receive the expected hike after graduation. Also, I didn't receive the bonus letter."
    
    feedback_text = st.text_area(
        "Enter Employee Feedback Comment:",
        default_text,
        height=180
    )

    if st.button("Analyze Feedback", use_container_width=True):
        if not feedback_text.strip():
            st.warning("Please enter a valid feedback comment to analyze.")
            
        else:
            with st.spinner('Running three classification models...'):
                start_time = time.time()
                
                # --- Run Analysis ---
                predicted_summary, predicted_cue, predicted_meta = analyze_feedback(feedback_text)
                
                end_time = time.time()
                elapsed_time = round(end_time - start_time, 2)

                st.success(f"Analysis Complete! (Models took {elapsed_time} seconds)")
                st.markdown("---")
                
                # --- Display Results in Sequential (Top-to-Bottom) Blocks ---
                
                # Helper function for consistent blue output (CRITICAL CHANGE 2)
                def colored_output(label):
                    # Use markdown with HTML inline styling for blue color
                    return f"<h3 style='color: #1E90FF; white-space: normal;'>{label}</h3>"

                # 1. Main Issue Summary
                st.subheader("1. Employee Expression Summary")
                st.markdown(colored_output(predicted_summary), unsafe_allow_html=True)
                st.markdown("---")

                # 2. Detailed Meta Tags
                st.subheader("2. Meta Data")
                st.markdown("**Tags Found (Score > 0.7):**")
                
                # Display the string output with the blue color styling
                st.markdown(colored_output(predicted_meta), unsafe_allow_html=True)
                st.markdown("---")
                
                # 3. Top-Level Dimension
                st.subheader("3. Cue Dimensions")
                st.markdown(colored_output(predicted_cue), unsafe_allow_html=True)
                
                st.markdown("---")
                st.caption(f"Models used: {FINAL_SUMMARY_MODEL} & {CUE_DIMENSION_MODEL}.")
