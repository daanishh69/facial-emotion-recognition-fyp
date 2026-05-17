"""
app.py  —  FYP2 Final Facial Emotion Recognition App
=====================================================
Models:  MobileNetV2 (live detection) + ResNet50 (image upload)
Dataset: RAF-DB (7 emotions)
Pages:
  1. Image Prediction  — upload or webcam snapshot
  2. Live Detection    — real-time webcam with emotion history
  3. Evaluation        — confusion matrix, classification report,
                         FYP1 vs FYP2 comparison

Usage:
    streamlit run app.py
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime
from predict import (
    load_model, predict, predict_frame, detect_face,
    EMOTIONS, EMOTION_COLORS, EMOTION_EMOJIS
)

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="FER — Mental Health Monitor",
    page_icon="🧠",
    layout="wide"
)

# ──────────────────────────────────────────────
# DARK NEON THEME
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0a0a0a; color: #e0e0e0; }
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #00ffcc33;
    }
    h1 { color: #00ffcc; text-shadow: 0 0 20px #00ffcc88; font-size: 2rem !important; }
    h2 { color: #ff00ff; text-shadow: 0 0 10px #ff00ff66; }
    h3 { color: #00ccff; }
    .stButton>button {
        background: linear-gradient(135deg, #ff00ff, #00ffcc);
        color: #0a0a0a; border: none; border-radius: 8px;
        font-weight: bold; padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 20px #00ffcc88;
    }
    [data-testid="stMetric"] {
        background-color: #1a1a2e;
        border: 1px solid #00ffcc33;
        border-radius: 10px; padding: 1rem;
    }
    [data-testid="stMetricValue"] { color: #00ffcc !important; font-size: 1.8rem !important; }
    [data-testid="stMetricLabel"] { color: #aaaaaa !important; }
    .stAlert { border-radius: 8px; }
    hr { border-color: #00ffcc33; }
    .emotion-badge {
        display: inline-block; padding: 0.4rem 1.2rem;
        border-radius: 20px; font-size: 1.2rem;
        font-weight: bold; margin: 0.5rem 0;
    }
    .tip-box {
        background: linear-gradient(135deg, #1a1a2e, #0d0d1a);
        border-left: 3px solid #00ffcc; border-radius: 8px;
        padding: 1rem 1.5rem; margin: 1rem 0;
        font-style: italic; color: #cccccc;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
MODEL_PATHS = {
    "MobileNetV2 (Live)": "models/emotion_mobilenet.keras",
    "ResNet50 (Image)":   "models/emotion_resnet50.keras",
}

MENTAL_HEALTH_TIPS = {
    'surprise': "💡 Surprises can be overwhelming. Give yourself a moment to process before reacting.",
    'fear':     "💡 Fear is a signal worth listening to. Take a breath and assess the situation calmly.",
    'disgust':  "💡 It's okay to feel repulsed. Step away from the source and ground yourself.",
    'happy':    "💡 Great to see you smiling! Positive emotions strengthen resilience — enjoy this moment.",
    'sad':      "💡 Sadness is a natural emotion. Consider talking to someone you trust today.",
    'angry':    "💡 Take 5 deep breaths. Anger is valid — try to identify the trigger before reacting.",
    'neutral':  "💡 Feeling balanced is healthy. Use this calm state for reflection or planning.",
}

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def get_model(model_choice):
    path = MODEL_PATHS[model_choice]
    if not os.path.exists(path):
        st.error(f"⚠ Model not found: `{path}`\n\nRun the training script first!")
        st.stop()
    return load_model(path)


def get_model_type(model_choice):
    return "MobileNetV2" if "MobileNetV2" in model_choice else "ResNet50"


def plot_emotion_history(history):
    if len(history) < 2:
        return None

    df = pd.DataFrame(history, columns=['time', 'emotion', 'confidence'])
    df['time_sec'] = df['time'] - df['time'].iloc[0]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), facecolor='#0a0a0a',
        gridspec_kw={'height_ratios': [2, 1]}
    )

    emotion_to_num = {e: i for i, e in enumerate(EMOTIONS)}
    colors = [EMOTION_COLORS[e] for e in df['emotion']]
    y_vals = [emotion_to_num[e] for e in df['emotion']]

    ax1.scatter(df['time_sec'], y_vals, c=colors, s=80, alpha=0.8, zorder=3)
    ax1.plot(df['time_sec'], y_vals, color='#333333', linewidth=0.8, alpha=0.5)
    ax1.set_yticks(range(len(EMOTIONS)))
    ax1.set_yticklabels(
        [f"{EMOTION_EMOJIS[e]} {e.capitalize()}" for e in EMOTIONS],
        color='#cccccc', fontsize=9
    )
    ax1.set_xlabel("Time (seconds)", color='#aaaaaa')
    ax1.set_title("Emotion Timeline", color='#00ffcc', fontsize=12, pad=10)
    ax1.set_facecolor('#111111')
    ax1.tick_params(colors='#aaaaaa')
    ax1.grid(True, alpha=0.1, color='#444444')
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333333')

    counts = df['emotion'].value_counts()
    bar_colors = [EMOTION_COLORS.get(e, '#888888') for e in counts.index]
    bars = ax2.bar(counts.index, counts.values, color=bar_colors,
                   edgecolor='#222222', linewidth=0.5)
    for bar, val in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(val), ha='center', va='bottom', color='#cccccc', fontsize=8)
    ax2.set_title("Emotion Distribution", color='#ff00ff', fontsize=11, pad=8)
    ax2.set_facecolor('#111111')
    ax2.tick_params(colors='#aaaaaa', axis='x', rotation=20)
    ax2.tick_params(colors='#aaaaaa', axis='y')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333333')

    plt.tight_layout(pad=2)
    return fig


def show_session_summary(history):
    if not history:
        return

    df = pd.DataFrame(history, columns=['time', 'emotion', 'confidence'])
    dominant    = df['emotion'].value_counts().idxmax()
    total       = len(df)
    avg_conf    = df['confidence'].mean()
    duration    = df['time'].iloc[-1] - df['time'].iloc[0]

    st.markdown("## 📊 Session Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Dominant Emotion", f"{EMOTION_EMOJIS[dominant]} {dominant.capitalize()}")
    with c2:
        st.metric("Frames Analyzed", total)
    with c3:
        st.metric("Avg Confidence", f"{avg_conf:.1%}")
    with c4:
        st.metric("Session Duration", f"{duration:.0f}s")

    st.markdown("### Emotion Breakdown")
    counts = df['emotion'].value_counts()
    for emotion in EMOTIONS:
        count = counts.get(emotion, 0)
        pct   = count / total * 100 if total > 0 else 0
        color = EMOTION_COLORS[emotion]
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin:4px 0'>"
            f"<span style='width:90px;color:{color};font-weight:bold'>"
            f"{EMOTION_EMOJIS[emotion]} {emotion.capitalize()}</span>"
            f"<div style='flex:1;background:#1a1a1a;border-radius:4px;height:18px'>"
            f"<div style='width:{pct:.1f}%;background:{color};height:100%;"
            f"border-radius:4px;opacity:0.85'></div></div>"
            f"<span style='color:#aaa;width:50px;text-align:right'>{pct:.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        f"<div class='tip-box'>{MENTAL_HEALTH_TIPS[dominant]}</div>",
        unsafe_allow_html=True
    )

    csv = df.to_csv(index=False)
    st.download_button(
        "⬇ Download Session Log (CSV)", data=csv,
        file_name=f"emotion_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime='text/csv'
    )


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🧠 FER Monitor")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🎯 Prediction", "🎥 Live Detection", "📊 Evaluation"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    model_choice = st.selectbox("🤖 Model", list(MODEL_PATHS.keys()))
    st.markdown("---")
    st.markdown(
        "<div style='color:#666;font-size:0.75rem;text-align:center'>"
        "FYP2 — Facial Emotion Recognition<br>"
        "for Mental Health Monitoring<br>"
        "UTeM 2025/2026</div>",
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────
# PAGE 1 — IMAGE PREDICTION
# ──────────────────────────────────────────────
if page == "🎯 Prediction":
    st.markdown("# 🎯 Emotion Prediction")
    st.markdown("Upload a photo or capture from webcam to detect emotions instantly.")
    st.markdown("---")

    model      = get_model(model_choice)
    model_type = get_model_type(model_choice)
    st.success(f"✓ Model loaded: **{model_choice}**")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])
    with col2:
        camera = st.camera_input("📷 Capture from Webcam")

    if uploaded or camera:
        img     = Image.open(uploaded or camera).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        face    = detect_face(img_bgr)

        if face is None:
            st.warning("⚠ No face detected. Please upload a photo containing a face.")
        else:
            img_arr          = np.array(img)
            probs, label, idx = predict(model, img_arr, model_type)

            col_img, col_result = st.columns([1, 1])
            with col_img:
                # Draw face box on display image
                x, y, w, h   = face
                color_hex     = EMOTION_COLORS[label]
                color_rgb     = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                color_bgr     = (color_rgb[2], color_rgb[1], color_rgb[0])
                display_frame = img_bgr.copy()
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color_bgr, 2)
                display_rgb   = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                st.image(display_rgb, caption="Detected Face", use_container_width=True)

            with col_result:
                color = EMOTION_COLORS[label]
                st.markdown(
                    f"<div style='text-align:center;padding:1rem'>"
                    f"<div style='font-size:4rem'>{EMOTION_EMOJIS[label]}</div>"
                    f"<div class='emotion-badge' style='background:{color}22;"
                    f"border:2px solid {color};color:{color}'>"
                    f"{label.upper()}</div>"
                    f"<div style='color:#aaa;margin-top:0.5rem'>"
                    f"Confidence: {probs[idx]:.1%}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

                st.markdown("#### Top 3 Predictions")
                top3 = np.argsort(probs)[-3:][::-1]
                for i in top3:
                    pct       = probs[i]
                    bar_color = EMOTION_COLORS[EMOTIONS[i]]
                    st.markdown(
                        f"<div style='margin:4px 0'>"
                        f"<span style='color:{bar_color};width:80px;display:inline-block'>"
                        f"{EMOTION_EMOJIS[EMOTIONS[i]]} {EMOTIONS[i].capitalize()}</span>"
                        f"<div style='display:inline-block;width:60%;background:#1a1a1a;"
                        f"border-radius:4px;height:14px;vertical-align:middle;margin:0 8px'>"
                        f"<div style='width:{pct*100:.1f}%;background:{bar_color};"
                        f"height:100%;border-radius:4px'></div></div>"
                        f"<span style='color:#aaa'>{pct:.1%}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

            st.markdown(
                f"<div class='tip-box'>{MENTAL_HEALTH_TIPS[label]}</div>",
                unsafe_allow_html=True
            )
    else:
        st.info("👆 Upload an image or use the webcam to get started!")

# ──────────────────────────────────────────────
# PAGE 2 — LIVE DETECTION
# ──────────────────────────────────────────────
elif page == "🎥 Live Detection":
    st.markdown("# 🎥 Live Emotion Detection")
    st.markdown("Real-time emotion tracking via webcam with session history.")
    st.markdown("---")

    model      = get_model(model_choice)
    model_type = get_model_type(model_choice)
    st.success(f"✓ Model loaded: **{model_choice}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶ Start Session", use_container_width=True):
            st.session_state.session_active = True
            st.session_state.emotion_history = []
            st.session_state.frame_count = 0
    with c2:
        if st.button("⏹ Stop Session", use_container_width=True):
            st.session_state.session_active = False
    with c3:
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.emotion_history = []
            st.session_state.frame_count = 0

    st.markdown("---")

    if st.session_state.session_active:
        st.markdown(
            "<div style='color:#00ffcc;font-weight:bold'>🔴 Session Active</div>",
            unsafe_allow_html=True
        )

        frame_input = st.camera_input(
            "Live Feed",
            key=f"frame_{st.session_state.frame_count}",
            label_visibility="collapsed"
        )

        if frame_input is not None:
            img_pil   = Image.open(frame_input).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            probs, label, face_coords = predict_frame(model, frame_bgr, model_type)

            col_feed, col_info = st.columns([3, 2])

            with col_feed:
                if face_coords is not None and label is not None:
                    x, y, w, h    = face_coords
                    color_hex     = EMOTION_COLORS[label]
                    color_rgb     = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    color_bgr     = (color_rgb[2], color_rgb[1], color_rgb[0])
                    display_frame = frame_bgr.copy()
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color_bgr, 2)
                    cv2.putText(display_frame, label.upper(),
                                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
                    display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    st.image(display_rgb, use_container_width=True,
                             caption="Live Feed with Face Detection")

                    confidence = float(probs[np.argmax(probs)])
                    st.session_state.emotion_history.append(
                        (time.time(), label, confidence)
                    )
                    st.session_state.frame_count += 1
                else:
                    display_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    st.image(display_rgb, use_container_width=True,
                             caption="No face detected")

            with col_info:
                if label is not None:
                    color = EMOTION_COLORS[label]
                    st.markdown(
                        f"<div style='text-align:center;padding:1rem'>"
                        f"<div style='font-size:3.5rem'>{EMOTION_EMOJIS[label]}</div>"
                        f"<div class='emotion-badge' style='background:{color}22;"
                        f"border:2px solid {color};color:{color}'>"
                        f"{label.upper()}</div>"
                        f"<div style='color:#aaa;margin-top:0.5rem'>"
                        f"Confidence: {probs[np.argmax(probs)]:.1%}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    st.markdown("**Probabilities**")
                    for emotion, prob in zip(EMOTIONS, probs):
                        bar_color = EMOTION_COLORS[emotion]
                        st.markdown(
                            f"<div style='margin:3px 0;font-size:0.85rem'>"
                            f"<span style='color:{bar_color};width:70px;"
                            f"display:inline-block'>{emotion.capitalize()}</span>"
                            f"<div style='display:inline-block;width:50%;"
                            f"background:#1a1a1a;border-radius:3px;height:12px;"
                            f"vertical-align:middle;margin:0 6px'>"
                            f"<div style='width:{prob*100:.1f}%;background:{bar_color};"
                            f"height:100%;border-radius:3px'></div></div>"
                            f"<span style='color:#888;font-size:0.8rem'>{prob:.1%}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                st.markdown("---")
                frames = len(st.session_state.emotion_history)
                st.markdown(
                    f"<div style='color:#aaa;font-size:0.85rem'>"
                    f"📸 Frames analyzed: <b style='color:#00ffcc'>{frames}</b>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        if st.session_state.session_active:
            st.rerun()

    else:
        st.info("👆 Press **Start Session** to begin live emotion detection.")

    if len(st.session_state.emotion_history) >= 2:
        st.markdown("---")
        st.markdown("## 📈 Session Emotion History")
        fig = plot_emotion_history(st.session_state.emotion_history)
        if fig:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        st.markdown("---")
        show_session_summary(st.session_state.emotion_history)

# ──────────────────────────────────────────────
# PAGE 3 — EVALUATION
# ──────────────────────────────────────────────
elif page == "📊 Evaluation":
    st.markdown("# 📊 Model Evaluation Dashboard")
    st.markdown("---")

    path = MODEL_PATHS[model_choice]
    if not os.path.exists(path):
        st.error(f"Model not found: `{path}`")
        st.stop()

    model      = tf.keras.models.load_model(path)
    model_type = get_model_type(model_choice)
    st.success(f"✓ Loaded: **{model_choice}**")

    if model_type == "MobileNetV2":
        target_size = (160, 160)
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as pi
    else:
        target_size = (224, 224)
        from tensorflow.keras.applications.resnet50 import preprocess_input as pi

    datagen  = ImageDataGenerator(preprocessing_function=pi)
    test_gen = datagen.flow_from_directory(
        r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\data\DATASET\test",
        target_size=target_size,
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    with st.spinner("Evaluating on test set..."):
        loss, acc = model.evaluate(test_gen, verbose=0)
        y_true    = test_gen.classes
        y_pred    = np.argmax(model.predict(test_gen, verbose=0), axis=1)

    labels = list(test_gen.class_indices.keys())
    report = classification_report(y_true, y_pred,
                                   target_names=labels, output_dict=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Test Accuracy", f"{acc:.2%}")
    with c2:
        st.metric("Test Loss", f"{loss:.4f}")
    with c3:
        st.metric("Macro F1-Score", f"{report['macro avg']['f1-score']:.4f}")

    st.markdown("---")
    col_cm, col_rep = st.columns([1, 1])

    with col_cm:
        st.markdown("### Confusion Matrix")
        cm  = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(7, 6), facecolor='#111111')
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                    xticklabels=labels, yticklabels=labels,
                    ax=ax, linewidths=0.5, annot_kws={"size": 8})
        ax.set_facecolor('#111111')
        ax.set_ylabel("True Label", color='#cccccc')
        ax.set_xlabel("Predicted Label", color='#cccccc')
        ax.tick_params(colors='#cccccc', rotation=30)
        ax.set_title(f"Confusion Matrix — {model_choice}", color='#00ffcc', pad=10)
        fig.patch.set_facecolor('#111111')
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_rep:
        st.markdown("### Classification Report")
        df_report = pd.DataFrame(report).transpose().round(3)
        df_report['support'] = df_report['support'].astype(int)
        styled = df_report.style \
            .background_gradient(cmap='RdYlGn',
                                 subset=['precision', 'recall', 'f1-score']) \
            .format("{:.3f}", subset=['precision', 'recall', 'f1-score']) \
            .format("{:d}", subset=['support'])
        st.dataframe(styled, use_container_width=True)

    # FYP1 vs FYP2 comparison
    st.markdown("---")
    st.markdown("### 📈 FYP1 vs FYP2 Comparison")

    comp_df = pd.DataFrame({
        'Model':          ['Baseline CNN', 'MobileNetV2', 'EfficientNetB0',
                           'MobileNetV2 (RAF-DB)', 'ResNet50 (RAF-DB)'],
        'Version':        ['FYP1', 'FYP1', 'FYP1', 'FYP2', 'FYP2'],
        'Dataset':        ['FER+', 'FER+', 'FER+', 'RAF-DB', 'RAF-DB'],
        'Val Accuracy':   [0.6384, 0.6291, 0.1335, 0.7496, 0.7696],
    })

    fig2, ax2 = plt.subplots(figsize=(10, 4), facecolor='#111111')
    colors = ['#FF4B4B', '#FF4B4B', '#FF4B4B', '#00ffcc', '#00ffcc']
    bars   = ax2.barh(comp_df['Model'], comp_df['Val Accuracy'],
                      color=colors, edgecolor='#333333')

    for bar, val in zip(bars, comp_df['Val Accuracy']):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.2%}", va='center', color='#cccccc', fontsize=9)

    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("Validation Accuracy", color='#aaaaaa')
    ax2.set_title("FYP1 vs FYP2 Model Comparison", color='#00ffcc', pad=10)
    ax2.set_facecolor('#111111')
    ax2.tick_params(colors='#cccccc')
    ax2.axvline(x=0.5, color='#444444', linestyle='--', linewidth=0.8)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333333')

    from matplotlib.patches import Patch
    legend = [Patch(color='#FF4B4B', label='FYP1'),
              Patch(color='#00ffcc', label='FYP2')]
    ax2.legend(handles=legend, facecolor='#1a1a1a',
               labelcolor='#cccccc', edgecolor='#444444')

    fig2.patch.set_facecolor('#111111')
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)