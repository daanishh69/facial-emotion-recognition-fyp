"""
app.py  —  FYP2 Final Facial Emotion Recognition App
=====================================================
Models:
  - MobileNetV2  → Live Detection page ONLY (optimized for real-time)
  - DenseNet121  → Image Prediction page ONLY (optimized for accuracy)

Dataset: RAF-DB (7 emotions)
Pages:
  1. Image Prediction  — DenseNet121, upload or webcam snapshot
  2. Live Detection    — MobileNetV2, real-time session tracking
  3. Evaluation        — both models, confusion matrix, comparison chart

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
    .model-badge {
        display: inline-block; padding: 0.2rem 0.8rem;
        border-radius: 12px; font-size: 0.75rem;
        font-weight: bold; margin-left: 8px;
        background: #1a1a2e; border: 1px solid #00ffcc44;
        color: #00ffcc;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
# Fixed model assignments
LIVE_MODEL_PATH  = "models/emotion_mobilenet.keras"   # MobileNetV2 — live detection
IMAGE_MODEL_PATH = "models/emotion_densenet.keras"    # DenseNet121 — image prediction

# For evaluation page
MODEL_PATHS = {
    "MobileNetV2 (Live Detection)": LIVE_MODEL_PATH,
    "DenseNet121 (Image Prediction)": IMAGE_MODEL_PATH,
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
def get_model(path):
    if not os.path.exists(path):
        st.error(f"⚠ Model not found: `{path}`\n\nRun the training script first!")
        st.stop()
    return load_model(path)


def plot_emotion_history(history):
    if len(history) < 2:
        return None
    df = pd.DataFrame(history, columns=['time', 'emotion', 'confidence'])
    df['time_sec'] = df['time'] - df['time'].iloc[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), facecolor='#0a0a0a',
                                    gridspec_kw={'height_ratios': [2, 1]})
    emotion_to_num = {e: i for i, e in enumerate(EMOTIONS)}
    colors = [EMOTION_COLORS[e] for e in df['emotion']]
    y_vals = [emotion_to_num[e] for e in df['emotion']]

    ax1.scatter(df['time_sec'], y_vals, c=colors, s=80, alpha=0.8, zorder=3)
    ax1.plot(df['time_sec'], y_vals, color='#333333', linewidth=0.8, alpha=0.5)
    ax1.set_yticks(range(len(EMOTIONS)))
    ax1.set_yticklabels([f"{EMOTION_EMOJIS[e]} {e.capitalize()}" for e in EMOTIONS],
                        color='#cccccc', fontsize=9)
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
    dominant = df['emotion'].value_counts().idxmax()
    total    = len(df)
    avg_conf = df['confidence'].mean()
    duration = df['time'].iloc[-1] - df['time'].iloc[0]

    st.markdown("## 📊 Session Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Dominant Emotion", f"{EMOTION_EMOJIS[dominant]} {dominant.capitalize()}")
    with c2: st.metric("Frames Analyzed", total)
    with c3: st.metric("Avg Confidence", f"{avg_conf:.1%}")
    with c4: st.metric("Session Duration", f"{duration:.0f}s")

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
            f"<div style='width:{pct:.1f}%;background:{color};height:100%;border-radius:4px;opacity:0.85'></div></div>"
            f"<span style='color:#aaa;width:50px;text-align:right'>{pct:.1f}%</span>"
            f"</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='tip-box'>{MENTAL_HEALTH_TIPS[dominant]}</div>", unsafe_allow_html=True)

    csv = df.to_csv(index=False)
    st.download_button("⬇ Download Session Log (CSV)", data=csv,
                       file_name=f"emotion_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                       mime='text/csv')


def draw_prob_bars(probs):
    for emotion, prob in zip(EMOTIONS, probs):
        bar_color = EMOTION_COLORS[emotion]
        st.markdown(
            f"<div style='margin:3px 0;font-size:0.85rem'>"
            f"<span style='color:{bar_color};width:70px;display:inline-block'>{emotion.capitalize()}</span>"
            f"<div style='display:inline-block;width:50%;background:#1a1a1a;border-radius:3px;"
            f"height:12px;vertical-align:middle;margin:0 6px'>"
            f"<div style='width:{prob*100:.1f}%;background:{bar_color};height:100%;border-radius:3px'></div></div>"
            f"<span style='color:#888;font-size:0.8rem'>{prob:.1%}</span>"
            f"</div>", unsafe_allow_html=True)


def draw_emotion_result(label, probs, idx):
    color = EMOTION_COLORS[label]
    st.markdown(
        f"<div style='text-align:center;padding:1rem'>"
        f"<div style='font-size:4rem'>{EMOTION_EMOJIS[label]}</div>"
        f"<div class='emotion-badge' style='background:{color}22;border:2px solid {color};color:{color}'>"
        f"{label.upper()}</div>"
        f"<div style='color:#aaa;margin-top:0.5rem'>Confidence: {probs[idx]:.1%}</div>"
        f"</div>", unsafe_allow_html=True)

    st.markdown("#### Top 3 Predictions")
    top3 = np.argsort(probs)[-3:][::-1]
    for i in top3:
        pct       = probs[i]
        bar_color = EMOTION_COLORS[EMOTIONS[i]]
        st.markdown(
            f"<div style='margin:4px 0'>"
            f"<span style='color:{bar_color};width:80px;display:inline-block'>"
            f"{EMOTION_EMOJIS[EMOTIONS[i]]} {EMOTIONS[i].capitalize()}</span>"
            f"<div style='display:inline-block;width:60%;background:#1a1a1a;border-radius:4px;"
            f"height:14px;vertical-align:middle;margin:0 8px'>"
            f"<div style='width:{pct*100:.1f}%;background:{bar_color};height:100%;border-radius:4px'></div></div>"
            f"<span style='color:#aaa'>{pct:.1%}</span>"
            f"</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🧠 FER Monitor")
    st.markdown("---")
    page = st.radio("Navigation",
                    ["🎯 Image Prediction", "🎥 Live Detection", "📊 Evaluation"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown(
        "<div style='color:#666;font-size:0.75rem;text-align:center'>"
        "FYP2 — Facial Emotion Recognition<br>for Mental Health Monitoring<br>"
        "UTeM 2025/2026</div>", unsafe_allow_html=True)
    st.markdown("---")
    # Model info
    if page == "🎯 Image Prediction":
        st.markdown("<div style='color:#00ffcc;font-size:0.8rem'>🤖 Model: DenseNet121</div>", unsafe_allow_html=True)
    elif page == "🎥 Live Detection":
        st.markdown("<div style='color:#00ffcc;font-size:0.8rem'>🤖 Model: MobileNetV2</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#aaa;font-size:0.8rem'>Select model in page</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# PAGE 1 — IMAGE PREDICTION (DenseNet121)
# ──────────────────────────────────────────────
if page == "🎯 Image Prediction":
    st.markdown("# 🎯 Image Emotion Prediction")
    st.markdown("Upload a photo or capture from webcam to detect emotions.")
    st.markdown("<span class='model-badge'>DenseNet121</span>", unsafe_allow_html=True)
    st.markdown("---")

    model = get_model(IMAGE_MODEL_PATH)
    st.success("✓ Model loaded: **DenseNet121**")

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
            st.warning("⚠ No face detected. Please upload a photo containing a clear face.")
        else:
            img_arr          = np.array(img)
            probs, label, idx = predict(model, img_arr, "DenseNet121")

            col_img, col_result = st.columns([1, 1])
            with col_img:
                # Draw face bounding box
                x, y, w, h   = face
                color_hex    = EMOTION_COLORS[label]
                color_rgb    = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                color_bgr    = (color_rgb[2], color_rgb[1], color_rgb[0])
                display      = img_bgr.copy()
                cv2.rectangle(display, (x, y), (x+w, y+h), color_bgr, 2)
                st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                         caption="Detected Face", use_container_width=True)

            with col_result:
                draw_emotion_result(label, probs, idx)

            st.markdown(f"<div class='tip-box'>{MENTAL_HEALTH_TIPS[label]}</div>",
                        unsafe_allow_html=True)
    else:
        st.info("👆 Upload an image or use the webcam to get started!")

# ──────────────────────────────────────────────
# PAGE 2 — LIVE DETECTION (MobileNetV2)
# ──────────────────────────────────────────────
elif page == "🎥 Live Detection":
    st.markdown("# 🎥 Live Emotion Detection")
    st.markdown("Real-time emotion tracking via webcam with session history.")
    st.markdown("<span class='model-badge'>MobileNetV2</span>", unsafe_allow_html=True)
    st.markdown("---")

    model = get_model(LIVE_MODEL_PATH)
    st.success("✓ Model loaded: **MobileNetV2**")

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
        st.markdown("<div style='color:#00ffcc;font-weight:bold'>🔴 Session Active</div>",
                    unsafe_allow_html=True)

        frame_input = st.camera_input("Live Feed",
                                      key=f"frame_{st.session_state.frame_count}",
                                      label_visibility="collapsed")

        if frame_input is not None:
            img_pil   = Image.open(frame_input).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            probs, label, face_coords = predict_frame(model, frame_bgr, "MobileNetV2")

            col_feed, col_info = st.columns([3, 2])

            with col_feed:
                if face_coords is not None and label is not None:
                    x, y, w, h = face_coords
                    color_hex  = EMOTION_COLORS[label]
                    color_rgb  = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                    color_bgr  = (color_rgb[2], color_rgb[1], color_rgb[0])
                    display    = frame_bgr.copy()
                    cv2.rectangle(display, (x, y), (x+w, y+h), color_bgr, 2)
                    cv2.putText(display, label.upper(), (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 2)
                    st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                             use_container_width=True, caption="Live Feed with Face Detection")
                    confidence = float(probs[np.argmax(probs)])
                    st.session_state.emotion_history.append((time.time(), label, confidence))
                    st.session_state.frame_count += 1
                else:
                    st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                             use_container_width=True, caption="No face detected")

            with col_info:
                if label is not None:
                    color = EMOTION_COLORS[label]
                    st.markdown(
                        f"<div style='text-align:center;padding:1rem'>"
                        f"<div style='font-size:3.5rem'>{EMOTION_EMOJIS[label]}</div>"
                        f"<div class='emotion-badge' style='background:{color}22;"
                        f"border:2px solid {color};color:{color}'>{label.upper()}</div>"
                        f"<div style='color:#aaa;margin-top:0.5rem'>"
                        f"Confidence: {probs[np.argmax(probs)]:.1%}</div>"
                        f"</div>", unsafe_allow_html=True)
                    st.markdown("**Probabilities**")
                    draw_prob_bars(probs)

                st.markdown("---")
                frames = len(st.session_state.emotion_history)
                st.markdown(f"<div style='color:#aaa;font-size:0.85rem'>"
                            f"📸 Frames analyzed: <b style='color:#00ffcc'>{frames}</b>"
                            f"</div>", unsafe_allow_html=True)

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
# PAGE 3 — EVALUATION (both models)
# ──────────────────────────────────────────────
elif page == "📊 Evaluation":
    st.markdown("# 📊 Model Evaluation Dashboard")
    st.markdown("---")

    model_choice = st.selectbox("Select Model to Evaluate", list(MODEL_PATHS.keys()))
    path         = MODEL_PATHS[model_choice]
    model_type   = "MobileNetV2" if "MobileNetV2" in model_choice else "DenseNet121"

    if not os.path.exists(path):
        st.error(f"Model not found: `{path}`")
        st.stop()

    # ── CACHE LAYER 1: in-memory via st.cache_resource ──
    # Prevents reloading model + re-evaluating within same session
    @st.cache_resource(show_spinner=False)
    def load_eval_model(model_path):
        return tf.keras.models.load_model(model_path)

    # ── CACHE LAYER 2: file-based via JSON ──
    # Persists evaluation results across app restarts
    import json
    CACHE_FILE = f"eval_cache_{model_type}.json"

    @st.cache_data(show_spinner=False)
    def load_cached_results(cache_file):
        """Load cached evaluation results from JSON file."""
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data = json.load(f)
            return data
        return None

    def run_evaluation(model_path, mt):
        """Run full evaluation — only called when no cache exists."""
        eval_model = load_eval_model(model_path)

        if mt == "MobileNetV2":
            target_size = (160, 160)
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as pi
        else:
            target_size = (224, 224)
            from tensorflow.keras.applications.densenet import preprocess_input as pi

        DATA_TEST = r"C:\Users\Danish Omar\Desktop\SEM5\FYPPROJECT\data\DATASET\test"
        datagen   = ImageDataGenerator(preprocessing_function=pi)
        test_gen  = datagen.flow_from_directory(
            DATA_TEST, target_size=target_size, color_mode="rgb",
            batch_size=32, class_mode="categorical", shuffle=False)

        loss, acc  = eval_model.evaluate(test_gen, verbose=0)
        y_true     = test_gen.classes.tolist()
        y_pred     = np.argmax(eval_model.predict(test_gen, verbose=0), axis=1).tolist()
        labels_out = list(test_gen.class_indices.keys())

        results = {
            "acc": acc, "loss": loss,
            "y_true": y_true, "y_pred": y_pred,
            "labels": labels_out
        }

        # Save to JSON cache file
        with open(CACHE_FILE, "w") as f:
            json.dump(results, f)

        return results

    # ── Try cache first, evaluate only if needed ──
    cached = load_cached_results(CACHE_FILE)

    if cached:
        st.success(f"✓ Loaded: **{model_choice}** — results from cache ⚡")
        col_info, col_btn = st.columns([4, 1])
        with col_info:
            st.info("Results loaded from cache. Click **Re-evaluate** to run fresh evaluation.")
        with col_btn:
            if st.button("🔄 Re-evaluate"):
                # Clear both caches and re-run
                if os.path.exists(CACHE_FILE):
                    os.remove(CACHE_FILE)
                load_cached_results.clear()
                st.rerun()
        results = cached
    else:
        st.success(f"✓ Loaded: **{model_choice}**")
        with st.spinner("⏳ Evaluating on test set — this may take a minute..."):
            results = run_evaluation(path, model_type)
        st.success("✓ Evaluation complete! Results cached for next time.")

    acc    = results["acc"]
    loss   = results["loss"]
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    labels = results["labels"]
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Test Accuracy", f"{acc:.2%}")
    with c2: st.metric("Test Loss", f"{loss:.4f}")
    with c3: st.metric("Macro F1-Score", f"{report['macro avg']['f1-score']:.4f}")

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
        ax.set_title(f"Confusion Matrix — {model_type}", color='#00ffcc', pad=10)
        fig.patch.set_facecolor('#111111')
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_rep:
        st.markdown("### Classification Report")
        df_report = pd.DataFrame(report).transpose().round(3)
        df_report['support'] = df_report['support'].astype(int)
        styled = df_report.style \
            .background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']) \
            .format("{:.3f}", subset=['precision', 'recall', 'f1-score']) \
            .format("{:d}", subset=['support'])
        st.dataframe(styled, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📈 FYP1 vs FYP2 Model Comparison")

    comp_df = pd.DataFrame({
        'Model':        ['Baseline CNN (FYP1)', 'MobileNetV2 (FYP1)',
                         'EfficientNetB0 (FYP1)', 'MobileNetV2 (FYP2)', 'DenseNet121 (FYP2)'],
        'Val Accuracy': [0.6384, 0.6291, 0.1335, 0.7859, 0.7353],
        'Version':      ['FYP1', 'FYP1', 'FYP1', 'FYP2', 'FYP2']
    })

    fig2, ax2 = plt.subplots(figsize=(10, 4), facecolor='#111111')
    colors = ['#FF4B4B' if v == 'FYP1' else '#00ffcc' for v in comp_df['Version']]
    bars   = ax2.barh(comp_df['Model'], comp_df['Val Accuracy'], color=colors, edgecolor='#333333')

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
    ax2.legend(handles=[Patch(color='#FF4B4B', label='FYP1'),
                        Patch(color='#00ffcc', label='FYP2')],
               facecolor='#1a1a1a', labelcolor='#cccccc', edgecolor='#444444')
    fig2.patch.set_facecolor('#111111')
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)