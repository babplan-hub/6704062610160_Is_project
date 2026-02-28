import streamlit as st

st.set_page_config(page_title="Overview Neural Network Model", layout="wide")

st.markdown("## 🧠 Neural Network Image Classification Model")
st.markdown("Deep Learning System for Dog Breed Classification")

# =========================
# HIGHLIGHT METRICS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Size", "≈ 40,000 Images")

with col2:
    st.metric("Model Architecture", "MobileNetV2 + Dense")

with col3:
    st.metric("Task Type", "Multi-Class Classification")

st.markdown("---")

# =========================
# DATA PREPARATION
# =========================
st.markdown("### 📊 1️⃣ Data Preparation")

st.markdown("""
The dataset consists of approximately **40,000 dog images** 
covering multiple dog breeds.

Data structure:

• Images are separated into folders by breed  
• 80% used for training  
• 20% used for validation  

Each folder represents one class label.
The system automatically maps folder names into numeric labels.
""")

# =========================
# PREPROCESSING
# =========================
st.markdown("---")
st.markdown("### 🖼 2️⃣ Image Preprocessing")

st.markdown("""
Before training, all images undergo preprocessing:

• Resize to 224 × 224 pixels  
• Normalize pixel values (Rescale 1/255)  
• Load in mini-batches using ImageDataGenerator  

Purpose:
• Reduce computational complexity  
• Improve convergence speed  
• Stabilize training process  
""")

# =========================
# MODEL ARCHITECTURE
# =========================
st.markdown("---")
st.markdown("### 🧠 3️⃣ Model Architecture")

st.markdown("""
The model is built using **Transfer Learning**:

Base Model:
• MobileNetV2 (pretrained on ImageNet)  
• Feature extraction layers frozen  

Custom Layers:
• GlobalAveragePooling2D  
• Dense Layer (ReLU activation)  
• Output Layer (Softmax activation)  

The model predicts probability distribution across all dog breeds.
""")

# =========================
# TRAINING STRATEGY
# =========================
st.markdown("---")
st.markdown("### 🔒 4️⃣ Training Strategy")

st.markdown("""
• Loss Function: Categorical Crossentropy  
• Optimizer: Adam  
• Epochs: 10  
• Batch Size: 32  
• Validation Monitoring  

Transfer learning helps reduce training time 
and improves accuracy even with limited dataset size.
""")

# =========================
# EVALUATION
# =========================
st.markdown("---")
st.markdown("### 📈 5️⃣ Model Evaluation")

st.markdown("""
Model performance is evaluated using:

• Validation Accuracy  
• Loss Curve  
• Class Prediction Confidence  

The trained model achieves high classification accuracy 
by leveraging pretrained feature extraction layers.
""")

st.markdown("---")
st.success("🚀 Production Model: MobileNetV2 Transfer Learning Classifier")