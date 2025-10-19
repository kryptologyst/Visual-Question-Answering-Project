"""
Streamlit web interface for Visual Question Answering.
"""

import streamlit as st
import logging
from pathlib import Path
from typing import Optional, List
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Import our modules
from vqa_model import VQAModel, VQAResult, create_sample_images
from config import ConfigManager, UIConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Visual Question Answering",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'vqa_model' not in st.session_state:
    st.session_state.vqa_model = None
if 'config_manager' not in st.session_state:
    st.session_state.config_manager = ConfigManager()
if 'history' not in st.session_state:
    st.session_state.history = []


def load_model() -> VQAModel:
    """Load VQA model with caching."""
    if st.session_state.vqa_model is None:
        with st.spinner("Loading VQA model..."):
            config = st.session_state.config_manager.get_model_config()
            st.session_state.vqa_model = VQAModel(
                model_name=config.model_name,
                device=config.device,
                use_pipeline=config.use_pipeline
            )
    return st.session_state.vqa_model


def display_image_with_question(image: Image.Image, question: str, answer: str) -> None:
    """Display image with question and answer."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    ax.set_title(f"Q: {question}\nA: {answer}", fontsize=14, pad=20)
    ax.axis('off')
    st.pyplot(fig)


def main():
    """Main Streamlit application."""
    config = st.session_state.config_manager.get_ui_config()
    
    # Header
    st.title("🔍 Visual Question Answering Demo")
    st.markdown("Ask questions about images using state-of-the-art AI models!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = {
            "BLIP Base": "Salesforce/blip-vqa-base",
            "BLIP Large": "Salesforce/blip-vqa-capfilt-large",
            "BLIP-2": "Salesforce/blip2-opt-2.7b"
        }
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0
        )
        
        # Device selection
        device_options = ["auto", "cpu", "cuda", "mps"]
        selected_device = st.selectbox("Device", device_options, index=0)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            max_length = st.slider("Max Answer Length", 10, 100, 50)
            num_beams = st.slider("Number of Beams", 1, 10, 5)
            temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        
        # Load model button
        if st.button("🔄 Load Model"):
            st.session_state.vqa_model = None
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Upload Image")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to ask questions about"
        )
        
        # Sample images
        st.subheader("🎨 Sample Images")
        if st.button("Generate Sample Images"):
            sample_dir = Path("data/samples")
            sample_images = create_sample_images(sample_dir, 3)
            st.success(f"Generated {len(sample_images)} sample images!")
        
        # Display sample images
        sample_dir = Path("data/samples")
        if sample_dir.exists():
            sample_files = list(sample_dir.glob("*.jpg"))
            if sample_files:
                selected_sample = st.selectbox(
                    "Select Sample Image",
                    options=sample_files,
                    format_func=lambda x: x.name
                )
                if selected_sample:
                    sample_image = Image.open(selected_sample)
                    st.image(sample_image, caption=selected_sample.name, use_column_width=True)
    
    with col2:
        st.header("❓ Ask Questions")
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="What do you see in this image?",
            help="Ask any question about the uploaded image"
        )
        
        # Preset questions
        st.subheader("💡 Preset Questions")
        preset_questions = [
            "What is in this image?",
            "What colors do you see?",
            "How many objects are there?",
            "What is the main subject?",
            "Describe the scene",
            "What is the mood or atmosphere?"
        ]
        
        cols = st.columns(2)
        for i, preset_q in enumerate(preset_questions):
            with cols[i % 2]:
                if st.button(preset_q, key=f"preset_{i}"):
                    question = preset_q
                    st.rerun()
        
        # Answer button
        if st.button("🔍 Get Answer", type="primary"):
            if uploaded_file is not None:
                # Process uploaded image
                image = Image.open(uploaded_file).convert('RGB')
                image_name = uploaded_file.name
            elif 'selected_sample' in locals():
                # Process sample image
                image = sample_image
                image_name = selected_sample.name
            else:
                st.error("Please upload an image or select a sample image!")
                return
            
            if question.strip():
                # Load model and get answer
                model = load_model()
                
                with st.spinner("Analyzing image and generating answer..."):
                    start_time = time.time()
                    result = model.answer_question(
                        image, 
                        question,
                        max_length=max_length,
                        num_beams=num_beams,
                        temperature=temperature
                    )
                    processing_time = time.time() - start_time
                
                # Display results
                st.success("Answer generated!")
                
                # Show answer
                st.subheader("🎯 Answer")
                st.write(f"**Question:** {result.question}")
                st.write(f"**Answer:** {result.answer}")
                
                # Show metadata
                with st.expander("📊 Details"):
                    st.write(f"**Model:** {result.model_name}")
                    st.write(f"**Processing Time:** {processing_time:.2f} seconds")
                    st.write(f"**Device:** {model.device}")
                
                # Display image with Q&A
                display_image_with_question(image, question, result.answer)
                
                # Add to history
                st.session_state.history.append({
                    'image_name': image_name,
                    'question': question,
                    'answer': result.answer,
                    'timestamp': time.time(),
                    'model': result.model_name
                })
            else:
                st.error("Please enter a question!")
    
    # History section
    if st.session_state.history:
        st.header("📚 History")
        
        for i, entry in enumerate(reversed(st.session_state.history[-10:])):  # Show last 10
            with st.expander(f"Q&A #{len(st.session_state.history) - i}: {entry['question'][:50]}..."):
                st.write(f"**Image:** {entry['image_name']}")
                st.write(f"**Question:** {entry['question']}")
                st.write(f"**Answer:** {entry['answer']}")
                st.write(f"**Model:** {entry['model']}")
                st.write(f"**Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry['timestamp']))}")
        
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using [Streamlit](https://streamlit.io), "
        "[Hugging Face Transformers](https://huggingface.co/transformers), "
        "and [BLIP](https://huggingface.co/Salesforce/blip-vqa-base)"
    )


if __name__ == "__main__":
    main()
