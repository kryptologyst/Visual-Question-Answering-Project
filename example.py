#!/usr/bin/env python3
"""
Simple example script demonstrating Visual Question Answering.

This script shows how to use the VQA model programmatically.
"""

import logging
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from src.vqa_model import VQAModel, create_sample_images
from src.config import ConfigManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    print("🔍 Visual Question Answering - Example Script")
    print("=" * 50)
    
    # Initialize configuration
    config_manager = ConfigManager()
    model_config = config_manager.get_model_config()
    
    print(f"📱 Using model: {model_config.model_name}")
    print(f"🎯 Device: {model_config.device or 'auto-detect'}")
    
    # Create sample images if they don't exist
    sample_dir = Path("data/samples")
    if not sample_dir.exists() or not list(sample_dir.glob("*.jpg")):
        print("🎨 Creating sample images...")
        sample_images = create_sample_images(sample_dir, 3)
        print(f"✅ Created {len(sample_images)} sample images")
    else:
        sample_images = list(sample_dir.glob("*.jpg"))
        print(f"📸 Found {len(sample_images)} existing sample images")
    
    # Initialize VQA model
    print("🔄 Loading VQA model...")
    model = VQAModel(
        model_name=model_config.model_name,
        device=model_config.device,
        use_pipeline=model_config.use_pipeline
    )
    
    model_info = model.get_model_info()
    print(f"✅ Model loaded successfully!")
    print(f"🤖 Model type: {model_info['model_type']}")
    print(f"📱 Running on: {model_info['device']}")
    
    # Example questions
    questions = [
        "What is in this image?",
        "What colors do you see?",
        "Describe the scene"
    ]
    
    # Process each sample image
    for i, image_path in enumerate(sample_images[:2]):  # Process first 2 images
        print(f"\n📸 Processing image {i+1}: {image_path.name}")
        
        # Load image
        image = Image.open(image_path)
        
        # Ask questions about the image
        for j, question in enumerate(questions):
            print(f"\n❓ Question {j+1}: {question}")
            
            # Get answer
            result = model.answer_question(
                image, 
                question,
                max_length=model_config.max_length,
                num_beams=model_config.num_beams,
                temperature=model_config.temperature
            )
            
            print(f"🎯 Answer: {result.answer}")
            print(f"⏱️  Processing time: {result.processing_time:.2f}s")
            
            # Display image with Q&A
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title(f"Image: {image_path.name}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.text(0.1, 0.7, f"Q: {question}", fontsize=12, weight='bold')
            plt.text(0.1, 0.5, f"A: {result.answer}", fontsize=12)
            plt.text(0.1, 0.3, f"Time: {result.processing_time:.2f}s", fontsize=10)
            plt.text(0.1, 0.1, f"Model: {result.model_name}", fontsize=10)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            plt.title("Q&A Result")
            
            plt.tight_layout()
            plt.show()
    
    print("\n🎉 Example completed successfully!")
    print("\n💡 Try running the web interface:")
    print("   streamlit run web_app/app.py")
    print("\n💡 Or use the CLI:")
    print("   python cli.py --interactive")


if __name__ == "__main__":
    main()
