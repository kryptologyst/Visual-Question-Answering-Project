"""
Command-line interface for Visual Question Answering.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List
import json

from src.vqa_model import VQAModel, VQAResult
from src.config import ConfigManager


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visual Question Answering CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python cli.py --image path/to/image.jpg --question "What is in this image?"
  
  # Batch processing
  python cli.py --batch images/ questions.txt --output results.json
  
  # Interactive mode
  python cli.py --interactive
  
  # Use specific model
  python cli.py --model Salesforce/blip-vqa-capfilt-large --image image.jpg --question "What colors do you see?"
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", "-i",
        type=str,
        help="Path to input image"
    )
    input_group.add_argument(
        "--batch", "-b",
        nargs=2,
        metavar=("IMAGES_DIR", "QUESTIONS_FILE"),
        help="Process multiple images with questions from file"
    )
    input_group.add_argument(
        "--interactive", "-int",
        action="store_true",
        help="Run in interactive mode"
    )
    
    # Question options
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Question to ask about the image"
    )
    
    # Model options
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Salesforce/blip-vqa-base",
        help="Model to use for VQA"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Use Hugging Face pipeline API"
    )
    
    # Generation options
    parser.add_argument(
        "--max-length",
        type=int,
        default=50,
        help="Maximum length of generated answer"
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=5,
        help="Number of beams for beam search"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (JSON format)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential output"
    )
    
    return parser.parse_args()


def process_single_image(
    model: VQAModel,
    image_path: str,
    question: str,
    args: argparse.Namespace
) -> VQAResult:
    """Process a single image with a question."""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not question:
        raise ValueError("Question is required for single image processing")
    
    result = model.answer_question(
        image_path,
        question,
        max_length=args.max_length,
        num_beams=args.num_beams,
        temperature=args.temperature
    )
    
    return result


def process_batch(
    model: VQAModel,
    images_dir: str,
    questions_file: str,
    args: argparse.Namespace
) -> List[VQAResult]:
    """Process multiple images with questions from file."""
    images_path = Path(images_dir)
    questions_path = Path(questions_file)
    
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    
    # Load questions
    with open(questions_path, 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in images_path.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    if len(image_files) != len(questions):
        raise ValueError(
            f"Number of images ({len(image_files)}) doesn't match "
            f"number of questions ({len(questions)})"
        )
    
    results = []
    for image_file, question in zip(image_files, questions):
        if not args.quiet:
            print(f"Processing {image_file.name}...")
        
        result = model.answer_question(
            image_file,
            question,
            max_length=args.max_length,
            num_beams=args.num_beams,
            temperature=args.temperature
        )
        results.append(result)
    
    return results


def interactive_mode(model: VQAModel, args: argparse.Namespace) -> None:
    """Run in interactive mode."""
    print("🔍 Visual Question Answering - Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'help' for available commands")
    print("-" * 50)
    
    while True:
        try:
            # Get image path
            image_path = input("\n📸 Enter image path: ").strip()
            if image_path.lower() in ['quit', 'exit']:
                break
            
            if not Path(image_path).exists():
                print(f"❌ Image not found: {image_path}")
                continue
            
            # Get question
            question = input("❓ Enter your question: ").strip()
            if question.lower() in ['quit', 'exit']:
                break
            
            if not question:
                print("❌ Please enter a question")
                continue
            
            # Process
            print("🔄 Processing...")
            result = model.answer_question(
                image_path,
                question,
                max_length=args.max_length,
                num_beams=args.num_beams,
                temperature=args.temperature
            )
            
            # Display result
            print(f"\n🎯 Answer: {result.answer}")
            if args.verbose:
                print(f"📊 Model: {result.model_name}")
                print(f"⏱️  Processing time: {result.processing_time:.2f}s")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def save_results(results: List[VQAResult], output_path: str) -> None:
    """Save results to JSON file."""
    output_data = []
    for result in results:
        output_data.append({
            'question': result.question,
            'answer': result.answer,
            'model_name': result.model_name,
            'processing_time': result.processing_time,
            'confidence': result.confidence
        })
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Results saved to {output_path}")


def main() -> None:
    """Main CLI function."""
    args = parse_arguments()
    
    # Setup logging
    if args.verbose:
        setup_logging("DEBUG")
    elif args.quiet:
        setup_logging("WARNING")
    else:
        setup_logging("INFO")
    
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize model
        if not args.quiet:
            print(f"🔄 Loading model: {args.model}")
        
        device = None if args.device == "auto" else args.device
        model = VQAModel(
            model_name=args.model,
            device=device,
            use_pipeline=args.pipeline
        )
        
        if not args.quiet:
            model_info = model.get_model_info()
            print(f"✅ Model loaded successfully")
            print(f"📱 Device: {model_info['device']}")
            print(f"🤖 Model type: {model_info['model_type']}")
        
        # Process based on mode
        if args.interactive:
            interactive_mode(model, args)
        elif args.batch:
            results = process_batch(model, args.batch[0], args.batch[1], args)
            if not args.quiet:
                print(f"✅ Processed {len(results)} images")
        else:
            result = process_single_image(model, args.image, args.question, args)
            results = [result]
            
            if not args.quiet:
                print(f"🎯 Answer: {result.answer}")
                if args.verbose:
                    print(f"📊 Model: {result.model_name}")
                    print(f"⏱️  Processing time: {result.processing_time:.2f}s")
        
        # Save results if requested
        if args.output and not args.interactive:
            save_results(results, args.output)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
