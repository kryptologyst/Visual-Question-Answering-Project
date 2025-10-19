"""
Visual Question Answering (VQA) Model Implementation

This module provides a modern, production-ready implementation of Visual Question Answering
using state-of-the-art transformer models from Hugging Face.
"""

import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Tuple
import warnings

import torch
from PIL import Image
from transformers import (
    BlipProcessor, 
    BlipForQuestionAnswering,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    pipeline
)
import numpy as np
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VQAResult:
    """Data class to hold VQA results with confidence scores."""
    question: str
    answer: str
    confidence: Optional[float] = None
    model_name: str = ""
    processing_time: Optional[float] = None


class VQAModel:
    """
    Modern Visual Question Answering model wrapper with multiple backend support.
    
    Supports both BLIP and BLIP-2 models with automatic fallback and error handling.
    """
    
    def __init__(
        self, 
        model_name: str = "Salesforce/blip-vqa-base",
        device: Optional[str] = None,
        use_pipeline: bool = False
    ):
        """
        Initialize the VQA model.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            use_pipeline: Whether to use Hugging Face pipeline API
        """
        self.model_name = model_name
        self.device = device or self._get_optimal_device()
        self.use_pipeline = use_pipeline
        
        logger.info(f"Initializing VQA model: {model_name} on {self.device}")
        
        try:
            if use_pipeline:
                self._load_pipeline_model()
            else:
                self._load_direct_model()
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self._load_fallback_model()
    
    def _get_optimal_device(self) -> str:
        """Determine the best available device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _load_pipeline_model(self) -> None:
        """Load model using Hugging Face pipeline API."""
        self.pipeline = pipeline(
            "visual-question-answering",
            model=self.model_name,
            device=self.device
        )
        self.processor = None
        self.model = None
    
    def _load_direct_model(self) -> None:
        """Load model components directly for more control."""
        if "blip2" in self.model_name.lower():
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            )
        else:
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForQuestionAnswering.from_pretrained(self.model_name)
        
        self.model.to(self.device)
        self.pipeline = None
    
    def _load_fallback_model(self) -> None:
        """Load a fallback model if the primary model fails."""
        logger.warning("Loading fallback model: Salesforce/blip-vqa-base")
        try:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            self.model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
            self.model.to(self.device)
            self.pipeline = None
            self.model_name = "Salesforce/blip-vqa-base"
        except Exception as e:
            logger.error(f"Failed to load fallback model: {e}")
            raise RuntimeError("Unable to load any VQA model")
    
    def answer_question(
        self, 
        image: Union[str, Path, Image.Image], 
        question: str,
        max_length: int = 50,
        num_beams: int = 5,
        temperature: float = 1.0
    ) -> VQAResult:
        """
        Answer a question about an image.
        
        Args:
            image: Image path, PIL Image, or image array
            question: Natural language question about the image
            max_length: Maximum length of generated answer
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            
        Returns:
            VQAResult object containing the answer and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Load and preprocess image
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Generate answer
            if self.pipeline:
                result = self._answer_with_pipeline(image, question)
            else:
                result = self._answer_with_direct_model(
                    image, question, max_length, num_beams, temperature
                )
            
            processing_time = time.time() - start_time
            
            return VQAResult(
                question=question,
                answer=result,
                model_name=self.model_name,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return VQAResult(
                question=question,
                answer=f"Error: {str(e)}",
                model_name=self.model_name,
                processing_time=time.time() - start_time
            )
    
    def _answer_with_pipeline(self, image: Image.Image, question: str) -> str:
        """Answer using Hugging Face pipeline."""
        result = self.pipeline(image, question)
        return result[0]['answer']
    
    def _answer_with_direct_model(
        self, 
        image: Image.Image, 
        question: str,
        max_length: int,
        num_beams: int,
        temperature: float
    ) -> str:
        """Answer using direct model inference."""
        # Preprocess inputs
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        
        # Generate answer
        with torch.no_grad():
            if hasattr(self.model, 'generate'):
                # BLIP-2 style generation
                output = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=temperature > 0
                )
            else:
                # BLIP style generation
                output = self.model.generate(**inputs)
        
        # Decode answer
        answer = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Clean up answer (remove question if present)
        if question.lower() in answer.lower():
            answer = answer.replace(question, "").strip()
        
        return answer
    
    def batch_answer(
        self, 
        images: List[Union[str, Path, Image.Image]], 
        questions: List[str]
    ) -> List[VQAResult]:
        """
        Answer multiple questions about multiple images.
        
        Args:
            images: List of images
            questions: List of questions
            
        Returns:
            List of VQAResult objects
        """
        if len(images) != len(questions):
            raise ValueError("Number of images must match number of questions")
        
        results = []
        for image, question in zip(images, questions):
            result = self.answer_question(image, question)
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "use_pipeline": self.use_pipeline,
            "model_type": "BLIP-2" if "blip2" in self.model_name.lower() else "BLIP"
        }


class VQADataset:
    """
    Synthetic dataset generator for VQA testing and demonstration.
    """
    
    def __init__(self, num_samples: int = 10):
        """
        Initialize synthetic dataset.
        
        Args:
            num_samples: Number of synthetic samples to generate
        """
        self.num_samples = num_samples
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[Dict[str, Any]]:
        """Generate synthetic VQA samples."""
        import numpy as np
        
        samples = []
        
        # Define some synthetic scenarios
        scenarios = [
            {
                "image_description": "A red car parked in front of a house",
                "questions": [
                    "What color is the car?",
                    "Where is the car parked?",
                    "What is in the background?"
                ],
                "answers": ["red", "in front of a house", "a house"]
            },
            {
                "image_description": "A dog playing in a park with trees",
                "questions": [
                    "What animal is in the image?",
                    "Where is the dog?",
                    "What is the environment like?"
                ],
                "answers": ["dog", "in a park", "outdoor with trees"]
            },
            {
                "image_description": "A person reading a book at a cafe",
                "questions": [
                    "What is the person doing?",
                    "Where is this taking place?",
                    "What object is the person holding?"
                ],
                "answers": ["reading", "at a cafe", "a book"]
            }
        ]
        
        for i in range(self.num_samples):
            scenario = scenarios[i % len(scenarios)]
            sample = {
                "id": i,
                "image_description": scenario["image_description"],
                "questions": scenario["questions"],
                "ground_truth_answers": scenario["answers"],
                "image_path": f"synthetic_image_{i}.jpg"  # Placeholder
            }
            samples.append(sample)
        
        return samples
    
    def get_samples(self) -> List[Dict[str, Any]]:
        """Get all generated samples."""
        return self.samples
    
    def get_sample(self, index: int) -> Dict[str, Any]:
        """Get a specific sample by index."""
        if 0 <= index < len(self.samples):
            return self.samples[index]
        raise IndexError(f"Sample index {index} out of range")


def create_sample_images(output_dir: Path, num_images: int = 5) -> List[Path]:
    """
    Create sample images for testing (placeholder implementation).
    
    Args:
        output_dir: Directory to save images
        num_images: Number of images to create
        
    Returns:
        List of created image paths
    """
    output_dir.mkdir(exist_ok=True)
    
    # This is a placeholder - in a real implementation, you would:
    # 1. Download images from a dataset
    # 2. Generate synthetic images
    # 3. Use a pre-existing image collection
    
    image_paths = []
    for i in range(num_images):
        # Create a simple colored rectangle as placeholder
        from PIL import Image, ImageDraw
        
        img = Image.new('RGB', (224, 224), color=(i * 50, 100, 200))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 174, 174], fill=(255, 255, 255))
        draw.text((100, 100), f"Sample {i+1}", fill=(0, 0, 0))
        
        image_path = output_dir / f"sample_image_{i+1}.jpg"
        img.save(image_path)
        image_paths.append(image_path)
    
    return image_paths
