"""
Test suite for Visual Question Answering project.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image

# Import modules to test
from src.vqa_model import VQAModel, VQAResult, VQADataset, create_sample_images
from src.config import ConfigManager, ModelConfig, AppConfig, UIConfig


class TestVQAResult(unittest.TestCase):
    """Test VQAResult data class."""
    
    def test_vqa_result_creation(self):
        """Test VQAResult creation with all parameters."""
        result = VQAResult(
            question="What is this?",
            answer="A cat",
            confidence=0.95,
            model_name="test-model",
            processing_time=1.5
        )
        
        self.assertEqual(result.question, "What is this?")
        self.assertEqual(result.answer, "A cat")
        self.assertEqual(result.confidence, 0.95)
        self.assertEqual(result.model_name, "test-model")
        self.assertEqual(result.processing_time, 1.5)
    
    def test_vqa_result_minimal(self):
        """Test VQAResult creation with minimal parameters."""
        result = VQAResult(question="Test?", answer="Test answer")
        
        self.assertEqual(result.question, "Test?")
        self.assertEqual(result.answer, "Test answer")
        self.assertIsNone(result.confidence)
        self.assertEqual(result.model_name, "")
        self.assertIsNone(result.processing_time)


class TestVQADataset(unittest.TestCase):
    """Test VQADataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dataset = VQADataset(num_samples=3)
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        self.assertEqual(len(self.dataset.samples), 3)
        self.assertEqual(self.dataset.num_samples, 3)
    
    def test_get_samples(self):
        """Test getting all samples."""
        samples = self.dataset.get_samples()
        self.assertEqual(len(samples), 3)
        
        # Check sample structure
        sample = samples[0]
        self.assertIn('id', sample)
        self.assertIn('image_description', sample)
        self.assertIn('questions', sample)
        self.assertIn('ground_truth_answers', sample)
        self.assertIn('image_path', sample)
    
    def test_get_sample(self):
        """Test getting specific sample."""
        sample = self.dataset.get_sample(0)
        self.assertEqual(sample['id'], 0)
        
        # Test invalid index
        with self.assertRaises(IndexError):
            self.dataset.get_sample(10)


class TestVQAModel(unittest.TestCase):
    """Test VQAModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = Path(self.temp_dir) / "test_image.jpg"
        
        # Create a test image
        test_image = Image.new('RGB', (224, 224), color='red')
        test_image.save(self.test_image_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.vqa_model.BlipProcessor')
    @patch('src.vqa_model.BlipForQuestionAnswering')
    def test_model_initialization(self, mock_model_class, mock_processor_class):
        """Test model initialization."""
        # Mock the model and processor
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        model = VQAModel(model_name="test-model")
        
        self.assertEqual(model.model_name, "test-model")
        self.assertIsNotNone(model.device)
    
    def test_get_optimal_device(self):
        """Test device selection logic."""
        model = VQAModel.__new__(VQAModel)  # Create without calling __init__
        
        # Test CPU fallback
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = model._get_optimal_device()
                self.assertEqual(device, "cpu")
    
    def test_create_sample_images(self):
        """Test sample image creation."""
        output_dir = Path(self.temp_dir) / "samples"
        image_paths = create_sample_images(output_dir, 2)
        
        self.assertEqual(len(image_paths), 2)
        self.assertTrue(output_dir.exists())
        
        for path in image_paths:
            self.assertTrue(path.exists())
            self.assertTrue(path.suffix == '.jpg')


class TestConfigManager(unittest.TestCase):
    """Test ConfigManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_config_manager_initialization(self):
        """Test ConfigManager initialization."""
        config_manager = ConfigManager()
        
        self.assertIsInstance(config_manager.model_config, ModelConfig)
        self.assertIsInstance(config_manager.app_config, AppConfig)
        self.assertIsInstance(config_manager.ui_config, UIConfig)
    
    def test_config_save_and_load(self):
        """Test configuration save and load."""
        config_manager = ConfigManager(self.config_path)
        
        # Modify configuration
        config_manager.model_config.model_name = "test-model"
        config_manager.app_config.debug = True
        
        # Save configuration
        config_manager.save_config()
        
        # Create new config manager and load
        new_config_manager = ConfigManager(self.config_path)
        
        self.assertEqual(new_config_manager.model_config.model_name, "test-model")
        self.assertTrue(new_config_manager.app_config.debug)


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig data class."""
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        
        self.assertEqual(config.model_name, "Salesforce/blip-vqa-base")
        self.assertIsNone(config.device)
        self.assertFalse(config.use_pipeline)
        self.assertEqual(config.max_length, 50)
        self.assertEqual(config.num_beams, 5)
        self.assertEqual(config.temperature, 1.0)
    
    def test_model_config_custom(self):
        """Test ModelConfig with custom values."""
        config = ModelConfig(
            model_name="custom-model",
            device="cuda",
            use_pipeline=True,
            max_length=100,
            num_beams=10,
            temperature=0.5
        )
        
        self.assertEqual(config.model_name, "custom-model")
        self.assertEqual(config.device, "cuda")
        self.assertTrue(config.use_pipeline)
        self.assertEqual(config.max_length, 100)
        self.assertEqual(config.num_beams, 10)
        self.assertEqual(config.temperature, 0.5)


class TestAppConfig(unittest.TestCase):
    """Test AppConfig data class."""
    
    def test_app_config_defaults(self):
        """Test AppConfig default values."""
        config = AppConfig()
        
        self.assertFalse(config.debug)
        self.assertEqual(config.log_level, "INFO")
        self.assertEqual(config.data_dir, "data")
        self.assertEqual(config.models_dir, "models")
        self.assertEqual(config.output_dir, "outputs")


class TestUIConfig(unittest.TestCase):
    """Test UIConfig data class."""
    
    def test_ui_config_defaults(self):
        """Test UIConfig default values."""
        config = UIConfig()
        
        self.assertEqual(config.title, "Visual Question Answering Demo")
        self.assertEqual(config.theme, "light")
        self.assertEqual(config.sidebar_width, 300)
        self.assertEqual(config.max_image_size, 512)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = Path(self.temp_dir) / "test_image.jpg"
        
        # Create a test image
        test_image = Image.new('RGB', (224, 224), color='blue')
        test_image.save(self.test_image_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('src.vqa_model.BlipProcessor')
    @patch('src.vqa_model.BlipForQuestionAnswering')
    def test_end_to_end_workflow(self, mock_model_class, mock_processor_class):
        """Test end-to-end workflow."""
        # Mock the model and processor
        mock_processor = Mock()
        mock_model = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock the processor methods
        mock_processor.decode.return_value = "A blue image"
        mock_model.generate.return_value = Mock()
        
        # Create model and test
        model = VQAModel(model_name="test-model")
        result = model.answer_question(self.test_image_path, "What color is this?")
        
        self.assertIsInstance(result, VQAResult)
        self.assertEqual(result.question, "What color is this?")
        self.assertEqual(result.answer, "A blue image")


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestVQAResult,
        TestVQADataset,
        TestVQAModel,
        TestConfigManager,
        TestModelConfig,
        TestAppConfig,
        TestUIConfig,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
