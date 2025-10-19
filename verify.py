#!/usr/bin/env python3
"""
Verification script for Visual Question Answering project.
This script checks that all components are properly structured.
"""

import sys
from pathlib import Path


def check_file_exists(file_path: str, description: str) -> bool:
    """Check if a file exists and print status."""
    path = Path(file_path)
    if path.exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (MISSING)")
        return False


def check_directory_exists(dir_path: str, description: str) -> bool:
    """Check if a directory exists and print status."""
    path = Path(dir_path)
    if path.exists() and path.is_dir():
        print(f"✅ {description}: {dir_path}")
        return True
    else:
        print(f"❌ {description}: {dir_path} (MISSING)")
        return False


def main():
    """Main verification function."""
    print("🔍 Visual Question Answering - Project Verification")
    print("=" * 60)
    
    # Check Python version
    print(f"🐍 Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠️  Warning: Python 3.8+ recommended")
    
    print("\n📁 Project Structure Verification:")
    
    # Core source files
    core_files = [
        ("src/__init__.py", "Source package init"),
        ("src/vqa_model.py", "Core VQA model"),
        ("src/config.py", "Configuration management"),
        ("web_app/__init__.py", "Web app package init"),
        ("web_app/app.py", "Streamlit web interface"),
        ("tests/test_vqa.py", "Test suite"),
        ("cli.py", "Command-line interface"),
        ("example.py", "Example usage script"),
        ("setup.py", "Setup script"),
    ]
    
    core_success = 0
    for file_path, description in core_files:
        if check_file_exists(file_path, description):
            core_success += 1
    
    # Configuration and project files
    config_files = [
        ("config/config.yaml", "Configuration file"),
        ("requirements.txt", "Dependencies file"),
        (".gitignore", "Git ignore file"),
        ("README.md", "Documentation"),
        ("MODERNIZATION_SUMMARY.md", "Modernization summary"),
    ]
    
    config_success = 0
    for file_path, description in config_files:
        if check_file_exists(file_path, description):
            config_success += 1
    
    # Directories
    directories = [
        ("src", "Source code directory"),
        ("web_app", "Web application directory"),
        ("tests", "Test directory"),
        ("config", "Configuration directory"),
        ("data", "Data directory"),
        ("models", "Models directory"),
    ]
    
    dir_success = 0
    for dir_path, description in directories:
        if check_directory_exists(dir_path, description):
            dir_success += 1
    
    # Summary
    total_files = len(core_files) + len(config_files)
    total_dirs = len(directories)
    total_success = core_success + config_success + dir_success
    
    print(f"\n📊 Verification Summary:")
    print(f"   Core files: {core_success}/{len(core_files)}")
    print(f"   Config files: {config_success}/{len(config_files)}")
    print(f"   Directories: {dir_success}/{len(directories)}")
    print(f"   Total: {total_success}/{total_files + total_dirs}")
    
    if total_success == total_files + total_dirs:
        print("\n🎉 All components verified successfully!")
        print("\n📋 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run setup: python setup.py")
        print("3. Test example: python example.py")
        print("4. Launch web app: streamlit run web_app/app.py")
        return True
    else:
        print(f"\n⚠️  {total_files + total_dirs - total_success} components missing")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
