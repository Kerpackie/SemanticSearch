#!/usr/bin/env python3
"""
Validation script for API setup.
Checks that all required files and dependencies are present.
"""

import os
import sys

def check_file(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {filepath}")
        return False

def check_module(module_name):
    """Check if a Python module is installed."""
    try:
        __import__(module_name)
        print(f"✅ Module installed: {module_name}")
        return True
    except ImportError:
        print(f"❌ Module NOT installed: {module_name}")
        return False

def main():
    print("=" * 60)
    print("FFNN Recommendation API - Setup Validation")
    print("=" * 60)
    print()

    all_ok = True

    # Check model files
    print("Checking Model Files:")
    print("-" * 60)
    all_ok &= check_file("models/ffnn_20251016_154913/model.keras", "Model file")
    all_ok &= check_file("models/ffnn_20251016_154913/preprocessor_nn.joblib", "Preprocessor")
    all_ok &= check_file("models/ffnn_20251016_154913/meta.json", "Metadata")
    print()

    # Check data files
    print("Checking Data Files:")
    print("-" * 60)
    all_ok &= check_file("transactions_train.csv", "Transactions")
    all_ok &= check_file("articles.csv", "Articles")
    print()

    # Check API files
    print("Checking API Files:")
    print("-" * 60)
    all_ok &= check_file("api.py", "API server")
    all_ok &= check_file("test_api_client.py", "Test client")
    print()

    # Check Python dependencies
    print("Checking Python Dependencies:")
    print("-" * 60)
    modules = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "numpy",
        "pandas",
        "sklearn",
        "tensorflow",
        "joblib",
        "requests"
    ]

    for module in modules:
        all_ok &= check_module(module)
    print()

    # Summary
    print("=" * 60)
    if all_ok:
        print("✅ ALL CHECKS PASSED - API is ready to run!")
        print()
        print("To start the API server, run:")
        print("  python api.py")
        print()
        print("Or use the startup script:")
        print("  ./start_api.sh")
        print()
        print("Then test it with:")
        print("  python test_api_client.py")
        print()
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please fix the issues above")
        print()
        print("To install missing Python packages, run:")
        print("  pip install -r requirements_api.txt")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())

