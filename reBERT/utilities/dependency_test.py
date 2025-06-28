# dependency_test.py - Comprehensive dependency and functionality test for src project

import sys
import tomllib
import re
from pathlib import Path
from typing import Dict, Any
import warnings

warnings.filterwarnings("ignore")  # Suppress minor warnings for cleaner output


def parse_pyproject_dependencies() -> Dict[str, str]:
    """Parse expected versions from pyproject.toml"""

    # Look for pyproject.toml in current directory or parent directory
    pyproject_path = None
    for path in [Path("../../pyproject.toml"), Path("../../pyproject.toml")]:
        if path.exists():
            pyproject_path = path
            break

    if not pyproject_path:
        print("❌ Could not find pyproject.toml file")
        return {}

    try:
        with open(pyproject_path, "rb") as f:
            pyproject_data = tomllib.load(f)

        dependencies = pyproject_data.get("project", {}).get("dependencies", [])

        expected_versions = {}
        for dep in dependencies:
            # Parse dependency strings like "numpy==2.3.0" or "torch==2.6.0+cu124"
            if "==" in dep:
                name, version = dep.split("==", 1)
                # Handle cases like "python-dotenv==1.0.1" -> map to "dotenv" import name
                import_name = name.strip().replace('"', "").replace("'", "")

                # Map package names to import names
                name_mapping = {"python-dotenv": "dotenv", "scikit-learn": "sklearn"}

                import_name = name_mapping.get(import_name, import_name)
                version = version.strip().replace('"', "").replace("'", "").rstrip(",")

                expected_versions[import_name] = version

        print(f"📁 Loaded dependencies from: {pyproject_path}")
        print(f"📦 Found {len(expected_versions)} dependencies with version pins")

        return expected_versions

    except Exception as e:
        print(f"❌ Error parsing pyproject.toml: {e}")
        return {}


def check_version_match(expected: str, actual: str) -> str:
    """Compare expected vs actual version and return status emoji"""
    if expected == "Not specified":
        return "✅"

    # Extract major.minor from both versions for comparison
    try:
        expected_parts = expected.split(".")
        actual_parts = actual.split(".")

        # Compare major.minor (ignore patch versions and suffixes like +cu124)
        if len(expected_parts) >= 2 and len(actual_parts) >= 2:
            expected_major_minor = f"{expected_parts[0]}.{expected_parts[1]}"
            actual_major_minor = f"{actual_parts[0]}.{actual_parts[1]}"

            if actual_major_minor == expected_major_minor:
                return "✅"
            else:
                return "⚠️"
    except:
        pass

    # Fallback: exact match
    return "✅" if actual == expected else "⚠️"


def test_package(
    package_name: str, import_name: str, expected_versions: dict, results: dict
) -> bool:
    """Generic function to test a package import and version"""
    try:
        module = __import__(import_name)
        expected = expected_versions.get(package_name, "Not specified")
        actual = getattr(module, "__version__", "Unknown")
        status = check_version_match(expected, actual)

        results[package_name] = {"expected": expected, "actual": actual, "status": status}
        print(f"{package_name.title()}: Expected {expected}, Got {actual} {status}")
        return True

    except Exception as e:
        results[package_name] = {"error": str(e)}
        print(f"❌ {package_name.title()} failed: {e}")
        return False


def test_versions_and_functionality():
    """Test all dependencies with expected vs actual versions and basic functionality"""

    print("🧪 src Project Dependency & Functionality Test")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print("=" * 60)

    # Parse expected versions from pyproject.toml
    expected_versions = parse_pyproject_dependencies()

    if not expected_versions:
        print("❌ No dependencies found in pyproject.toml - cannot run tests")
        return

    results = {}

    # Core Data Science Packages
    print("\n📊 CORE DATA SCIENCE PACKAGES")
    print("-" * 40)

    # NumPy
    if test_package("numpy", "numpy", expected_versions, results):
        try:
            import numpy as np

            arr = np.array([1, 2, 3, 4, 5])
            assert arr.mean() == 3.0, "NumPy basic operations failed"
            print("  ✅ NumPy array operations working")
        except Exception as e:
            print(f"  ❌ NumPy functionality test failed: {e}")

    # Pandas
    if test_package("pandas", "pandas", expected_versions, results):
        try:
            import pandas as pd

            df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
            assert len(df) == 3, "Pandas DataFrame creation failed"
            print("  ✅ Pandas DataFrame operations working")
        except Exception as e:
            print(f"  ❌ Pandas functionality test failed: {e}")

    # Matplotlib
    if test_package("matplotlib", "matplotlib", expected_versions, results):
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 2])
            plt.close(fig)  # Don't show plot in test
            print("  ✅ Matplotlib plotting working")
        except Exception as e:
            print(f"  ❌ Matplotlib functionality test failed: {e}")

    # Seaborn
    if test_package("seaborn", "seaborn", expected_versions, results):
        print("  ✅ Seaborn styling available")

    # Scikit-learn
    if test_package("sklearn", "sklearn", expected_versions, results):
        try:
            from sklearn.datasets import make_classification

            X, y = make_classification(n_samples=100, n_features=4, random_state=42)
            assert X.shape == (100, 4), "Sklearn dataset generation failed"
            print("  ✅ Scikit-learn dataset generation working")
        except Exception as e:
            print(f"  ❌ Scikit-learn functionality test failed: {e}")

    # AI/ML Packages
    print("\n🤖 AI/ML PACKAGES")
    print("-" * 40)

    # PyTorch
    if test_package("torch", "torch", expected_versions, results):
        try:
            import torch

            print(f"  CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA version: {torch.version.cuda}")

            # Test tensor operations
            x = torch.randn(3, 4)
            y = torch.randn(4, 5)
            z = torch.mm(x, y)
            assert z.shape == (3, 5), "PyTorch tensor operations failed"
            print("  ✅ PyTorch tensor operations working")

            # Test GPU if available
            if torch.cuda.is_available():
                x_gpu = x.cuda()
                assert x_gpu.is_cuda, "GPU tensor transfer failed"
                print("  ✅ PyTorch GPU operations working")

        except Exception as e:
            print(f"  ❌ PyTorch functionality test failed: {e}")

    # Transformers
    if test_package("transformers", "transformers", expected_versions, results):
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            tokens = tokenizer("Hello world!")
            assert "input_ids" in tokens, "Transformers tokenization failed"
            print("  ✅ Transformers tokenization working")
        except Exception as e:
            print(f"  ❌ Transformers functionality test failed: {e}")

    # Accelerate
    if test_package("accelerate", "accelerate", expected_versions, results):
        print("  ✅ Accelerate available for distributed training")

    # Datasets
    if test_package("datasets", "datasets", expected_versions, results):
        print("  ✅ HuggingFace datasets available")

    # Development Environment
    print("\n💻 DEVELOPMENT ENVIRONMENT")
    print("-" * 40)

    # IPython
    if test_package("ipython", "IPython", expected_versions, results):
        print("  ✅ Interactive Python shell available")

    # JupyterLab
    if test_package("jupyterlab", "jupyterlab", expected_versions, results):
        print("  ✅ JupyterLab environment available")

    # Notebook
    if test_package("notebook", "notebook", expected_versions, results):
        print("  ✅ Classic Jupyter notebook available")

    # Utilities and Tools
    print("\n🛠️ UTILITIES AND TOOLS")
    print("-" * 40)

    # Loguru
    if test_package("loguru", "loguru", expected_versions, results):
        try:
            from loguru import logger

            logger.info("Test log message")
            print("  ✅ Loguru logging working")
        except Exception as e:
            print(f"  ❌ Loguru functionality test failed: {e}")

    # Python-dotenv (maps to 'dotenv' import)
    if test_package("dotenv", "dotenv", expected_versions, results):
        print("  ✅ Environment variable loading available")

    # Tqdm
    if test_package("tqdm", "tqdm", expected_versions, results):
        try:
            from tqdm import tqdm as progress_bar

            for i in progress_bar(range(3), desc="Testing tqdm"):
                pass
            print("  ✅ Progress bars working")
        except Exception as e:
            print(f"  ❌ Tqdm functionality test failed: {e}")

    # Typer
    if test_package("typer", "typer", expected_versions, results):
        print("  ✅ CLI framework available")

    # Ruff
    if test_package("ruff", "ruff", expected_versions, results):
        print("  ✅ Code linting and formatting available")

    # Documentation
    print("\n📚 DOCUMENTATION")
    print("-" * 40)

    # MkDocs
    if test_package("mkdocs", "mkdocs", expected_versions, results):
        print("  ✅ Documentation generator available")

    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)

    success_count = sum(1 for r in results.values() if "status" in r and r["status"] == "✅")
    warning_count = sum(1 for r in results.values() if "status" in r and r["status"] == "⚠️")
    error_count = sum(1 for r in results.values() if "error" in r)
    total_count = len(results)

    print(f"✅ Success: {success_count}/{total_count}")
    print(f"⚠️  Warnings: {warning_count}/{total_count}")
    print(f"❌ Errors: {error_count}/{total_count}")

    if error_count == 0 and warning_count <= 2:
        print("\n🎉 Environment is ready for src development!")
    elif error_count == 0:
        print("\n✅ Environment is mostly ready - check version warnings")
    else:
        print("\n⚠️  Some dependencies failed - check errors above")

    return results

if __name__ == "__main__":
    test_versions_and_functionality()