"""
Setup script for Customer Churn Prediction Project
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="customer-churn-prediction",
    version="1.0.0",
    author="Data Science Professional",
    author_email="your.email@example.com",
    description="A comprehensive machine learning solution for customer churn prediction and financial impact analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/customer-churn-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "jupyter>=1.0",
            "notebook>=6.0",
        ],
        "advanced": [
            "xgboost>=1.6.0",
            "lightgbm>=3.3.0",
            "shap>=0.40.0",
            "optuna>=3.0.0",
            "plotly>=5.0.0",
            "streamlit>=1.20.0",
        ],
        "all": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "jupyter>=1.0",
            "notebook>=6.0",
            "xgboost>=1.6.0",
            "lightgbm>=3.3.0",
            "shap>=0.40.0",
            "optuna>=3.0.0",
            "plotly>=5.0.0",
            "streamlit>=1.20.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "churn-analysis=demo_kaggle:main",
            "churn-dashboard=dashboard.streamlit_app:main",
        ],
    },
    keywords="machine-learning, customer-churn, data-science, business-intelligence, predictive-analytics, customer-retention, clv, roi-analysis",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/customer-churn-prediction/issues",
        "Source": "https://github.com/yourusername/customer-churn-prediction",
        "Documentation": "https://github.com/yourusername/customer-churn-prediction#readme",
    },
)
