from setuptools import setup, find_packages

setup(
    name="dualguard",
    version="0.1",
    description="Code for Paper : DualGuard: A Parameter Space Transformation Approach for Bidirectional Defense in Split-Based LLM Fine-Tuning (ACL main 2025) ",
    author="Shreyansh",
    author_email="shreyansh.solanki53@gmail.com",
    url="https://github.com/Bin4ry16nki/DualGuard",
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Ubuntu 22.04",
    ],
    python_requires=">=3.10",
)