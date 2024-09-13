from setuptools import find_packages, setup
from pathlib import Path

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

PRJ_NAME = "Floral_Gift_Classifier"
SRC_REPO = "flowerClassifier"
AUTHOR_USER_NAME = "Yuanfeng92"
AUTHOR_EMAIL = "yuanfengluan@gmail.com"

setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A tool to help you pick out the perfect flower!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https:/github.com/{AUTHOR_USER_NAME}/{PRJ_NAME}",
    project_urls={
        "Bug Tracker": f"https://github/{AUTHOR_USER_NAME}/{PRJ_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)