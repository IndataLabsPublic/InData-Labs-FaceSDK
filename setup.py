from setuptools import setup, find_packages
import os

PATH_ROOT = os.path.dirname(__file__)


def load_requirements(path_dir=PATH_ROOT, comment_char="#"):
    with open(os.path.join(path_dir, "requirements.txt"), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)]
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="face_recognition_sdk",
    version="0.0.2",
    author="InData Labs",
    author_email="info@indatalabs.com",
    description="A python package for face detection and recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
    url="https://gitlab.indatalabs.com/cv_rg/face-recognition-sdk",
    packages=find_packages(exclude=["demo", "test/*", "docs"]),
    python_requires=">=3.6",
    install_requires=load_requirements(PATH_ROOT),
)
