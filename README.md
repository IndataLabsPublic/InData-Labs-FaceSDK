## Installation

Clone and install dependencies for development

```
git clone https://github.com/IndataLabsPublic/InData-Labs-FaceSDK.git
cd face-recognition-sdk
conda env create -f environment.yml
conda activate face_sdk
pre-commit install
git lfs install
python setup.py build develop
```

install only face_recognition_sdk with dependencies

```
pip install .
```
