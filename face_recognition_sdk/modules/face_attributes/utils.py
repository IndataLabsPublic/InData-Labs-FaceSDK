import torch
import json
from torchvision import transforms
from face_recognition_sdk.utils.load_utils import get_file_from_url
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _load_model(url, ):
    model_path = get_file_from_url(url, progress=True, unzip=False)
    model = torch.jit.load(model_path)

    hparams = json.loads(model.hparams)

    categories = hparams['categories']
    categories_values = hparams.get('categories_values', None)

    if "preprocess" in hparams:
        preprocess = A.from_dict(hparams["preprocess"])
    else:
        preprocess = get_default_preprocess()

    loaded = {
        'model': model,
        'categories': categories,
        'categories_values': categories_values,
        'preprocess': preprocess
    }

    if categories_values is None:
        loaded.pop('categories_values')

    return loaded


def load_models(url_or_urls):
    if isinstance(url_or_urls, str):
        url_or_urls = [url_or_urls]

    models = []
    for url in url_or_urls:
        models.append(_load_model(url))

    return models


def get_default_preprocess():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])