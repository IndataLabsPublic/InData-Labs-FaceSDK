import torch
import numpy as np
import albumentations as A
from typing import List
from PIL import Image
import json

from albumentations.pytorch import ToTensorV2

from ..base_attribute_classifier import BaseAttributeClassifier
from face_recognition_sdk.modules.face_attributes.utils import load_models


model_urls = {
    "res18": "https://face-demo.indatalabs.com/weights/attr_resnet18_jit_best.pt",
    "mbnet2": "https://face-demo.indatalabs.com/weights/attr_mbnet2_jit_best.pt",
    "GARClsf": "https://face-demo.indatalabs.com/weights/attr_gar_facenet_jit_best.pt" # model for gender, age and race prediction
}


class AttributeClassifier(BaseAttributeClassifier):
    """
    Implements inference for attribute classifier
    """

    def __init__(self, config):
        super().__init__(config)

        arch_model = config["architecture"]
        self.models = load_models([model_urls[arch_model], model_urls["GARClsf"]])

        self.device = torch.device(self.config["device"])
        for model in self.models:
            model["model"].to(self.device)

        self.threshold = self.config["decision_threshold"]

    def _preprocess(self, image: np.ndarray) -> List[torch.Tensor]:
        preprocessed_imgs = []
        for model in self.models:
            if isinstance(model["preprocess"], A.core.composition.Compose):
                preprocessed = model["preprocess"](image=image)
                img = preprocessed["image"].to(self.device)
            else:
                img = model["preprocess"](Image.fromarray(image)).to(self.device)

            preprocessed_imgs.append(img)

        return preprocessed_imgs

    def _predict_raw(self, images: List[torch.Tensor]) -> list:
        predictions = []
        for image, model in zip(images, self.models):
            image = image.unsqueeze(0)
            predictions.append(model["model"](image))

        return predictions

    def _postprocess(self, raw_predictions: list) -> dict:
        result = {}
        for prediction, model in zip(raw_predictions, self.models):
            if "categories_values" not in model:
                prediction = torch.sigmoid(prediction[0])
                prediction[prediction >= self.threshold] = 1
                prediction[prediction < self.threshold] = 0
                prediction = prediction.detach().cpu().numpy()

                for i, cat in enumerate(model["categories"]):
                    result[cat] = int(prediction[i])
            else:
                for i, cat in enumerate(model["categories"]):
                    category_pred = prediction[i]
                    _, indx = torch.max(category_pred.data, 1)
                    indx = indx.detach().cpu().tolist()[0]

                    result[cat] = model["categories_values"][cat][indx]

        return result

    def predict(self, image: np.ndarray) -> dict:
        images = self._preprocess(image)
        raw_preds = self._predict_raw(images)
        result = self._postprocess(raw_preds)
        return result
