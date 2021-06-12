import os
import cv2
from flask import Flask, request, Response
import json
import numpy as np
from PIL import Image

from face_recognition_sdk.utils.database import FaceRecognitionSystem
from face_recognition_sdk.utils.io_utils import read_yaml, read_image, save_image

import argparse

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

sdk_config = read_yaml("../../face_recognition_sdk/config/config.yaml")

system = FaceRecognitionSystem(sdk_config)

db_folder_path = "../../../employees/database"

# create, save and load database initialized from folders containing user photos
system.load_database(db_folder_path)


@app.route("/")
def index():
    return Response("WebCam demo")


@app.route("/local")
def local():
    return Response(open("./static/local.html").read(), mimetype="text/html")


@app.route("/image", methods=["POST"])
def image():
    try:
        image_file = request.files["image"]  # get the image

        img = np.array(Image.open(image_file))

        bboxes, landmarks, user_ids, similarities = system.sdk.recognize_faces(img)
        names = [system.get_user_name(uid) for uid in user_ids]

        height, width, _ = img.shape

        objects = [
            {"name": name, "bbox": [float(x) for x in bbox[:4]], "score": float(score)}
            for name, bbox, score in zip(names, bboxes, similarities)
        ]

        return json.dumps(objects)

    except Exception as e:
        print(f"POST /image error: {e}")
        return e


if __name__ == "__main__":

    app.run(debug=True, host="0.0.0.0")
