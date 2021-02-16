import numpy as np
import cv2
import logging
from typing import List, Tuple
from pathlib import Path

from .modules.detection.retinaface import RetinaFace
from .modules.recognition.insightface import InsightFaceEmbedder
from .modules.face_attributes import AttributeClassifier
from .modules.alignment import align_and_crop_face
from .modules.database import FaissFaceStorage
from .utils.io_utils import read_yaml


logger = logging.getLogger(__name__)


class FaceRecognitionSDK:
    def __init__(self, config: dict = None):

        if config is None:
            path_to_default_config = Path(Path(__file__).parent, "config/config.yaml").as_posix()
            config = read_yaml(path_to_default_config)

        logger.info("Start SDK initialization.")
        self.detector = RetinaFace(config["detector"])
        self.embedder = InsightFaceEmbedder(config["embedder"])
        self.attr_classifier = AttributeClassifier(config["attributes"])
        self.database = FaissFaceStorage(config["database"])
        logger.info("Finish SDK initialization")

    def load_database(self, path: str) -> None:
        """
        Loads database from disk.

        Args:
            path: path to database
        """
        logger.info(f"Loading the database of face descriptors from {path}.")
        self.database.load(path)
        logger.debug("Finish loading the database of face descriptors.")

    def save_database(self, path: str) -> None:
        """
        Saves database to disk.

        Args:
            path: path to database

        """
        logger.info(f"Saving the database of face descriptors to {path}.")
        self.database.save(path)
        logger.debug("Finish saving the database of face descriptors.")

    def reset_database(self) -> None:
        """Reset/clear database."""
        logger.info("Resetting database of face descriptors.")
        self.database.reset()
        logger.debug("Finish database of face descriptors reset.")

    def extract_face_descriptor(self, image: np.ndarray):
        """
        Extracts descriptor from image with single face.

        Args:
            image: numpy image (H,W,3) in RGB format.
        """
        logger.debug("Start extracting face descriptor.")
        bboxes, landmarks = self.detect_faces(image)

        if len(bboxes) > 1:
            raise ValueError("Detected more than one face on provided image.")
        elif len(bboxes) == 0:
            raise ValueError("Can't detect any faces on provided image.")

        face = self.align_face(image, landmarks[0])
        descriptor = self.get_descriptor(face)

        face_coordinates = (bboxes[0], landmarks[0])

        logger.debug("Finish face extraction")
        return descriptor, face_coordinates

    def add_photo_by_user_id(self, image: np.ndarray, user_id: int):
        """
        Adds photo of the user to the database.

        Args:
            image: numpy image (H,W,3) in RGB format.
            user_id: id of the user.
        """
        logger.info(f"Adding photo of user with user_id={user_id}")
        descriptor, _ = self.extract_face_descriptor(image)
        self.add_descriptor(descriptor, user_id)
        logger.debug(f"Finish adding user photo for user_id={user_id}")

    def add_descriptor(self, descriptor: np.ndarray, user_id: int) -> Tuple[None, int]:
        """
        Add descriptor for user specified by 'user_id'.

        Args:
            descriptor: descriptor of the photo (face) to use as a search query.
            user_id: if of the user

        Returns:

        """
        logger.info(f"Adding descriptor for user with user_id={user_id}")
        self.database.add_descriptor(descriptor, user_id)
        logger.debug(f"Finish adding descriptor for user with user_id={user_id}")

    def delete_photo_by_id(self, photo_id: int) -> None:
        """
        Removes photo (descriptor) from the database.

        Args:
            photo_id: id of the photo in the database.

        """
        raise NotImplementedError()

    def delete_user_by_id(self, user_id: int) -> None:
        """
        Removes all photos of the user from the database.

        Args:
            user_id: id of the user.
        """
        logger.info(f"Deleting user with user_id={user_id} from faces descriptors database.")
        self.database.remove_user(user_id)
        logger.debug(f"Finish deleting user with user_id={user_id} from faces descriptors database.")

    def find_most_similar(self, descriptor: np.ndarray, top_k: int = 1):
        """
        Find most similar-looking photos (and their user id's) in the database.

        Args:
            descriptor: descriptor of the photo (face) to use as a search query.
            top_k: number of most similar results to return.
        """
        logger.debug("Searching for a descriptor in the database.")
        indicies, distances = self.database.find(descriptor, top_k)
        logger.debug("Finish searching for a descriptor in the database.")
        return indicies, distances

    def verify_faces(self, first_face: np.ndarray, second_face: np.ndarray):
        """
        Check if two face images are of the same person.

        Args:
            first_face: image of the first face.
            second_face: image of the second face.
        """
        logger.debug("Start verifying faces.")
        first_descriptor, first_face_coordinates = self.extract_face_descriptor(first_face)
        second_descriptor, second_face_coordinates = self.extract_face_descriptor(second_face)
        similarity = self.get_similarity(first_descriptor, second_descriptor)
        logger.debug(f"Finish faces verifying. Similarity={float(similarity)}")
        return similarity, first_face_coordinates, second_face_coordinates

    def detect_faces(self, image: np.ndarray):
        """
        Detect all faces on the image.

        Args:
            image: numpy image (H,W,3) in RGB format.
        """
        logger.debug("Start faces detection.")
        bboxes, landmarks = self.detector.predict(image)
        logger.debug(f"Finish faces detection. Count of detected faces: {len(bboxes)}.")
        return bboxes, landmarks

    def recognize_faces(self, image: np.ndarray):
        """
        Recognize all faces on the image.

        Args:
            image: numpy image (H,W,3) in RGB format.
        """
        logger.debug("Start faces recognition.")
        bboxes, landmarks = self.detect_faces(image)

        user_ids = []
        similarities = []

        for i, face_keypoints in enumerate(landmarks):

            face = self.align_face(image, face_keypoints)
            descriptor = self.get_descriptor(face)
            indicies, distances = self.find_most_similar(descriptor)
            user_ids.append(indicies[0])
            similarities.append(distances[0])

        logger.debug(f"Finish faces recognition. Count of processed faces: {len(bboxes)}")
        return bboxes, landmarks, user_ids, similarities

    def get_descriptor(self, face_image: np.ndarray) -> np.ndarray:
        """
        Get descriptor of the face image.

        Args:
            face_image: numpy image (112,112,3) in RGB format.

        Returns:
            descriptor: float array of length 'descriptor_size' (default: 512).
        """
        logger.debug("Start descriptor extraction from image of face.")
        descriptor = self.embedder(face_image)
        logger.debug("Finish descriptor extraction.")
        return descriptor

    def get_similarity(self, first_descriptor: np.ndarray, second_descriptor: np.ndarray):
        """
        Calculate dot similarity of 2 descriptors

        Args:
            first_descriptor: float array of length 'descriptor_size' (default: 512).
            second_descriptor: float array of length 'descriptor_size' (default: 512.
        Returns:
            similarity: similarity score. Value - from 0 to 1.
        """
        similarity = np.dot(first_descriptor, second_descriptor)
        return similarity

    def align_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Align face on the image.

        Args:
            image: numpy image (H,W,3) in RGB format.
            landmarks: 5 keypoints of the face to align.
        Returns:
            face: aligned and cropped face image of shape (112,112,3)
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        face = align_and_crop_face(image, landmarks, size=(112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        return face

    def get_face_attributes(self, face_image: np.ndarray) -> dict:
        """
        Get attributes of face. Currently supported: "Wearing_Hat", "Mustache", "Eyeglasses", "Beard", "Mask",
        "Gender", "Age", "Race".

        Args:
            face_image: numpy image (112,112,3) in RGB format.

        Returns: dict with attributes flags (1 - True (present), 0 - False (not present)).

        """
        logger.debug("Start face attributes classification.")
        attrs = self.attr_classifier.predict(face_image)
        logger.debug("Finish face attributes classification.")
        return attrs

    def set_configuration(self, config: dict):
        """Configure face recognition sdk."""
        raise NotImplementedError()
