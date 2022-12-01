import cv2
import mediapipe as mp
import numpy as np

BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white
class bgMask:
    def __init__(self) -> None:
        mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.model = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
        mp_face_detection = mp.solutions.face_detection
        self.faceModel = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    def run(self, image):
        image_height, image_width, _ = image.shape
        results = self.faceModel.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            print("Face not detected")
            return None
        results = self.model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # blurred_image = cv2.GaussianBlur(image,(55,55),0)
        # outputImage = np.where(condition, img[:, :, ::-1], MASK_COLOR)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        # outputImage = cv2.GaussianBlur(image,(55,55),0)
        outputImage = np.where(condition, image[:, :, ::-1], MASK_COLOR)
        output_image = np.where(condition, image, outputImage)
        return output_image