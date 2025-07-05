# from deepface import DeepFace
# from transformers import pipeline
# import torch
# import cv2
# import numpy as np
# from PIL import Image
# import tempfile

# class FeatureExtractor:
#     def __init__(self):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
#         # Initialize fashion models
#         # self.fashion_model = pipeline(
#         #     "image-classification",
#         #     model="valhalla/clip-clothing-attributes",
#         #     device=self.device
#         # )
        
#         # self.color_model = pipeline(
#         #     "image-classification",
#         #     model="cafeai/cafeai-color-attributes",
#         #     device=self.device
#         # )

#     def extract(self, person_image):
#         if isinstance(person_image, np.ndarray):
#             # Convert numpy array to PIL Image for fashion models
#             pil_img = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))
            
#             # Save temp file for DeepFace
#             with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
#                 cv2.imwrite(tmp.name, person_image)
#                 return self._extract_features(tmp.name, pil_img)
#         else:
#             # Input is file path
#             pil_img = Image.open(person_image)
#             return self._extract_features(person_image, pil_img)

#     def _extract_features(self, face_img_path, fashion_img):
#         try:
#             # 1. Get face attributes using DeepFace
#             face_analysis = DeepFace.analyze(
#                 img_path=face_img_path,
#                 actions=['age', 'gender', 'emotion'],
#                 enforce_detection=False,
#                 detector_backend='retinaface'
#             )[0]  # Take first face found
            
#             # 2. Get clothing attributes
#             # fashion_attrs = self.fashion_model(fashion_img, top_k=5)
#             # color_attrs = self.color_model(fashion_img, top_k=3)
            
#             return {
#                 'gender': face_analysis['gender'],
#                 'age': int(face_analysis['age']),
#                 'emotion': face_analysis['dominant_emotion'].lower(),
#                 # 'clothing': [attr['label'] for attr in fashion_attrs],
#                 # 'colors': [attr['label'] for attr in color_attrs],
#                 # 'style': self._detect_style(fashion_attrs)
#             }
            
#         except Exception as e:
#             print(f"Feature extraction error: {e}")
#             return self._get_fallback_features()

#     # def _detect_style(self, fashion_attrs):
#     #     """Determine clothing style from attributes"""
#     #     attrs_text = " ".join([attr['label'] for attr in fashion_attrs])
        
#     #     if 'formal' in attrs_text or 'suit' in attrs_text:
#     #         return 'formal'
#     #     elif 'sport' in attrs_text or 'athletic' in attrs_text:
#     #         return 'sporty'
#     #     return 'casual'

#     def _get_fallback_features(self):
#         """Return when feature extraction fails"""
#         return {
#             'gender': 'unknown',
#             'age': 0,
#             'emotion': 'neutral',
#             'clothing': [],
#             'colors': [],
#             'style': 'unknown'
#         }

from collections import defaultdict
from deepface import DeepFace
from transformers import pipeline
import torch
import cv2
import numpy as np
from PIL import Image

class FeatureExtractor:
    def __init__(self):
        
        self.profile = defaultdict(lambda: {
            'gender': 'unknown',
            'age': 0,
            'emotion': 'neutral',
            'clothing': [],
            'clothing_scores': [],
            'style': 'unknown'
        })
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize fashion detection pipeline
        self.fashion_pipe = pipeline(
            "object-detection", 
            model="valentinafeve/yolos-fashionpedia",
            device=self.device
        )
        
        # # Initialize color classification pipeline
        # self.color_model = pipeline(
        #     "image-classification",
        #     model="cafeai/cafeai-color-attributes",
        #     device=self.device
        # )
        
    def __str__(self):
        return f"Person Profile: {dict(self.profile)}"


    def extract(self, person_image):
        if isinstance(person_image, np.ndarray):
            # Convert numpy array to PIL Image for fashion models
            pil_img = Image.fromarray(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))
            
            # Directly use numpy array if DeepFace supports it
            return self._extract_features(person_image, pil_img)
        else:
            # Input is file path
            pil_img = Image.open(person_image)
            return self._extract_features(person_image, pil_img)

    def _extract_features(self, face_img, fashion_img):
        try:
            # 1. Get face attributes using DeepFace
            face_analysis = DeepFace.analyze(
                img_path=face_img,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False,
                detector_backend='retinaface'
            )[0]  # Take first face found
            
            # print("Face Analysis: ", face_analysis)
            if face_analysis:
                self.profile['gender'] = face_analysis['dominant_gender'].lower()
                self.profile['age'] = int(face_analysis['age'])
                self.profile['emotion'] = face_analysis['dominant_emotion'].lower()            
            
            # 2. Get clothing attributes using the pipeline
            fashion_results = self.fashion_pipe(fashion_img)
            if fashion_results:
                self.profile['clothing'] = list(set([attr['label'] for attr in fashion_results]))
                self.profile['style'] = self._detect_style(self.profile['clothing'])
            return self.profile   
            
            # 3. Get color attributes
            # color_attrs = self.color_model(fashion_img, top_k=3)
        
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return self._get_fallback_features()

    def _detect_style(self, fashion_attrs):
        """Determine clothing style from attributes"""
        
        if 'formal' in fashion_attrs or 'suit' in fashion_attrs:
            return 'formal'
        elif 'sport' in fashion_attrs or 'athletic' in fashion_attrs:
            return 'sporty'
        return 'casual'

    def _get_fallback_features(self):
        """Return when feature extraction fails"""
        return {
            'gender': 'unknown',
            'age': 0,
            'emotion': 'neutral',
            'clothing': [],
            'clothing_scores': [],
            'colors': [],
            'style': 'unknown'
        }