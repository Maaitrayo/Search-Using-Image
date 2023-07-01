from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.models import Model
import numpy as np
import cv2


import tensorflow 

class FeatureExtractor:
    def __init__(self) -> None:
        print(tensorflow.version.VERSION)
        base_model = tensorflow.keras.applications.vgg16.VGG16(weights="imagenet")
        self.model = tensorflow.keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)

    def extract(self, img):
        img = np.array(img)
        img = cv2.resize(img,(224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img.resize((224,224)).convert("RGB")

        x = image.img_to_array(img)
        x = np.expand_dims(img,axis=0) # (H,W,C) --> (1,H,W,C) {1 -> BATCH SIZE}
        x = tensorflow.keras.applications.vgg16.preprocess_input(x)

        feature = self.model.predict(x)[0]
        feature_norm = feature/np.linalg.norm(feature)
        
        return feature_norm