import cv2
import os
import numpy as np

from feature_extractor import FeatureExtractor


if __name__ == "__main__":
    # img_folder_path = "static/image/"
    img_folder_path = "D:\DEVELOPMENT PROJECTS\IIMAGE BASED SEARCH ENGINE PROJECT\dataset"
    feature_folder_path = "static/feature/"
    feature_type = ".npy"

    fe = FeatureExtractor()

    img_list = os.listdir(img_folder_path)
    print(f"[!] FOUND {len(img_list)} IMAGES [!]")

    for img_name in img_list:
        img_path = os.path.join(img_folder_path,img_name)
        img = cv2.imread(img_path)

        feature = fe.extract(img)
        print(type(feature), feature.shape)
        
        name = img_name.split(".")[0]
        feature_name = name+feature_type
        feature_path = os.path.join(feature_folder_path, feature_name)
        # print(feature_path)

        np.save(feature_path, feature)
