import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
import os

app = Flask(__name__)

fe  = FeatureExtractor()
features = []
img_paths = []

feature_folder_path = "static/feature"
img_folder_path = "static\image"

feature_path_list = os.listdir(feature_folder_path)
image_path_list = os.listdir(img_folder_path)

for feature_name in feature_path_list:
    feature_path = os.path.join("static/feature",feature_name)
    features.append(np.load(feature_path))

    name = feature_name.split(".")[0]+".jpg"
    img_path = os.path.join(img_folder_path, name)
    img_paths.append(img_path)

    

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        file = request.files["query_img"]

        img =  Image.open(file.stream)
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":",".") + "_" + file.filename
        img.save(uploaded_img_path)

        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)
        ids = np.argsort(dists)[:30]
        scores = [(dists[id], img_paths[id]) for id in ids]

        print(scores)

        return render_template("index.html", query_path=uploaded_img_path, scores=scores)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run()