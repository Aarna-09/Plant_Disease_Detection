import os
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data using relative paths
disease_info = pd.read_csv(os.path.join(BASE_DIR, "disease_info.csv"), encoding='cp1252')
supplement_info = pd.read_csv(os.path.join(BASE_DIR, "supplement_info.csv"), encoding='cp1252')

# Load model
model_path = os.path.join(BASE_DIR, "plant_disease_model_1_latest.pt")
model = CNN.CNN(39)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).unsqueeze(0)
    output = model(input_data).detach().numpy()
    return np.argmax(output)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    image = request.files['image']
    upload_dir = os.path.join(app.static_folder, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, image.filename)
    image.save(file_path)

    pred = prediction(file_path)
    title = disease_info['disease_name'][pred]
    desc = disease_info['description'][pred]
    prevent = disease_info['Possible Steps'][pred]
    image_url = disease_info['image_url'][pred]
    sname = supplement_info['supplement name'][pred]
    simage = supplement_info['supplement image'][pred]

    return render_template('submit.html', title=title, desc=desc,
                           prevent=prevent, image_url=image_url,
                           sname=sname, simage=simage)

@app.route('/market')
def market():
    return render_template('market.html',
                           supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
