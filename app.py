import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import torch
from torchvision import models
from PIL import Image
import torch.optim as optim
import numpy as np
import requests
from io import BytesIO
import webbrowser
from threading import Timer
from torchvision import transforms


# Setup Flask app
app = Flask(__name__)

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VGG19 model
vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad_(False)
vgg.to(device)

# Helper functions (same as you provided)

def load_image(img_path, max_size=400, shape=None):
    ''' Load in and transform an image.'''
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')
    
    # resize the image to max_size
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([ 
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image

def im_convert(tensor):
    """ Convert a tensor to an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a tensor. """
    batch_size, d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def get_features(image, model, layers=None):
    """ Run an image through the model and get features from specified layers. """
    if layers is None:
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
    
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """ Handle image uploads and apply neural style transfer. """
    # Get content and style images from the form
    content_file = request.files['content']
    style_file = request.files['style']
    
    # Save images locally
    content_path = os.path.join('static', secure_filename(content_file.filename))
    style_path = os.path.join('static', secure_filename(style_file.filename))
    content_file.save(content_path)
    style_file.save(style_path)
    
    # Load images
    content = load_image(content_path).to(device)
    style = load_image(style_path, shape=content.shape[-2:]).to(device)

    # Neural Style Transfer (as you wrote)
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    # Initialize target image
    target = content.clone().requires_grad_(True).to(device)

    style_weights = {'conv1_1': 1., 'conv2_1': 0.75, 'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}
    content_weight = 1
    style_weight = 1e6

    optimizer = optim.Adam([target], lr=0.003)
    steps = 20
    show_every = 10

    # Optimization loop
    for ii in range(1, steps+1):
        target_features = get_features(target, vgg)
        
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            style_loss += layer_style_loss / (d * h * w)
        
        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if ii % show_every == 0:
            print('Total loss: ', total_loss.item())

    # Save the target image
    result_path = os.path.join('static', 'result.jpg')
    result_image = im_convert(target)
    result_image = Image.fromarray((result_image * 255).astype(np.uint8))
    result_image.save(result_path)

    return jsonify({"result_image": result_path})

# Automatically open the browser when running the app
def open_browser():
    """Open the browser to the Flask app URL automatically."""
    webbrowser.open("http://127.0.0.1:5000")

if __name__ == "__main__":
    Timer(1, open_browser).start()  # Wait a second before opening the browser
    app.run(debug=True)
