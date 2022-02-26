from sys import path
import torch
import clip
from PIL import Image
import glob


static = 'F:\Internship\Search Engine\static'

def print_similarity(query, paths):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize(query).to(device)
    text_features = model.encode_text(text)
    image = []
    image_features = []
    for path in paths:
        image.append(preprocess(Image.open(path)).unsqueeze(0).to(device))
    with torch.no_grad():
        for img in image:
            image_features.append(model.encode_image(img))
    
    
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features @ image_features.T
        print(similarity)    
            


query = input("Search for: ")

paths = glob.glob(static + "\*.jpeg") + glob.glob(static + "\*.jpg") 
print_similarity(query, paths)






