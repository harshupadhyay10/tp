import torch
import clip
from PIL import Image
import glob
import pandas as pd



device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

STATIC = 'F:\Internship\Search Engine\static'


def preprocess_text(text):
    return clip.tokenize(text).to(device)

def encode_text(tokenized_text):
    text_features = model.encode_text(tokenized_text)
    # text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

def preprocess_image(path):
    return preprocess(Image.open(path)).unsqueeze(0).to(device)

def encode_image(preprocessed_image):
    image_features = model.encode_image(preprocessed_image).detach().numpy()
    # image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features
 
def find_similarity(text_features, image_features):
    image_features = torch.from_numpy(image_features)
    similarity = image_features @ text_features.T
    return similarity[0][0].detach().numpy()

def create_image_embedding():
    paths = glob.glob(STATIC + "\*.jpeg") + glob.glob(STATIC + "\*.jpg")
    df = pd.DataFrame()
    df['path'] = paths
    encoded_list = []
    for path in paths:
        preprocessed_image = preprocess_image(path)
        #encoded_image = encode_image(preprocessed_image)
        encoded_list.append(encode_image(preprocessed_image))
        #images_dict[path] = find_similarity(search_for, encoded_image)
        
    df['embedding'] = encoded_list
    return df



query = input("Search for: ")

search_for = preprocess_text(query)
search_for = encode_text(search_for)

df = create_image_embedding()
df['similarity'] = [find_similarity(search_for, embedded_image) for embedded_image in df['embedding']]
print(df)