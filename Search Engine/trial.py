import torch
import clip
from fastapi import FastAPI, Body, Request, Form
import uvicorn
from PIL import Image
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates



statics = 'F:\Internship\Search Engine\static'
templates = Jinja2Templates(directory='templates')

app = FastAPI()


# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

# image = preprocess(Image.open("F:\Internship\Search Engine\dog.jpg")).unsqueeze(0).to(device)
# text = clip.tokenize(["a dog", "a cat"]).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
    
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)




@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
