import torch
import clip
from fastapi import FastAPI, Body, Request, Form
import uvicorn
from PIL import Image
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import glob


STATIC = 'F:\Internship\Search Engine\static'
templates = Jinja2Templates(directory='templates')

app = FastAPI()



@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

paths = glob.glob(STATIC + "\*.jpeg") + glob.glob(STATIC + "\*.jpg")
df = pd.DataFrame()
df['path'] = paths
print(df)
print(df['path'].tolist())

@app.post('/imagesearch')
async def search_image(request: Request):
    form_data = await request.form()
    text = form_data['query']
    print(text)
    return templates.TemplateResponse("images.html", {"request": request, "images": df['path'].tolist()})