from fastapi import FastAPI
import gradio as gr

from app import pipeline_function,iface

app = FastAPI()
@app.get("/")
async def predict():
    return "Running successfully. go to /predict"

app = gr.mount_gradio_app(app,iface,path='/predict')