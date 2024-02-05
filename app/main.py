import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import predict_pipline
from app.predict import __version__ as model_version
import argparse
import uvicorn


app = FastAPI()

class StudentLogPath(BaseModel):
    Path : str
    uid : int
    devices : str


class PredictionOut(BaseModel):
    output_path : str

@app.get("/")
def home():
    return {"Status" : "OK", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
def predict(payload : StudentLogPath):
    out_path =predict_pipline(payload.Path, payload.uid, payload.devices)
    return {"output_path": out_path}


