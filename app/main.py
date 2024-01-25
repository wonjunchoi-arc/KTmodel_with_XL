from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import predict_pipline
from app.predict import __version__ as model_version

import uvicorn


app = FastAPI()

class StudentLogPath(BaseModel):
    Path : str
    uid : int


class PredictionOut(BaseModel):
    output_path : str

@app.get("/")
def home():
    return {"Status" : "OK", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
def predict(payload : StudentLogPath):
    out_path =predict_pipline(payload.Path, payload.uid)
    return {"output_path": out_path}



# if __name__ == '__main__':
#     uvicorn.run(app)


