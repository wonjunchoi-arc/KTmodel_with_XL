from fastapi import FastAPI
from pydantic import BaseModel


import uvicorn


app = FastAPI()

class StudentLogPath(BaseModel):
    Path : str
    uid : int


class PredictionOut(BaseModel):
    Path : str

@app.get("/")
def home():
    return {"Status" : "OK", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
def predict(payload : StudentLogPath):
    out_path =predict_pipline(payload.Path, payload.uid)
    return {"output_path": out_path}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8008)


