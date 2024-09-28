import base64
import pickle
from io import BytesIO

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from PIL import Image
from pydantic import BaseModel

# Carga del modelo de clasificación previamente entrenado
model_path = "model/clf.pickle"
model = pickle.load(open(model_path, "rb"))

app = FastAPI()


class SubjectModel(BaseModel):
    type: str
    value: str


class Input(BaseModel):
    subject: SubjectModel
    subject_type: str


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Función para procesar la imagen y convertirla en un formato que el modelo pueda usar

    Args:
        image (Image.Image): Imagen Image.Image querespresenta un objeto imagen de Pillow.

    Returns:
        np.ndarray: Representación en array de la imagen.
    """
    if image.mode != "L":
        # Tener la escala de color adecuada
        image = image.convert("L")
    if image.size != (8, 8):
        # Redimensionar al tamaño adecuado para el modelo
        image = image.resize((8, 8))
    return image


@app.get("/")
def read_root():
    return {"msg": "Digit Image Classification."}


@app.post("/predict")
def predict_class(input: Input):
    data = jsonable_encoder(input)

    if data["subject_type"] != "Image":
        raise HTTPException(status_code=600, detail="Subject_type is not 'Image'")
    if data["subject"]["type"] == "base64":
        img_b64 = data["subject"]["value"]
    else:
        raise HTTPException(
            status_code=600, detail="Type field (image encoded) must be 'base64'"
        )

    try:
        img_b = BytesIO(base64.b64decode(img_b64))
    except Exception as err:
        raise HTTPException(status_code=601, detail=f"Error decoding image: {err}")

    try:
        image = Image.open(img_b)
        image = preprocess_image(image)
        image_np = np.array(image).reshape((1, -1))
    except Exception as err:
        raise HTTPException(
            status_code=602, detail=f"Error reading or preprocessing image: {err}"
        )

    try:
        prediction = model.predict(image_np)
        return {"prediction": prediction.item()}
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Error in the prediction: {err}")
