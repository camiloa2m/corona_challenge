import base64
import pickle
import secrets
from io import BytesIO

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import Image
from pydantic import BaseModel

# Carga del modelo de clasificación previamente entrenado
model_path = "model/clf.pickle"
model = pickle.load(open(model_path, "rb"))

app = FastAPI()
security = HTTPBasic()  # Utilizamos seguridad básica


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


def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    # Definimos el nombre de usuario y la contraseña correctos
    correct_username = "admin"
    correct_password = "password123"

    correct_username_bytes = bytes(correct_username, "utf8")
    correct_password_bytes = bytes(correct_password, "utf8")

    user_current_username_bytes = credentials.username.encode("utf8")
    user_current_password_bytes = credentials.password.encode("utf8")

    # Comparamos las credenciales proporcionadas
    is_correct_username = secrets.compare_digest(
        user_current_username_bytes, correct_username_bytes
    )
    is_correct_password = secrets.compare_digest(
        user_current_password_bytes, correct_password_bytes
    )

    # Verificamos que el nombre de usuario y la contraseña coincidan
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


@app.get("/")
def read_root():
    return {"msg": "Digit Image Classification."}


@app.post("/predict")
def predict_class(
    input: Input, credentials: HTTPBasicCredentials = Depends(authenticate)
):
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
