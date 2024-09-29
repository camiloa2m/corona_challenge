import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException
from io import BytesIO
import base64
import numpy as np
from PIL import Image
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app, authenticate, preprocess_image

client = TestClient(app)

class TestFastAPIApp(unittest.TestCase):
    
    @patch("main.security")
    @patch("main.secrets.compare_digest")
    def test_authenticate_success(self, mock_compare_digest, mock_security):
        """
        Test successful authentication with correct username and password.
        """
        mock_compare_digest.side_effect = [True, True]  # Mocks username and password match
        credentials = MagicMock(username="admin", password="password123")
        
        # Should not raise an exception if authentication is successfull
        authenticate(credentials)
    
    @patch("main.security")
    @patch("main.secrets.compare_digest")
    def test_authenticate_failure(self, mock_compare_digest, mock_security):
        """
        Test authentication failure when an incorrect password is provided.
        """       
        mock_compare_digest.side_effect = [True, False]  # Mocks incorrect password
        credentials = MagicMock(username="admin", password="wrong_password")
        
        with self.assertRaises(HTTPException):
            authenticate(credentials)
    
    def test_read_root(self):
        """
        Test the root endpoint (GET /) to check if the correct response is returned.
        """
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"msg": "Digit Image Classification."})
    
    @patch("main.model.predict")
    def test_predict_class_success(self, mock_predict):
        """
        Test the /predict endpoint with valid base64-encoded image data and ensure
        the correct prediction is returned.
        """
        # Create a sample image and convert it to base64
        img = Image.new("L", (8, 8))  # A simple grayscale image of size 8x8
        img_b = BytesIO()
        img.save(img_b, format="PNG")
        img_b64 = base64.b64encode(img_b.getvalue()).decode("utf-8")

        # Mock the model's prediction
        mock_predict.return_value = np.array([3])  # Example prediction

        # Define the input data
        input_data = {
            "subject": {
                "type": "base64",
                "value": img_b64
            },
            "subject_type": "Image"
        }

        # Make the POST request
        response = client.post("/predict", json=input_data, auth=("admin", "password123"))

        # Validate the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"prediction": 3})

    def test_preprocess_image(self):
        """
        Test the preprocess_image function to ensure the image is resized to 8x8 and 
        converted to grayscale (mode "L").
        """
        # Create a sample image
        img = Image.new("RGB", (10, 10))  # Create an image that's not 8x8 and in RGB mode

        # Call the preprocess_image function
        processed_img = preprocess_image(img)

        # Check if the processed image has the correct size and mode
        self.assertEqual(processed_img.size, (8, 8))
        self.assertEqual(processed_img.mode, "L")
    
    def test_predict_class_invalid_subject_type(self):
        """
        Test the /predict endpoint with an invalid subject_type to ensure the correct
        error message is returned.
        """
        input_data = {
            "subject": {
                "type": "base64",
                "value": "some_base64_string"
            },
            "subject_type": "NotImage"
        }

        response = client.post("/predict", json=input_data, auth=("admin", "password123"))

        self.assertEqual(response.status_code, 600)
        self.assertEqual(response.json()["detail"], "Subject_type is not 'Image'")
        
    def test_predict_class_invalid_type(self):
        """
        Test the /predict endpoint with an invalid type to ensure the correct
        error message is returned.
        """
        input_data = {
            "subject": {
                "type": "base8",
                "value": "some_base64_string"
            },
            "subject_type": "Image"
        }

        response = client.post("/predict", json=input_data, auth=("admin", "password123"))

        self.assertEqual(response.status_code, 600)
        self.assertEqual(response.json()["detail"], "Type field (image encoded) must be 'base64'")
    
    def test_predict_class_invalid_base64(self):
        """
        Test the /predict endpoint with an invalid base64 string to ensure
        the correct error message is returned.
        """
        input_data = {
            "subject": {
                "type": "base64",
                "value": "invalid_base64_string"
            },
            "subject_type": "Image"
        }

        response = client.post("/predict", json=input_data, auth=("admin", "password123"))

        self.assertEqual(response.status_code, 601)
        self.assertIn("Error decoding image", response.json()["detail"])
    
    def test_predict_class_invalid_image_processing(self):
        """
        Test the /predict endpoint with a valid base64 string but invalid image data
        to ensure the correct error message is returned during image processing.
        """
        # Valid base64 but invalid image data (not an actual image)
        invalid_image_data = base64.b64encode(b"not an image").decode("utf-8")
        
        input_data = {
            "subject": {
                "type": "base64",
                "value": invalid_image_data
            },
            "subject_type": "Image"
        }

        response = client.post("/predict", json=input_data, auth=("admin", "password123"))

        self.assertEqual(response.status_code, 602)
        self.assertIn("Error reading or preprocessing image", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
