from flask import Flask, request, send_file, jsonify
import os
from werkzeug.utils import secure_filename
import logging
import torch
import torch.nn as nn
from flask_cors import CORS

app = Flask(__name__)  # Fixed: _name_ to __name__
BASE_FOLDER = "uploads"
PTH_MODEL_PATH = "model/signature_cnn.pth"  # Original model
PT_MODEL_PATH = "model/signature_cnn.pt"  # Converted model
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Logging settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # Fixed: _name_ to __name__


# Model class definition
class SignatureCNN(nn.Module):
    def __init__(self):  # Fixed: _init to __init__
        super(SignatureCNN, self).__init__()  # Fixed: _init to __init__
        IMG_HEIGHT, IMG_WIDTH = 128, 128

        # 1. Convolutional Blocks
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm2d(256)

        # 2. Fully Connected Layers
        self.fc1 = nn.Linear(256 * (IMG_HEIGHT // 16) * (IMG_WIDTH // 16), 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional Layers
        x = self.pool(nn.functional.leaky_relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(nn.functional.leaky_relu(self.batchnorm2(self.conv2(x))))
        x = self.pool(nn.functional.leaky_relu(self.batchnorm3(self.conv3(x))))
        x = self.pool(nn.functional.leaky_relu(self.batchnorm4(self.conv4(x))))

        # Flatten
        x = x.view(-1, 256 * (128 // 16) * (128 // 16))

        # Fully Connected Layers
        x = nn.functional.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.functional.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        # Change: Return output as a single number (as tensor)
        return torch.sigmoid(x)  # Limit to 0-1 with Sigmoid


# Fixed model conversion function
def convert_model_to_torchscript():
    try:
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(PT_MODEL_PATH), exist_ok=True)

        # Load original model
        model = SignatureCNN()

        # Load model state within try-except
        try:
            model.load_state_dict(torch.load(PTH_MODEL_PATH, map_location=torch.device('cpu')))
            logger.info("Model state loaded successfully")
        except Exception as e:
            logger.error(f"Could not load model state: {str(e)}")
            # Use model as is (untrained) if model doesn't exist
            logger.info("Using untrained model")

        model.eval()  # Switch to evaluation mode

        # Example input - exactly the same size as expected by the model
        example_input = torch.rand(1, 1, 128, 128)  # 1 batch, 1 channel (grayscale), 128x128 size

        # Convert to TorchScript format - using script
        # IMPORTANT: Using script not trace - model output will be more consistent
        script_module = torch.jit.script(model)

        # Test and log
        with torch.no_grad():
            test_output = script_module(example_input)
            logger.info(f"Test output: {test_output}")
            logger.info(f"Test output type: {type(test_output)}")
            logger.info(f"Test output shape: {test_output.shape}")

        # Save in PyTorch Mobile format
        script_module.save(PT_MODEL_PATH)
        logger.info(f"Model successfully converted to TorchScript format: {PT_MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Model conversion error: {str(e)}")
        return False


@app.route("/", methods=["GET"])
def home():
    """Home page - API information"""
    return jsonify({
        "message": "Welcome to Signature Verification API",
        "endpoints": {
            "test": "/test - To check if the API is working",
            "get-model": "/get-model - To download the TorchScript model"
        }
    })


@app.route("/get-model", methods=["GET"])
def get_model():
    """Send the model file"""
    try:
        # Convert if PT model doesn't exist or PTH model is newer
        if not os.path.exists(PT_MODEL_PATH) or (
                os.path.exists(PTH_MODEL_PATH) and
                os.path.getmtime(PTH_MODEL_PATH) > os.path.getmtime(PT_MODEL_PATH)
        ):
            if not convert_model_to_torchscript():
                return jsonify({"error": "Model conversion error"}), 500

        if not os.path.exists(PT_MODEL_PATH):
            logger.error("TorchScript model file not found")
            return jsonify({"error": "Model file not found"}), 404

        return send_file(
            PT_MODEL_PATH,
            as_attachment=True,
            download_name="signature_cnn.pt",
            mimetype="application/octet-stream"
        )
    except Exception as e:
        logger.error(f"Model sending error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Add test endpoint - to check if the API is working
@app.route("/test", methods=["GET"])
def test_api():
    return jsonify({"status": "API is working"}), 200


if __name__ == "__main__":
    # Enable CORS for all domains
    CORS(app)
    
    # Configure logging for network access
    logger.info("Starting API server...")
    logger.info("API will be accessible on all network interfaces")
    
    # Run the Flask application
    port = int(os.environ.get('PORT', 5000))
    app.run(
        debug=True,
        host="0.0.0.0",  # Allows external access
        port=port,
        threaded=True    # Enable multiple concurrent connections
    )