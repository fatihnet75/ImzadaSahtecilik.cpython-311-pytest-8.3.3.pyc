from flask import Flask, request, send_file, jsonify
import os
from datetime import datetime
import time
import uuid
from werkzeug.utils import secure_filename
import logging
import csv
import pandas as pd
import numpy as np
from pathlib import Path
import json
import torch
from flask_cors import CORS
from logging.handlers import RotatingFileHandler
import traceback

# Güvenlik için dotenv ekleyelim
from dotenv import load_dotenv

# Özel modülleri import edelim (SignatureCNN ve SignatureTrainer sınıfları için)
# Not: Bu dosyaları uygun konumda oluşturmanız gerekiyor
# signature_model.py içinde önceki kodda tanımladığımız sınıfları içermelidir
try:
    from signature_model import SignatureCNN, SignatureTrainer

    MODEL_IMPORTED = True
except ImportError:
    MODEL_IMPORTED = False
    print("Uyarı: signature_model.py dosyası bulunamadı. Tahmin özellikleri devre dışı.")

# .env dosyasını yükle (varsa)
load_dotenv()

# Flask uygulaması
app = Flask(__name__)
# CORS desteği ekleyelim (gerekirse özelleştirilebilir)
CORS(app)


# Yapılandırma ayarları
class Config:
    """Uygulama yapılandırma sınıfı"""
    # Klasör yapılandırması
    BASE_FOLDER = os.getenv("BASE_FOLDER", "uploads")
    CSV_FOLDER = os.getenv("CSV_FOLDER", "features")
    MODEL_FOLDER = os.getenv("MODEL_FOLDER", "models")
    LOG_FOLDER = os.getenv("LOG_FOLDER", "logs")

    # Model dosyası
    MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(MODEL_FOLDER, "signature_cnn_model.pth"))

    # İzin verilen dosya uzantıları
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'svg', 'pdf'}

    # Sunucu ayarları
    DEBUG = os.getenv("DEBUG", "True").lower() in ('true', '1', 't')
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))

    # Güvenlik ayarları
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))  # 16MB

    # Rate limiting - basit bir uygulama için (production için daha karmaşık çözümler gerekebilir)
    RATE_LIMIT = int(os.getenv("RATE_LIMIT", 100))  # Her IP için maksimum istek sayısı
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 3600))  # Saniye cinsinden pencere (1 saat)


# Yapılandırmayı uygula
app.config.from_object(Config)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Klasörleri oluştur
os.makedirs(Config.BASE_FOLDER, exist_ok=True)
os.makedirs(Config.CSV_FOLDER, exist_ok=True)
os.makedirs(Config.MODEL_FOLDER, exist_ok=True)
os.makedirs(Config.LOG_FOLDER, exist_ok=True)

# Logging ayarları - Gelişmiş ayarlar
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# Konsol logger
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Dosya logger (rotating file handler)
file_handler = RotatingFileHandler(
    os.path.join(Config.LOG_FOLDER, 'app.log'),
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(log_formatter)

# Root logger yapılandırması
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Flask uygulaması için özel logger
logger = logging.getLogger('signature_api')

# Rate limiting için basit in-memory çözüm
# Production ortamında Redis gibi bir çözüm kullanılmalıdır
rate_limit_store = {}


# Helper fonksiyonlar
def allowed_file(filename):
    """Dosya uzantısı kontrolü"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def create_csv_folder():
    """CSV dosyaları için klasör oluştur"""
    csv_path = Path(Config.CSV_FOLDER)
    csv_path.mkdir(parents=True, exist_ok=True)
    return csv_path


def rate_limit_check(ip_address):
    """
    Basit bir rate limiting kontrolü
    Returns:
        bool: Rate limiti aşıldıysa False, aşılmadıysa True
    """
    current_time = time.time()

    # İlk kez gelen IP için kayıt oluştur
    if ip_address not in rate_limit_store:
        rate_limit_store[ip_address] = {
            'count': 1,
            'reset_time': current_time + Config.RATE_LIMIT_WINDOW
        }
        return True

    # Süresi dolan limitleri sıfırla
    if current_time > rate_limit_store[ip_address]['reset_time']:
        rate_limit_store[ip_address] = {
            'count': 1,
            'reset_time': current_time + Config.RATE_LIMIT_WINDOW
        }
        return True

    # Limit kontrolü
    if rate_limit_store[ip_address]['count'] >= Config.RATE_LIMIT:
        return False

    # Sayacı artır
    rate_limit_store[ip_address]['count'] += 1
    return True


def save_features_to_csv(name, features):
    """
    İmza özelliklerini CSV dosyasına kaydet

    Args:
        name (str): Özellik seti adı
        features (list): Özellik dictionaries listesi

    Returns:
        tuple: (success, file_path, message)
    """
    try:
        csv_folder = create_csv_folder()
        filename = secure_filename(f"{name}_features.csv")
        csv_file = csv_folder / filename

        # Özellik isimlerini al
        feature_names = list(features[0].keys()) if features else []

        if not feature_names:
            return False, None, "No feature keys found in the provided data"

        # CSV dosyası var mı kontrol et
        file_exists = csv_file.exists()

        mode = 'a' if file_exists else 'w'
        with open(csv_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=feature_names)

            # Başlıkları sadece yeni dosya oluşturulurken yaz
            if not file_exists:
                writer.writeheader()

            # Özellikleri yaz
            for feature_set in features:
                writer.writerow(feature_set)

        logger.info(f"Features saved to CSV for {name}")
        return True, str(csv_file), "Features saved successfully"

    except Exception as e:
        logger.error(f"Error saving features to CSV: {str(e)}")
        logger.error(traceback.format_exc())
        return False, None, str(e)


def validate_features(features):
    """
    Özellik verilerini doğrula

    Args:
        features (list): Doğrulanacak özellikler

    Returns:
        tuple: (valid, error_message)
    """
    if not isinstance(features, list):
        return False, "Features must be a list"

    if not features:
        return False, "Features list is empty"

    # Tüm öğelerin sözlük olduğundan emin ol
    if not all(isinstance(item, dict) for item in features):
        return False, "All items in features must be dictionaries"

    # İlk özellik setindeki anahtarları al
    first_keys = set(features[0].keys())

    # Tüm özellik setlerinin aynı anahtarlara sahip olduğunu kontrol et
    for i, feature_set in enumerate(features[1:], 1):
        if set(feature_set.keys()) != first_keys:
            return False, f"Feature set at index {i} has different keys than the first feature set"

    # Sayısal değer kontrolü
    for i, feature_set in enumerate(features):
        for key, value in feature_set.items():
            if value is not None and not isinstance(value, (int, float)):
                try:
                    float(value)  # Dönüştürülebilirlik kontrolü
                except (ValueError, TypeError):
                    return False, f"Non-numeric value '{value}' for key '{key}' in feature set at index {i}"

    return True, ""


# Model yükleme fonksiyonu
def load_model():
    """
    Eğitilmiş modeli yükle (varsa)

    Returns:
        tuple: (model, trainer, success, error_message)
    """
    if not MODEL_IMPORTED:
        return None, None, False, "Model module not imported"

    try:
        model_path = Path(Config.MODEL_PATH)
        if not model_path.exists():
            return None, None, False, f"Model file not found at {model_path}"

        trainer = SignatureTrainer()
        model = trainer.load_model(str(model_path))

        return model, trainer, True, ""
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, False, str(e)


# Endpoint tanımlamaları için middleware işlevi
@app.before_request
def before_request():
    """Her istekten önce çalışacak middleware"""
    # Rate limiting kontrolü
    if not rate_limit_check(request.remote_addr):
        return jsonify({
            "error": "Rate limit exceeded. Please try again later."
        }), 429

    # İstek logları
    logger.info(f"Request: {request.method} {request.path} - IP: {request.remote_addr}")


# Error handler
@app.errorhandler(Exception)
def handle_exception(e):
    """Genel hata yakalayıcı"""
    logger.error(f"Unhandled exception: {str(e)}")
    logger.error(traceback.format_exc())
    return jsonify({
        "error": "An unexpected error occurred",
        "message": str(e)
    }), 500


# API endpoint'leri
@app.route("/", methods=["GET"])
def index():
    """API ana sayfası"""
    return jsonify({
        "name": "Signature Analysis API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "POST /features/upload": "Upload signature features",
            "GET /features/<name>": "Get features for a specific name",
            "GET /features": "List all available feature sets",
            "POST /predict": "Make predictions using trained model"
        }
    })


@app.route("/features/upload", methods=["POST"])
def upload_features():
    """İmza özelliklerini al ve CSV'ye kaydet"""
    try:
        # JSON içeriği kontrol et
        if not request.is_json:
            return jsonify({
                "error": "Request must be JSON"
            }), 400

        data = request.get_json()

        # Gerekli alanları kontrol et
        if not data or 'name' not in data or 'features' not in data:
            return jsonify({
                "error": "Missing required fields. Please provide 'name' and 'features'"
            }), 400

        name = data['name']

        # Güvenlik için ad kontrolü
        if not name or not isinstance(name, str):
            return jsonify({
                "error": "Invalid name. Name must be a non-empty string."
            }), 400

        # Güvenli dosya adı oluştur
        safe_name = secure_filename(name)
        if not safe_name:
            return jsonify({
                "error": "Invalid name. Please use a valid name with allowed characters."
            }), 400

        features = data['features']

        # Özellikleri doğrula
        valid, error_msg = validate_features(features)
        if not valid:
            return jsonify({
                "error": error_msg
            }), 400

        # Özellikleri CSV'ye kaydet
        success, file_path, message = save_features_to_csv(safe_name, features)

        if success:
            return jsonify({
                "message": message,
                "name": safe_name,
                "feature_count": len(features),
                "csv_path": file_path,
                "timestamp": datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                "error": f"Failed to save features: {message}"
            }), 500

    except Exception as e:
        logger.error(f"Error processing features upload: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/features/<name>", methods=["GET"])
def get_features(name):
    """Belirli bir ismin özelliklerini CSV'den getir"""
    try:
        # Güvenli dosya adı
        safe_name = secure_filename(name)
        if not safe_name:
            return jsonify({
                "error": "Invalid name format"
            }), 400

        # CSV dosyasını bul
        csv_file = Path(Config.CSV_FOLDER) / f"{safe_name}_features.csv"

        if not csv_file.exists():
            return jsonify({
                "error": f"No features found for {safe_name}"
            }), 404

        # CSV'yi oku
        df = pd.read_csv(csv_file)

        # NaN değerleri None'a dönüştür (JSON serialization için)
        df = df.replace({np.nan: None})

        return jsonify({
            "name": safe_name,
            "feature_count": len(df),
            "features": df.to_dict('records'),
            "columns": df.columns.tolist()
        }), 200

    except Exception as e:
        logger.error(f"Error retrieving features: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/features", methods=["GET"])
def list_features():
    """Mevcut tüm özellik setlerini listele"""
    try:
        csv_folder = Path(Config.CSV_FOLDER)
        if not csv_folder.exists():
            return jsonify({
                "error": "Features folder not found"
            }), 404

        # Tüm CSV dosyalarını bul
        csv_files = list(csv_folder.glob("*_features.csv"))

        feature_sets = []
        for csv_file in csv_files:
            # Dosya adından ismi çıkar
            name = csv_file.name.replace("_features.csv", "")

            # Dosya bilgilerini al
            stats = csv_file.stat()

            # CSV hakkında temel bilgileri al
            try:
                df = pd.read_csv(csv_file)
                row_count = len(df)
                column_count = len(df.columns)
            except Exception:
                row_count = 0
                column_count = 0

            feature_sets.append({
                "name": name,
                "file": str(csv_file),
                "row_count": row_count,
                "column_count": column_count,
                "size_bytes": stats.st_size,
                "modified": datetime.fromtimestamp(stats.st_mtime).isoformat()
            })

        return jsonify({
            "feature_sets": feature_sets,
            "count": len(feature_sets)
        }), 200

    except Exception as e:
        logger.error(f"Error listing features: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Modeli kullanarak tahmin yap"""
    # Model modülünün import edildiğini kontrol et
    if not MODEL_IMPORTED:
        return jsonify({
            "error": "Prediction functionality is not available. Model module not imported."
        }), 501

    try:
        # JSON içeriği kontrol et
        if not request.is_json:
            return jsonify({
                "error": "Request must be JSON"
            }), 400

        data = request.get_json()

        # Gerekli alanları kontrol et
        if not data or ('features' not in data and 'name' not in data):
            return jsonify({
                "error": "Missing required fields. Please provide 'features' or 'name'"
            }), 400

        # Modeli yükle
        model, trainer, success, error_msg = load_model()
        if not success:
            return jsonify({
                "error": f"Failed to load model: {error_msg}"
            }), 500

        # Tahmin için verileri hazırla
        if 'features' in data:
            # Doğrudan özelliklerle tahmin
            features = data['features']

            # Özellikleri doğrula
            valid, error_msg = validate_features([features] if isinstance(features, dict) else features)
            if not valid:
                return jsonify({
                    "error": error_msg
                }), 400

            if isinstance(features, dict):
                # Tek bir özellik seti
                df = pd.DataFrame([features])
            else:
                # Özellik seti listesi
                df = pd.DataFrame(features)

        elif 'name' in data:
            # İsimden CSV dosyasını yükle
            name = secure_filename(data['name'])
            csv_file = Path(Config.CSV_FOLDER) / f"{name}_features.csv"

            if not csv_file.exists():
                return jsonify({
                    "error": f"No features found for {name}"
                }), 404

            df = pd.read_csv(csv_file)

        # Tahmin yap
        try:
            predictions = trainer.predict(df.values)

            # Sonuçları hazırla
            result = []
            for i, pred in enumerate(predictions):
                result.append({
                    "index": i,
                    "input": df.iloc[i].to_dict(),
                    "prediction": pred.tolist()
                })

            return jsonify({
                "success": True,
                "predictions": result,
                "count": len(result)
            }), 200

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "error": f"Prediction failed: {str(e)}"
            }), 500

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/train", methods=["POST"])
def train_model():
    """Model eğitimi başlatmak için API endpoint'i"""
    # Model modülünün import edildiğini kontrol et
    if not MODEL_IMPORTED:
        return jsonify({
            "error": "Training functionality is not available. Model module not imported."
        }), 501

    try:
        # JSON içeriği kontrol et
        if not request.is_json:
            return jsonify({
                "error": "Request must be JSON"
            }), 400

        data = request.get_json()

        # Gerekli alanları kontrol et
        if not data or 'csv_path' not in data:
            return jsonify({
                "error": "Missing required field 'csv_path'"
            }), 400

        # Eğitim parametrelerini al
        csv_path = data['csv_path']
        epochs = int(data.get('epochs', 100))
        batch_size = int(data.get('batch_size', 32))
        learning_rate = float(data.get('learning_rate', 0.001))
        patience = int(data.get('patience', 10))

        # CSV dosyasının varlığını kontrol et
        if not os.path.exists(csv_path):
            return jsonify({
                "error": f"CSV file not found: {csv_path}"
            }), 404

        # Eğiticiyi oluştur
        trainer = SignatureTrainer(model_save_path=Config.MODEL_PATH)

        # Asenkron eğitim başlat (gerçek uygulamada bu kısım
        # ayrı bir thread veya Celery gibi bir görev kuyruğunda çalıştırılabilir)
        # Bu örnekte senkron eğitim yapıyoruz, uzun sürebilir!
        try:
            model, best_loss = trainer.train(
                csv_path=csv_path,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                patience=patience
            )

            return jsonify({
                "success": True,
                "message": "Model training completed successfully",
                "model_path": str(Config.MODEL_PATH),
                "best_loss": best_loss,
                "training_params": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "patience": patience
                }
            }), 200

        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                "error": f"Training failed: {str(e)}"
            }), 500

    except Exception as e:
        logger.error(f"Error during training request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/model/status", methods=["GET"])
def model_status():
    """Model durumunu kontrol et"""
    try:
        model_path = Path(Config.MODEL_PATH)

        if not model_path.exists():
            return jsonify({
                "exists": False,
                "message": "Model file not found"
            }), 404

        # Model bilgilerini al
        model_stats = model_path.stat()

        # Model yüklemeyi dene
        if MODEL_IMPORTED:
            try:
                model, trainer, success, error_msg = load_model()
                model_loaded = success
                error = error_msg if not success else None
            except Exception as e:
                model_loaded = False
                error = str(e)
        else:
            model_loaded = False
            error = "Model module not available"

        return jsonify({
            "exists": True,
            "file_path": str(model_path),
            "size_bytes": model_stats.st_size,
            "modified": datetime.fromtimestamp(model_stats.st_mtime).isoformat(),
            "loaded": model_loaded,
            "error": error
        }), 200

    except Exception as e:
        logger.error(f"Error checking model status: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """API sağlık kontrolü"""
    memory_usage = "Not available"
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_usage = f"{process.memory_info().rss / 1024 / 1024:.2f} MB"
    except ImportError:
        pass

    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": time.time(),
        "memory_usage": memory_usage
    })


if __name__ == "__main__":
    # Başlangıç bilgisi
    logger.info(f"Starting Signature Analysis API on {Config.HOST}:{Config.PORT}")
    logger.info(f"Debug mode: {Config.DEBUG}")
    logger.info(f"Upload folder: {Config.BASE_FOLDER}")
    logger.info(f"CSV folder: {Config.CSV_FOLDER}")
    logger.info(f"Model folder: {Config.MODEL_FOLDER}")
    logger.info(f"Model file: {Config.MODEL_PATH}")

    # Gerekli klasörleri oluştur
    os.makedirs(Config.BASE_FOLDER, exist_ok=True)
    os.makedirs(Config.CSV_FOLDER, exist_ok=True)
    os.makedirs(Config.MODEL_FOLDER, exist_ok=True)

    # Model durumunu kontrol et
    model_path = Path(Config.MODEL_PATH)
    if model_path.exists():
        logger.info(f"Model file found: {model_path}")
    else:
        logger.warning(f"Model file not found: {model_path}")

    # Uygulamayı başlat
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)