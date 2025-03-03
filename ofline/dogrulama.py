import torch
import os
import sys


def verify_model(model_path):
    """
    Model dosyasını doğrula ve bilgileri yazdır
    """
    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı: {model_path}")
        return False

    try:
        print(f"PyTorch Sürümü: {torch.__version__}")
        print(f"Model dosyası boyutu: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")

        # Modeli yüklemeyi dene
        model = torch.load(model_path, map_location=torch.device('cpu'))
        print(f"Model başarıyla yüklendi!")

        # Model türünü kontrol et
        if isinstance(model, torch.nn.Module):
            print(f"Model türü: PyTorch nn.Module")
            print(f"Model mimarisi:")
            print(model)
            return True
        elif isinstance(model, dict):
            print(f"Model türü: state_dict")
            print(f"state_dict anahtarları:")
            for key in model.keys():
                print(f"  {key}: {model[key].shape}")
            return True
        else:
            print(f"Model türü: {type(model)}")
            return True

    except Exception as e:
        print(f"HATA: Model yüklenirken bir hata oluştu: {e}")
        return False


if __name__ == "__main__":
    # Model yolu parametresi
    model_path = "model/signature_cnn.pth"

    # Komut satırından yol verilmişse onu kullan
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    # Modeli doğrula
    if verify_model(model_path):
        print("Model doğrulama başarılı.")
        sys.exit(0)
    else:
        print("Model doğrulama başarısız.")
        sys.exit(1)