import os
from rembg import remove
from PIL import Image
import numpy as np
from io import BytesIO
from scipy import ndimage


def process_and_save_images(input_dir, output_dir):
    """
    İmza görüntülerini işleyerek arka planı kaldırır ve mavi imzayı tamamen korur.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Girdi dizini bulunamadı: {input_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            input_path = os.path.join(root, file)
            rel_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            try:
                # Görüntüyü oku
                with open(input_path, "rb") as f:
                    input_image = f.read()

                # Arka planı kaldır
                output_image = remove(input_image)
                image_pil = Image.open(BytesIO(output_image)).convert("RGBA")
                image_np = np.array(image_pil)

                # RGB kanallarını al
                r = image_np[:, :, 0].astype(float)
                g = image_np[:, :, 1].astype(float)
                b = image_np[:, :, 2].astype(float)
                alpha = image_np[:, :, 3]

                # Çok daha hassas mavi renk tespiti
                blue_stronger_than_red = b > (r + 5)
                blue_stronger_than_green = b > (g + 5)
                min_blue = b > 20
                non_transparent = alpha > 0
                blue_stronger_than_avg = b > ((r + g) / 2 - 10)
                blue_is_max = (b > r) & (b > g)

                # Mantıksal operatörleri düzgün kullan
                is_blue = (
                        ((blue_stronger_than_red) | (blue_stronger_than_green)) &
                        min_blue &
                        non_transparent &
                        (blue_stronger_than_avg | blue_is_max)
                )

                # Kenar koruması için çoklu genişletme ve erozyon
                blue_mask = is_blue
                # Önce genişlet
                blue_mask = ndimage.binary_dilation(blue_mask, iterations=3)
                # Sonra erode et (içerideki boşlukları doldur)
                blue_mask = ndimage.binary_erosion(blue_mask, iterations=1)
                # Tekrar genişlet
                blue_mask = ndimage.binary_dilation(blue_mask, iterations=2)

                # Gaussian bulanıklaştırma ile yumuşat
                blue_mask = ndimage.gaussian_filter(blue_mask.astype(float), sigma=0.7)
                blue_mask = blue_mask > 0.05  # Çok düşük eşik değeri

                # Orijinal görüntüyü koru
                processed = np.copy(image_np)

                # Mavi olmayan alanları şeffaf yap
                processed[~blue_mask] = [0, 0, 0, 0]

                # Sonucu kaydet
                output_image = Image.fromarray(processed)
                output_path = os.path.splitext(output_path)[0] + ".png"
                output_image.save(output_path)
                print(f"Kaydedildi: {output_path}")

            except Exception as e:
                print(f"Hata: {input_path} işlenemedi. {e}")


if __name__ == "__main__":
    input_directory = "Data"
    output_directory = "NewData"
    process_and_save_images(input_directory, output_directory)