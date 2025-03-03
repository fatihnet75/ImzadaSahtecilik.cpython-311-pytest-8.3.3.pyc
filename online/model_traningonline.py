import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import os

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SignatureDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        """
        İmza verileri için veri kümesi sınıfı

        Args:
            features (numpy.ndarray): Özellik verileri
            labels (numpy.ndarray): Etiket verileri
            transform (callable, optional): Verilere uygulanacak dönüşüm
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        if self.transform:
            feature = self.transform(feature)

        return feature, label


class SignatureCNN(nn.Module):
    def __init__(self, input_features, output_features, dropout_rate=0.5):
        """
        İmza analizi için CNN modeli

        Args:
            input_features (int): Giriş özelliklerinin sayısı
            output_features (int): Çıkış özelliklerinin sayısı
            dropout_rate (float): Dropout oranı
        """
        super(SignatureCNN, self).__init__()

        # Giriş verilerini 2D tensöre dönüştürmek için özellik boyutlarını hesapla
        self.feature_size = int(np.sqrt(input_features))
        # Eğer tam kare değilse, bir üst kareye yuvarla
        if self.feature_size * self.feature_size < input_features:
            self.feature_size += 1

        self.reshape_size = self.feature_size * self.feature_size

        # Geliştirilmiş konvolüsyon katmanları
        self.conv_layers = nn.Sequential(
            # İlk katman: (batch_size, 1, feature_size, feature_size)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),  # Daha iyi gradyan akışı için LeakyReLU
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),  # 2D Dropout eklendi

            # İkinci katman
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            # Üçüncü katman
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),

            # Ek Residual bağlantı katmanı
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
        )

        # Çıkış boyutunu hesapla
        # feature_size // 8 yerine, tam sayı bölme yaparak hesaplama
        downsample_factor = 2 ** 3  # 3 adet MaxPool2d(2) katmanı için
        conv_output_dim = self.feature_size // downsample_factor
        if conv_output_dim == 0:  # Çok küçük feature_size için kontrol
            conv_output_dim = 1

        conv_output_size = 128 * conv_output_dim * conv_output_dim

        # Geliştirilmiş tam bağlantılı katmanlar
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate * 0.8),

            nn.Linear(128, output_features)
        )

        # Modeli başlatma
        self._initialize_weights()

    def _initialize_weights(self):
        """Ağırlıkları He başlatma yöntemi ile başlat"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        # Giriş verilerini yeniden boyutlandır, eksik değerleri 0 ile doldur
        if x.size(1) < self.reshape_size:
            padding = torch.zeros(batch_size, self.reshape_size - x.size(1), device=x.device)
            x = torch.cat([x, padding], dim=1)
        elif x.size(1) > self.reshape_size:
            x = x[:, :self.reshape_size]  # Fazla özellikleri kes

        # 2D tensöre dönüştür
        x = x.view(batch_size, 1, self.feature_size, self.feature_size)

        # CNN katmanlarından geçir
        x = self.conv_layers(x)

        # Flatten
        x = x.view(batch_size, -1)

        # Tam bağlantılı katmanlardan geçir
        x = self.fc_layers(x)

        return x


class SignatureTrainer:
    def __init__(self, model_dir="signature_models"):
        """
        İmza analizi modelini eğitmek için sınıf

        Args:
            model_dir (str): Model ve log dosyalarının kaydedileceği dizin
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)

        self.model_save_path = self.model_dir / "signature_cnn_model.pth"
        self.scaler = StandardScaler()

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.model_dir / "logs"))

        logger.info(f"Device: {self.device}")
        logger.info(f"Model will be saved to: {self.model_save_path}")

    def prepare_data(self, csv_path):
        """
        CSV dosyasından veriyi hazırla

        Args:
            csv_path (str): CSV dosyasının yolu

        Returns:
            tuple: Eğitim ve test verileri, özellik sayısı
        """
        logger.info(f"Loading data from {csv_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # CSV'yi oku
        df = pd.read_csv(csv_path)
        logger.info(f"Data loaded: {df.shape[0]} samples, {df.shape[1]} columns")

        # İlk sütun ad ise, onu ayır
        if df.columns[0].lower() in ['name', 'id', 'identifier', 'imza_id']:
            feature_names = df.columns[1:]
            logger.info(f"First column ({df.columns[0]}) excluded as identifier")
        else:
            feature_names = df.columns
            logger.info("All columns used as features")

        X = df[feature_names].values
        y = df[feature_names].values  # Aynı değerleri öğrenecek (autoencoder)

        # Eksik değerleri doldur
        if np.isnan(X).any():
            logger.warning(f"Data contains {np.isnan(X).sum()} missing values. Filling with zeros.")
            X = np.nan_to_num(X, nan=0.0)

        # Veriyi ölçeklendir
        X = self.scaler.fit_transform(X)

        # Eğitim ve test verilerini ayır
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True)

        logger.info(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")

        return X_train, X_test, y_train, y_test, len(feature_names)

    def create_data_loaders(self, X_train, X_test, y_train, y_test, batch_size=32):
        """
        Veri yükleyicilerini oluştur

        Args:
            X_train, X_test, y_train, y_test: Eğitim ve test verileri
            batch_size (int): Mini-batch boyutu

        Returns:
            tuple: Eğitim ve test veri yükleyicileri
        """
        # Veri kümelerini oluştur
        train_dataset = SignatureDataset(X_train, y_train)
        test_dataset = SignatureDataset(X_test, y_test)

        # DataLoader'ları oluştur
        # num_workers sayısını sisteme göre otomatik ayarla
        num_workers = min(4, os.cpu_count() or 1)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        logger.info(f"Created data loaders with batch size {batch_size} and {num_workers} workers")

        return train_loader, test_loader

    def train(self, csv_path, epochs=100, batch_size=32, learning_rate=0.001, patience=10):
        """
        Model eğitimi

        Args:
            csv_path (str): CSV dosyasının yolu
            epochs (int): Eğitim döngüsü sayısı
            batch_size (int): Mini-batch boyutu
            learning_rate (float): Başlangıç öğrenme oranı
            patience (int): Erken durdurma sabırlılığı
        """
        start_time = time.time()
        logger.info(f"Starting training with {epochs} epochs, batch size {batch_size}, learning rate {learning_rate}")

        # Veriyi hazırla
        X_train, X_test, y_train, y_test, num_features = self.prepare_data(csv_path)
        train_loader, test_loader = self.create_data_loaders(
            X_train, X_test, y_train, y_test, batch_size)

        # Model oluştur
        model = SignatureCNN(num_features, num_features).to(self.device)
        logger.info(f"Model created with {num_features} input/output features")

        # Model parametrelerinin sayısını hesapla
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total model parameters: {total_params:,}")

        # TensorBoard'a model yapısını ekle
        example_input = torch.zeros((1, num_features), device=self.device)
        self.writer.add_graph(model, example_input)

        # Kayıp fonksiyonu - MSE yanında L1 kaybı da ekle
        criterion = lambda outputs, targets: (
                nn.MSELoss()(outputs, targets) + 0.1 * nn.L1Loss()(outputs, targets)
        )

        # Optimizer - AdamW kullan (daha iyi weight decay)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # Öğrenme oranı programlayıcısı - CosineAnnealingWarmRestarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=learning_rate / 100
        )

        # Metrics takibi
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        # Eğitim döngüsü
        for epoch in range(epochs):
            epoch_start_time = time.time()

            # Eğitim
            model.train()
            train_loss = 0
            for batch_idx, (features, labels) in enumerate(train_loader):
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # Gradient clipping - aşırı gradyan birikimini önle
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

                # Mini-batch progress
                if batch_idx % 10 == 0:
                    logger.debug(
                        f'Epoch: {epoch + 1}/{epochs} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}')

            # Doğrulama
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for features, labels in test_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(features)
                    val_loss += criterion(outputs, labels).item()

            # Ortalama kayıpları hesapla
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)

            # Kayıpları kaydet
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Öğrenme oranını güncelle
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            # TensorBoard'a metrikleri ekle
            self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/validation', avg_val_loss, epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)

            # Örnek tahmin görselleştir
            if epoch % 10 == 0:
                with torch.no_grad():
                    example_batch = next(iter(test_loader))
                    example_features = example_batch[0].to(self.device)
                    example_labels = example_batch[1].to(self.device)
                    example_outputs = model(example_features)

                    # İlk örneği al
                    example_feature = example_features[0].cpu().numpy()
                    example_label = example_labels[0].cpu().numpy()
                    example_output = example_outputs[0].cpu().numpy()

                    # Karşılaştırma grafiği
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(example_label, label='Gerçek', alpha=0.7)
                    ax.plot(example_output, label='Tahmin', alpha=0.7)
                    ax.set_title(f'Epoch {epoch + 1} - Tahmin vs Gerçek')
                    ax.legend()

                    self.writer.add_figure('Predictions', fig, epoch)

            # Geçen süreyi hesapla
            epoch_time = time.time() - epoch_start_time

            # Sonuçları yazdır
            logger.info(f'Epoch [{epoch + 1}/{epochs}] - '
                        f'Train Loss: {avg_train_loss:.6f}, '
                        f'Val Loss: {avg_val_loss:.6f}, '
                        f'LR: {current_lr:.6f}, '
                        f'Time: {epoch_time:.2f}s')

            # En iyi modeli kaydet
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0

                # Modeli kaydet
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': best_loss,
                    'scaler': self.scaler,  # Ölçekleyiciyi de kaydet
                    'input_features': num_features,
                    'output_features': num_features,
                }, self.model_save_path)

                logger.info(f'Model saved to {self.model_save_path} (val_loss: {best_loss:.6f})')
            else:
                patience_counter += 1

            # Erken durdurma kontrolü
            if patience_counter >= patience:
                logger.info(f'Early stopping after {epoch + 1} epochs (no improvement for {patience} epochs)')
                break

        # Eğitim tamamlandı
        total_time = time.time() - start_time
        logger.info(f'Training completed in {total_time / 60:.2f} minutes')
        logger.info(f'Best validation loss: {best_loss:.6f}')

        # Eğitim grafiklerini kaydet
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Eğitim Kaybı')
        plt.plot(val_losses, label='Doğrulama Kaybı')
        plt.xlabel('Epoch')
        plt.ylabel('Kayıp')
        plt.legend()
        plt.title('Eğitim ve Doğrulama Kayıpları')
        plt.savefig(str(self.model_dir / "training_loss.png"))

        # TensorBoard'a son grafiği ekle
        self.writer.add_figure('Training/Loss', plt.gcf())
        self.writer.close()

        return model, best_loss

    def predict(self, features, model_path=None):
        """
        Eğitilmiş model ile tahmin yap

        Args:
            features (numpy.ndarray): Tahmin edilecek özellikler
            model_path (str, optional): Model dosyasının yolu. None ise varsayılan yol kullanılır.

        Returns:
            numpy.ndarray: Tahmin edilen değerler
        """
        if model_path is None:
            model_path = self.model_save_path

        # Modeli yükle
        checkpoint = torch.load(model_path, map_location=self.device)

        # Model parametrelerini al
        input_features = checkpoint.get('input_features')
        output_features = checkpoint.get('output_features')

        # Eğer parametreler yoksa, varsayılan değerleri kullan
        if input_features is None or output_features is None:
            input_features = output_features = features.shape[1]
            logger.warning("Model parameters not found in checkpoint, using default values")

        # Modeli oluştur ve ağırlıkları yükle
        model = SignatureCNN(input_features, output_features).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Scaler'ı yükle
        if 'scaler' in checkpoint:
            self.scaler = checkpoint['scaler']

        # Özellikleri ölçeklendir
        scaled_features = self.scaler.transform(features)

        # Tahmin yap
        model.eval()
        with torch.no_grad():
            tensor_features = torch.FloatTensor(scaled_features).to(self.device)
            predictions = model(tensor_features)

        # NumPy dizisine dönüştür
        return predictions.cpu().numpy()


def main():
    """Ana fonksiyon"""
    # Komut satırı argümanları için argparse eklenebilir (şu an basit tutuyoruz)

    # CSV dosyasının yolunu belirt
    csv_path = "features/signature_features.csv"

    # Model dizini
    model_dir = "signature_models"

    # Eğitim parametreleri
    epochs = 150
    batch_size = 64
    learning_rate = 0.001
    patience = 15

    # Eğiticiyi oluştur ve eğitimi başlat
    trainer = SignatureTrainer(model_dir=model_dir)

    try:
        model, best_loss = trainer.train(
            csv_path=csv_path,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience
        )
        logger.info(f"Training completed successfully with best loss: {best_loss:.6f}")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()