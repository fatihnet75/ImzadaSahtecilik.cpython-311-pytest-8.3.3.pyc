import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# Sabit değişkenler
IMG_HEIGHT, IMG_WIDTH = 128, 128
CLASSES = ["Fake", "Real"]
MODEL_PATH = "signature_cnn.pth"
BEST_MODEL_PATH = "signature_cnn_best.pth"


class SignatureDataset(Dataset):
    def __init__(self, directory, classes, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.classes = classes
        self.load_data(directory)

    def load_data(self, directory):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Veri dizini bulunamadı: {directory}")

        for person in os.listdir(directory):
            person_dir = os.path.join(directory, person)
            if not os.path.isdir(person_dir):
                continue

            for label in self.classes:
                class_dir = os.path.join(person_dir, label)
                if not os.path.exists(class_dir):
                    continue

                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Uyarı: {img_path} yüklenemedi.")
                        continue
                    img = self.resize_and_pad(img)
                    self.images.append(img)
                    self.labels.append(1 if label == "Real" else 0)

    def resize_and_pad(self, image):
        # Resmi normalize et (kontrast artırma)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        # Gürültü azaltma
        image = cv2.GaussianBlur(image, (3, 3), 0)

        h, w = image.shape
        scale = min(IMG_WIDTH / w, IMG_HEIGHT / h)
        resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))
        delta_w = IMG_WIDTH - resized_image.shape[1]
        delta_h = IMG_HEIGHT - resized_image.shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return padded_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0
            image = torch.tensor(image)
        label = torch.tensor([label], dtype=torch.float32)
        return image, label


# Veri dönüşümleri - Eğitim için daha güçlü augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Test için minimal dönüşüm
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class SignatureCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SignatureCNN, self).__init__()

        # Feature Extraction Blocks
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (IMG_HEIGHT // 16) * (IMG_WIDTH // 16), 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {-val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path, device='cpu'):
    model = SignatureCNN()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None,
                epochs=60, device='cpu', early_stopping_patience=10):
    model.to(device)
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).long()
            correct += (predicted == labels.long()).sum().item()
            total += labels.size(0)

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item(), 'acc': correct / total})

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (predicted == labels.long()).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_accuracy = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)

        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_epoch_loss)

        # Print statistics
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        print(f"Val   - Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy:.4f}")

        # Early stopping
        early_stopping(val_epoch_loss, model, BEST_MODEL_PATH)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }


def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(1)
            predicted = (probs > 0.5).long()

            all_labels.extend(labels.cpu().numpy().flatten())
            all_predictions.extend(predicted.cpu().numpy().flatten())
            all_probabilities.extend(probs.cpu().numpy().flatten())

    # Metrics calculation
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Print results
    print("\n===== Test Results =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Format confusion matrix
    if len(cm) >= 2:
        TN, FP, FN, TP = cm.ravel()
        print("\n===== Confusion Matrix =====")
        print(f"True Positive (Real imzalar doğru tahmin): {TP}")
        print(f"True Negative (Sahte imzalar doğru tahmin): {TN}")
        print(f"False Positive (Sahte imzalar yanlış tahmin - Type II error): {FP}")
        print(f"False Negative (Gerçek imzalar yanlış tahmin - Type I error): {FN}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'probabilities': all_probabilities,
        'predictions': all_predictions,
        'true_labels': all_labels
    }


def plot_metrics(history):
    plt.figure(figsize=(15, 10))

    # Plot training & validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['val_losses'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training & validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_accuracies'], label='Training Accuracy')
    plt.plot(history['val_accuracies'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()


def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    plt.show()


if __name__ == "__main__":
    # Cihaz belirleme
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Veri yükle ve dönüşümleri uygula
    dataset = SignatureDataset("NewData/train", CLASSES)
    test_dataset = SignatureDataset("NewData/test", CLASSES, transform=test_transform)

    # Test ve doğrulama için ayrılma
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    generator = torch.Generator().manual_seed(42)  # Tekrarlanabilirlik için
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Dönüşümleri ayarla (random_split'ten sonra)
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform

    # DataLoader'ları oluştur
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Sınıf dengesizliğini kontrol et
    num_real = sum(label.item() == 1 for _, label in train_dataset)
    num_fake = len(train_dataset) - num_real
    pos_weight = torch.tensor([num_fake / num_real], dtype=torch.float32).to(device)

    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    print(f"Class distribution in training - Real: {num_real}, Fake: {num_fake}")

    # Model oluştur
    model = SignatureCNN(dropout_rate=0.5)

    # Loss ve optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Eğitim
    print("Starting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=60,
        device=device,
        early_stopping_patience=10
    )

    # Eğitim sonrası modeli kaydet
    save_model(model, MODEL_PATH)

    # Grafikleri çiz
    plot_metrics(history)

    # En iyi modeli yükle ve test et
    best_model = load_model(BEST_MODEL_PATH, device)
    test_results = evaluate_model(best_model, test_loader, device)

    # Confusion matrix'i çiz
    plot_confusion_matrix(test_results['confusion_matrix'], CLASSES)

    print("\nTraining and evaluation completed!")