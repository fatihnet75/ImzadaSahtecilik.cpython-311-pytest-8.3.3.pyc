import torch
from torchviz import make_dot
import torch.nn as nn

# CNN Modeli
class SignatureCNN(nn.Module):
    def __init__(self):
        super(SignatureCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * (128 // 16) * (128 // 16), 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(nn.LeakyReLU()(self.batchnorm1(self.conv1(x))))
        x = self.pool(nn.LeakyReLU()(self.batchnorm2(self.conv2(x))))
        x = self.pool(nn.LeakyReLU()(self.batchnorm3(self.conv3(x))))
        x = self.pool(nn.LeakyReLU()(self.batchnorm4(self.conv4(x))))

        x = x.view(-1, 256 * (128 // 16) * (128 // 16))

        x = nn.LeakyReLU()(self.fc1(x))
        x = self.dropout(x)
        x = nn.LeakyReLU()(self.fc2(x))
        x = self.fc3(x)
        return x

# Modeli oluştur
model = SignatureCNN()

# Dummy input (1 sample, 1 channel, 128x128 image)
dummy_input = torch.randn(1, 1, 128, 128)

# Modeli çalıştır
output = model(dummy_input)

# Modelin yapısını görselleştir
dot = make_dot(output, params=dict(model.named_parameters()))

# Görselleştirilmiş modeli kaydet
dot.render("cnn_model", format="png")

print("Model yapısı cnn_model.png olarak kaydedildi.")
