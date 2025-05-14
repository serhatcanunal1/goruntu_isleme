import sys
import requests
import json
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from PIL import Image
import torch
from torchvision import transforms, models


class ImageClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hayvan Sınıflandırıcı")
        self.setFixedSize(800, 700)

        self.model = self.load_model()
        self.labels = self.load_labels()
        self.image_path = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        self.image_label = QLabel("Görsel buraya yüklenecek")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(400, 400)
        self.image_label.setStyleSheet("""
            background-color: #2c2c2e;
            border: 2px dashed #5e5ce6;
            border-radius: 20px;
            color: #888;
        """)

        image_layout = QHBoxLayout()
        image_layout.addStretch()
        image_layout.addWidget(self.image_label)
        image_layout.addStretch()

        self.top_class_label = QLabel("En olası sınıf: -")
        self.top_class_label.setAlignment(Qt.AlignCenter)
        self.top_class_label.setFont(QFont("Segoe UI", 26, QFont.Bold))
        self.top_class_label.setStyleSheet("""
            color: #fdd835;
            padding: 10px;
        """)

        load_button = QPushButton("Görsel Yükle")
        load_button.setStyleSheet("""
            QPushButton {
                background-color: #5e5ce6;
                color: white;
                font-size: 18px;
                border-radius: 20px;
                padding: 15px 40px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #7c4dff;
                border: 2px solid #ffffff;
            }
        """)
        load_button.clicked.connect(self.load_image)

        classify_button = QPushButton("Sınıflandır")
        classify_button.setStyleSheet("""
            QPushButton {
                background-color: #5e5ce6;
                color: white;
                font-size: 18px;
                border-radius: 20px;
                padding: 15px 40px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #7c4dff;
                border: 2px solid #ffffff;
            }
        """)
        classify_button.clicked.connect(self.classify_image)

        layout = QVBoxLayout()
        layout.addLayout(image_layout)
        layout.addWidget(load_button, alignment=Qt.AlignCenter)
        layout.addWidget(classify_button, alignment=Qt.AlignCenter)
        layout.addWidget(self.top_class_label)

        container = QWidget()
        container.setLayout(layout)

        main_layout.addWidget(container)
        self.setCentralWidget(QWidget(self))
        self.centralWidget().setLayout(main_layout)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QWidget {
                font-family: 'Segoe UI';
                font-size: 18px;
                color: #E0E0E0;
            }
        """)

    def load_model(self):
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.eval()
        return model

    def load_labels(self):
        url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"HTTP Hatası: {response.status_code}")
            return {}
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError as e:
            print(f"JSON Hatası: {e}")
            return {}
        return {int(k): v[1] for k, v in data.items()}

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Görsel Seç", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name).scaled(400, 400, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.top_class_label.setText("En olası sınıf: -")

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image)
        return input_tensor.unsqueeze(0)

    def classify_image(self):
        if not self.image_path:
            self.top_class_label.setText("Lütfen önce bir görsel yükleyin.")
            return

        input_batch = self.preprocess_image(self.image_path)

        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)

        class_id = top_catid[0].item()
        label = self.labels.get(class_id, "Bilinmiyor")
        prob = top_prob[0].item() * 100

        self.top_class_label.setText(f"En olası sınıf: {label} ({prob:.2f}%)")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifierApp()
    window.show()
    sys.exit(app.exec_())
