
import io
import timm
import matplotlib.pyplot as plt
from ultralytics import YOLO
import requests
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog,QSplitter
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import nltk
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
nltk.download('words')
from nltk.corpus import words
correct_words = words.words()
text_det_model_path = 'best_s_200.pt'
yolo = YOLO(text_det_model_path)

char_to_idx = {'-': 1, '0': 2, '1': 3, '2': 4, '3': 5, '4': 6, '5': 7, '6': 8, '7': 9, '8': 10, '9': 11, 'a': 12, 'b': 13, 'c': 14, 'd': 15, 'e': 16, 'f': 17, 'g': 18, 'h': 19, 'i': 20, 'j': 21, 'k': 22, 'l': 23, 'm': 24, 'n': 25, 'o': 26, 'p': 27, 'q': 28, 'r': 29, 's': 30, 't': 31, 'u': 32, 'v': 33, 'w': 34, 'x': 35, 'y': 36, 'z': 37}
idx_to_char = {1: '-', 2: '0', 3: '1', 4: '2', 5: '3', 6: '4', 7: '5', 8: '6', 9: '7', 10: '8', 11: '9', 12: 'a', 13: 'b', 14: 'c', 15: 'd', 16: 'e', 17: 'f', 18: 'g', 19: 'h', 20: 'i', 21: 'j', 22: 'k', 23: 'l', 24: 'm', 25: 'n', 26: 'o', 27: 'p', 28: 'q', 29: 'r', 30: 's', 31: 't', 32: 'u', 33: 'v', 34: 'w', 35: 'x', 36: 'y', 37: 'z'}

def text_detection(img_path,text_det_model):
    text_det_results = text_det_model(img_path,verbose=False)[0]
    bboxes = text_det_results.boxes.xyxy.tolist()
    classes = text_det_results.boxes.cls.tolist()
    names = text_det_results.names
    confs = text_det_results.boxes.conf.tolist()

    return bboxes, classes,names,confs

def decode(encoded_sequences, idx_to_char, blank_char='-'):
    decode_sequences = []
    for seq in encoded_sequences:
        decoded_label = []
        for idx, token in enumerate(seq):
            if token !=0:
                char = idx_to_char[token.item()]
                if char != blank_char:
                    decoded_label.append(char)
        decode_sequences.append(''.join(decoded_label))
    return decode_sequences

def text_recognition(img,data_transforms, text_reg_model, idx_to_char, device):
    transformed_image = data_transforms(img)
    transformed_image = transformed_image.unsqueeze(0).to(device)
    text_reg_model.eval()
    with torch.no_grad():
        logits = text_reg_model(transformed_image).detach().cpu()
    text = decode(logits.permute(1,0,2).argmax(2), idx_to_char)
    # nltk
    word = text[0]
    temp = [(jaccard_distance(set(ngrams(word, 2)),
                              set(ngrams(w, 2))), w)
            for w in correct_words if w[0] == word[0]]
    if temp:
        text_return = []
        text_return.append(sorted(temp, key=lambda val: val[0])[0][1])
        return text_return
    else:
        print("No similar words found for:", word)
        return text
    return text

def remove_duplicates(words):
    word = words[0]
    cleaned_word = ''
    prev_char = ''
    for char in word:
        if char != prev_char:
            cleaned_word += char
            prev_char = char
    return cleaned_word


def visualize_detections(img, detections):
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')

    for bbox, detected_class, confidence, transcribed_text in detections:
        x1, y1, x2, y2 = bbox
        plt.gca().add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor="red", linewidth=2
            )
        )
        plt.text(
            x1, y1 - 10, f"{detected_class} ({confidence:.2f}):{transcribed_text}",
            fontsize=9, bbox=dict(facecolor='red', alpha=0.5)
        )

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    pixmap = QPixmap()
    pixmap.loadFromData(buffer.getvalue())
    buffer.close()

    return pixmap


def predict(img_path, data_transforms, text_det_model, text_reg_model, idx_to_char, device):
    bboxes, classes, names, confs = text_detection(img_path, text_det_model)

    img = Image.open(img_path)
    predictions = []

    for bbox, cls, conf in zip(bboxes, classes, confs):
        x1,y1,x2,y2 = bbox
        confidence = conf
        detected_class = cls
        name = names[int(cls)]

        cropped_image = img.crop((x1,y1,x2,y2))

        transcribed_text = text_recognition(
            cropped_image,
            data_transforms,
            text_reg_model,
            idx_to_char,
            device,
        )

        predictions.append((bbox,name, confidence, transcribed_text))

    visualize_detections(img,predictions)

    return predictions

class CRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        n_layers,
        dropout=0.2,
        unfreeze_layers=3
    ):
        super(CRNN, self).__init__()

        backbone = timm.create_model(
            'resnet101',
            in_chans=1,
            pretrained=True
        )

        modules = list(backbone.children())[:-2]

        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)

        for parameter in self.backbone[-unfreeze_layers:].parameters():
            parameter.requires_grad = True

        self.mapSeq = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            1024, hidden_size,
            n_layers, bidirectional=True, batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, vocab_size),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.mapSeq(x)
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        x = self.out(x)
        x = x.permute(1, 0, 2)

        return x

hidden_size = 256
n_layers = 2
dropout_prob = 0.3
unfreeze_layers=3
device = 'cpu'
vocab_size = 37

crnn_model = CRNN(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    n_layers=n_layers,
    dropout=dropout_prob,
    unfreeze_layers=unfreeze_layers
).to(device)

save_model_path = 'ocr_crnn_resnet_best_3.pt'
crnn_model.load_state_dict(torch.load(save_model_path, map_location=device))

data_transforms  = {
    'train': transforms.Compose([
        transforms.Resize((100,420)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.Grayscale(num_output_channels=1),
        transforms.GaussianBlur(3),
        transforms.RandomAffine(degrees=1, shear=1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3, interpolation=3),
        transforms.RandomRotation(degrees=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ]),
    "val": transforms.Compose([
        transforms.Resize((100,420)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
}

class ImageSelector(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.image_paths = []  # Danh sách chứa đường dẫn ảnh đã chọn

    def initUI(self):
        self.setWindowTitle("Chọn ảnh và phân tích văn bản")
        self.setGeometry(100, 100, 800, 600)

        main_layout = QHBoxLayout()

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.left_label = QLabel("Chưa chọn ảnh")
        self.left_label.setStyleSheet("border: 2px solid black; text-align: center;")
        left_button = QPushButton("Chọn ảnh")
        left_button.clicked.connect(self.selectImage)
        left_layout.addWidget(self.left_label)
        left_layout.addWidget(left_button)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        top_right_widget = QWidget()
        top_right_layout = QVBoxLayout(top_right_widget)
        self.top_right_image_label = QLabel()
        self.top_right_image_label.setStyleSheet("border: 2px solid black;")
        top_right_layout.addWidget(self.top_right_image_label)

        bottom_right_widget = QWidget()
        bottom_right_layout = QVBoxLayout(bottom_right_widget)
        self.text_label = QLabel("Văn bản phân tích:")
        self.text_label.setStyleSheet("border: 2px solid black; font-size: 18px; padding: 10px; text-align: center;")
        bottom_right_layout.addWidget(self.text_label)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)

        right_layout.addWidget(top_right_widget)
        right_layout.addWidget(bottom_right_widget)

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def selectImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "All Files (*);;Image Files (*.jpg *.png *.jpeg)", options=options)
        if fileName:
            self.image_paths.append(fileName)  # Thêm đường dẫn ảnh vào danh sách
            self.processImages()  # Tiến hành phân tích ảnh

    def processImages(self):
        if self.image_paths:
            img_path = self.image_paths.pop(0)  # Lấy đường dẫn ảnh đầu tiên trong danh sách
            inf_transforms = data_transforms['val']
            predictions = predict(img_path, data_transforms=inf_transforms, text_det_model=yolo, text_reg_model=crnn_model, idx_to_char=idx_to_char, device=device)
            self.displayImage(img_path)
            self.displayTopRightImage(img_path)
            self.displayText(predictions)
            self.displayVisualizedImage(img_path,predictions)
    def displayTranslate(self,word):
        url = "https://google-translate1.p.rapidapi.com/language/translate/v2"

        payload = {
            "q": word,
            "target": "vi",
            "source": "en"
        }
        headers = {
            "content-type": "application/x-www-form-urlencoded",
            "Accept-Encoding": "application/gzip",
            "X-RapidAPI-Key": "f5efa31a8fmshea51248983501ddp19cdaejsn29532745f902",
            "X-RapidAPI-Host": "google-translate1.p.rapidapi.com"
        }

        response = requests.post(url, data=payload, headers=headers)
        return response.json()["data"]["translations"][0]["translatedText"]

    def displayVisualizedImage(self, img_path, predictions):
        pixmap = visualize_detections(Image.open(img_path), predictions)
        self.top_right_image_label.setPixmap(pixmap)
        self.top_right_image_label.setAlignment(Qt.AlignCenter)

    def displayImage(self, img_path):
        pixmap = QPixmap(img_path)
        self.left_label.setPixmap(pixmap)
        self.left_label.setAlignment(Qt.AlignCenter)

    def displayTopRightImage(self, img_path):
        pixmap = QPixmap(img_path)
        self.top_right_image_label.setPixmap(pixmap)
        self.top_right_image_label.setAlignment(Qt.AlignCenter)

    def displayText(self, predictions):
        text = ""
        for _, _, _, transcribed_text in predictions:
            text += transcribed_text[0]+": "+self.displayTranslate(transcribed_text[0]) + "\n"
        self.text_label.setText("Văn bản phân tích:\n" + text )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    selector = ImageSelector()
    selector.show()
    sys.exit(app.exec_())
