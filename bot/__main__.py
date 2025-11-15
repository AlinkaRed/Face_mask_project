import db_client
import tg_client
import time
import io
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Classifier'))

from model import MaskClassifier
import torch
from PIL import Image
import numpy as np


class BotImageClassifier:
    """Классификатор для Telegram бота"""
    
    def __init__(self, model_path: str = "../Classifier/face_mask_classifier_18.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.classifier = None
        self._load_model()
    
    def _load_model(self):
        """Загружает модель для классификации"""
        try:
            self.classifier = MaskClassifier()
            self.classifier.load_model(self.model_path)
            print(f"Модель {self.model_path} загружена для бота")
        except Exception as e:
            print(f"Ошибка загрузки модели для бота: {e}")
    
    def classify_image(self, image_bytes: bytes) -> str:
        """Классифицирует изображение и возвращает результат"""
        if self.classifier is None:
            return "Модель не загружена"
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert('RGB')
            image = image.resize((224, 224))
            
            transform = self.classifier.transform['test']
            tensor = transform(image).unsqueeze(0).to(self.device)
            
            self.classifier.model.eval()
            with torch.no_grad():
                outputs = self.classifier.model(tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                class_name = "в маске 🟢" if predicted.item() == 0 else "без маски 🔴"
                confidence = probabilities[0][predicted.item()].item() * 100
                
                return f"Результат: человек {class_name}"
                
        except Exception as e:
            return f"Ошибка обработки изображения: {str(e)}"


def main() -> None:
    next_up_offset = 0
    classifier = BotImageClassifier()
    
    try:
        while True:
            updates = tg_client.getUpdates(next_up_offset)
            db_client.persist_updates(updates)
            
            for update in updates:
                chat_id = update["message"]["chat"]["id"]
                
                if "photo" in update["message"]:
                    photo = update["message"]["photo"][-1]
                    file_id = photo["file_id"]
                    
                    file_info = tg_client.getFile(file_id)
                    file_path = file_info["file_path"]
                    
                    image_bytes = tg_client.download_file(file_path)
                    
                    result = classifier.classify_image(image_bytes)
                    
                    tg_client.sendMessage(
                        chat_id=chat_id,
                        text=f"📸 Анализ фото:\n{result}"
                    )
                    
                else:
                    tg_client.sendMessage(
                        chat_id=chat_id,
                        text="Привет! Отправь мне фото человека, и я определю, в маске он или нет👨‍⚕️"
                    )
                
                print(".", end="", flush=True)
                
                next_up_offset = update["update_id"] + 1
            
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nBye!")


if __name__ == "__main__":
    main()