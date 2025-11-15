import os
import shutil
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy as np
from config import Config


class ImageClassifier:
    """Класс для классификации изображений"""
    
    def __init__(self, classifier):
        self.classifier = classifier
        self.device = Config.DEVICE
        
    def classify_and_organize(self, test_data: list, output_dir: str = "mask_classification_results"):
        """
        Классифицирует изображения и распределяет по папкам
        WithMask - человек в маске
        WithoutMask - человек без маски
        """
        self.classifier.model.eval()
        
        # Создаем папки для результатов
        with_mask_dir = os.path.join(output_dir, "WithMask")
        without_mask_dir = os.path.join(output_dir, "WithoutMask")
        os.makedirs(with_mask_dir, exist_ok=True)
        os.makedirs(without_mask_dir, exist_ok=True)
        
        with_mask_count = 0
        without_mask_count = 0
        
        with torch.no_grad():
            for item in tqdm(test_data, desc="Классификация изображений"):
                try:
                    # Преобразуем изображение
                    image = Image.fromarray(item['image'])
                    tensor = self.classifier.transform['test'](image).unsqueeze(0).to(self.device)
                    
                    # Предсказание
                    outputs = self.classifier.model(tensor)
                    _, predicted = torch.max(outputs, 1)
                    
                    # Определяем класс
                    if predicted.item() == 0:  # WithMask
                        dest_dir = with_mask_dir
                        with_mask_count += 1
                    else:  # WithoutMask
                        dest_dir = without_mask_dir
                        without_mask_count += 1
                    
                    # Копируем файл в соответствующую папку
                    src_path = item['filepath']
                    dst_path = os.path.join(dest_dir, item['filename'])
                    shutil.copy2(src_path, dst_path)
                    
                except Exception as e:
                    print(f"Ошибка обработки {item['filename']}: {str(e)}")
        
        print(f"\nКлассификация завершена!")
        print(f"Результаты сохранены в папке: {output_dir}")
        print(f"WithMask (в маске): {with_mask_count} изображений")
        print(f"WithoutMask (без маски): {without_mask_count} изображений")
    
    def classify_single_image(self, image_path: str):
        """
        Классифицирует одно изображение и выводит результат
        """
        self.classifier.model.eval()
        
        if not os.path.exists(image_path):
            print(f"Ошибка: файл {image_path} не найден")
            return
        
        try:
            # Загружаем и обрабатываем изображение
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img = img.resize(Config.IMG_SIZE)
                tensor = self.classifier.transform['test'](img).unsqueeze(0).to(self.device)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.classifier.model(tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                class_name = "WithMask" if predicted.item() == 0 else "WithoutMask"
                confidence = probabilities[0][predicted.item()].item() * 100
                
                print(f"\nРезультат классификации:")
                print(f"Изображение: {os.path.basename(image_path)}")
                print(f"Класс: {class_name}")
                print(f"Уверенность: {confidence:.2f}%")
                
                # Показываем изображение
                plt.figure(figsize=(8, 6))
                plt.imshow(np.array(img))
                plt.title(f"Результат: {class_name} ({confidence:.2f}%)")
                plt.axis('off')
                plt.show()
                
        except Exception as e:
            print(f"Ошибка обработки изображения: {str(e)}")