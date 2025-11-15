import os
from torchvision import transforms
from torch.utils.data import DataLoader

from config import Config
from data_loader import FaceMaskDataset, CustomMaskDataset
from model import MaskClassifier
from trainer import ModelTrainer
from classifier import ImageClassifier
from evaluater import AccuracyEvaluator


def train_model():
    """Обучение модели с нуля"""
    print("=== ОБУЧЕНИЕ МОДЕЛИ ===")
    
    train_dataset = FaceMaskDataset(
        root_dir=r"C:\Users\USER\Desktop\Face Mask Dataset\Train",
        img_size=Config.IMG_SIZE,
        mode='train'
    )
    train_data = train_dataset.load()
    
    train_custom_dataset = CustomMaskDataset(
        train_data, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]), 
        augment=True
    )
    
    train_loader = DataLoader(train_custom_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    classifier = MaskClassifier()
    trainer = ModelTrainer(classifier)
    trainer.train(train_loader)
    
    classifier.save_model("face_mask_model.pth")
    
    return classifier

def classify_images(model_path: str = None):
    """Классификация изображений и распределение по папкам"""
    print("=== КЛАССИФИКАЦИЯ ИЗОБРАЖЕНИЙ ===")
    
    if model_path is None:
        print("Выберите модель для классификации:")
        print("1. face_mask_classifier_18.pth")
        print("2. face_mask_classifier_34.pth")
        choice = input("Введите 1 или 2: ").strip()
        
        if choice == "1":
            model_path = "face_mask_classifier_18.pth"
        elif choice == "2":
            model_path = "face_mask_classifier_34.pth"
        else:
            print("Неверный выбор, используется face_mask_classifier_18.pth по умолчанию")
            model_path = "face_mask_classifier_18.pth"
    
    if not os.path.exists(model_path):
        print(f"Ошибка: файл модели {model_path} не найден!")
        return
    
    test_dataset = FaceMaskDataset(
        root_dir=r"C:\Users\USER\Desktop\Face Mask Dataset\Test",
        img_size=Config.IMG_SIZE,
        mode='test'
    )
    test_data = test_dataset.load()
    
    classifier = MaskClassifier()
    classifier.load_model(model_path)
    
    image_classifier = ImageClassifier(classifier)
    image_classifier.classify_and_organize(test_data, "classification_results")

def classify_single_image():
    """Классификация одного изображения"""
    print("=== КЛАССИФИКАЦИЯ ОДНОГО ИЗОБРАЖЕНИЯ ===")
    
    image_path = input("Введите путь к изображению: ").strip()
    
    if not os.path.exists(image_path):
        print("Ошибка: файл не существует")
        return
    
    print("Выберите модель для классификации:")
    print("1. face_mask_classifier_18.pth")
    print("2. face_mask_classifier_34.pth")
    choice = input("Введите 1 или 2: ").strip()
    
    if choice == "1":
        model_path = "face_mask_classifier_18.pth"
    elif choice == "2":
        model_path = "face_mask_classifier_34.pth"
    else:
        print("Неверный выбор, используется face_mask_classifier_18.pth по умолчанию")
        model_path = "face_mask_classifier_18.pth"
    
    if not os.path.exists(model_path):
        print(f"Ошибка: файл модели {model_path} не найден")
        return
    
    # Загрузка модели и классификация
    classifier = MaskClassifier()
    classifier.load_model(model_path)
    image_classifier = ImageClassifier(classifier)
    image_classifier.classify_single_image(image_path)

def main():
    """Основное меню программы"""
    while True:
        print("\n" + "="*50)
        print("Классификатор масок на лицах")
        print("="*50)
        print("1. Обучить новую модель")
        print("2. Классифицировать все тестовые изображения")
        print("3. Классифицировать одно изображение")
        print("4. Оценить точность классификации")
        print("5. Выход")
        
        choice = input("\nВыберите действие (1-5): ").strip()
        
        if choice == "1":
            train_model()
            print("\nХотите сразу классифицировать тестовые изображения? (y/n)")
            if input().lower() == 'y':
                classify_images("face_mask_model.pth")
                
        elif choice == "2":
            classify_images()
            
        elif choice == "3":
            classify_single_image()
            
        elif choice == "4":
            AccuracyEvaluator.evaluate_accuracy()
            
        elif choice == "5":
            print("Выход из программы")
            break
            
        else:
            print("Неверный выбор. Пожалуйста, выберите от 1 до 5.")

if __name__ == "__main__":
    main()