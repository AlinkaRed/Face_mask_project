import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import Dict
from config import Config


class MaskClassifier:
    """Классификатор для определения масок на лицах"""
    
    def __init__(self, model_name: str = Config.MODEL_NAME, pretrained: bool = Config.PRETRAINED):
        self.device = Config.DEVICE
        self.model = self._initialize_model(model_name, pretrained)
        self.transform = self._get_transforms()
        
    def _initialize_model(self, model_name: str, pretrained: bool) -> nn.Module:
        """Инициализирует модель"""
        model_func = getattr(models, model_name)
        model = model_func(pretrained=pretrained)
        
        # Заменяем последний слой для бинарной классификации
        if hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.in_features, 2)
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Linear):
                model.classifier = nn.Linear(model.classifier.in_features, 2)
        
        return model.to(self.device)
    
    def _get_transforms(self) -> Dict[str, transforms.Compose]:
        """Возвращает трансформации для изображений"""
        return {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }
    
    def save_model(self, filepath: str):
        """Сохраняет модель"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, filepath)
        print(f"Модель сохранена в {filepath}")
    
    def load_model(self, filepath: str):
        """Загружает модель"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Модель загружена из {filepath}")