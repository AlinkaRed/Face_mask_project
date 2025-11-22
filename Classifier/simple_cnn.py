import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class SimpleCNN(nn.Module):
    """Простая CNN модель для классификации масок"""
    
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 28 * 28, 512) 
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x))) 
        
        x = x.view(-1, 128 * 28 * 28)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class SimpleMaskClassifier:
    """Классификатор на основе простой CNN"""
    
    def __init__(self):
        self.device = Config.DEVICE
        self.model = SimpleCNN().to(self.device)
        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        """Возвращает трансформации для изображений"""
        from torchvision import transforms
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
        print(f"Простая CNN модель сохранена в {filepath}")
    
    def load_model(self, filepath: str):
        """Загружает модель"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Простая CNN модель загружена из {filepath}")