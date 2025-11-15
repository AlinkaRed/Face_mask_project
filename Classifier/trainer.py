import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import Config


class ModelTrainer:
    """Класс для обучения модели"""
    
    def __init__(self, classifier):
        self.classifier = classifier
        self.device = Config.DEVICE
        
    def train(self, train_loader, num_epochs: int = Config.NUM_EPOCHS):
        """Обучение модели"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.classifier.model.parameters(), lr=Config.LEARNING_RATE)
        
        for epoch in range(num_epochs):
            self.classifier.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.classifier.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
            train_loss = running_loss / len(train_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")