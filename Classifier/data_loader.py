import os
import random
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FaceMaskDataset:
    """
    Загрузка изображений для классификации масок
    """
    def __init__(self, root_dir: str, img_size: Tuple[int, int] = (224, 224), mode: str = 'train'):
        self.root_dir = root_dir
        self.img_size = img_size
        self.mode = mode
        self.dataset = []
        self.class_to_idx = {'WithMask': 0, 'WithoutMask': 1}
        self.idx_to_class = {0: 'WithMask', 1: 'WithoutMask'}
        
    def load(self) -> List[Dict]:
        """Основной метод загрузки данных"""
        if self.mode == 'train':
            return self._load_train_data()
        else:
            return self._load_test_data()
    
    def _load_train_data(self) -> List[Dict]:
        """Загрузка тренировочных данных"""
        for class_name in ['WithMask', 'WithoutMask']:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Предупреждение: папка {class_dir} не найдена")
                continue
                
            for filename in os.listdir(class_dir):
                if self._is_image_file(filename):
                    filepath = os.path.join(class_dir, filename)
                    try:
                        img = self._process_image(filepath)
                        if img is not None:
                            self.dataset.append({
                                'image': img,
                                'class_idx': self.class_to_idx[class_name],
                                'class_name': class_name,
                                'filename': filename,
                                'filepath': filepath
                            })
                    except Exception as e:
                        print(f"Ошибка загрузки {filepath}: {str(e)}")
        
        print(f"Загружено {len(self.dataset)} изображений")
        return self.dataset
    
    def _load_test_data(self) -> List[Dict]:
        """Загрузка тестовых данных"""
        for filename in os.listdir(self.root_dir):
            filepath = os.path.join(self.root_dir, filename)
            if not os.path.isfile(filepath):
                continue
                
            if self._is_image_file(filename):
                try:
                    img = self._process_image(filepath)
                    self.dataset.append({
                        'image': img,
                        'class_idx': -1,
                        'class_name': 'unknown',
                        'filename': filename,
                        'filepath': filepath
                    })
                except Exception as e:
                    print(f"Ошибка загрузки {filepath}: {str(e)}")
        
        print(f"Загружено {len(self.dataset)} тестовых изображений")
        return self.dataset
    
    def _process_image(self, path: str) -> np.ndarray:
        """Загружает и обрабатывает изображение"""
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
                img = img.resize(self.img_size)
                return np.array(img)
        except Exception as e:
            print(f"Ошибка обработки {path}: {str(e)}")
            return None
    
    def _is_image_file(self, filename: str) -> bool:
        """Проверяет, является ли файл изображением"""
        return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))

class CustomMaskDataset(Dataset):
    """Датасет для изображений с масками"""
    
    def __init__(self, data, transform=None, augment: bool = False):
        self.data = data
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_array = img_data['image']
        label = img_data['class_idx']
        
        img = Image.fromarray(img_array)
        
        if self.augment:
            # Простые аугментации
            if random.random() < 0.5:
                img = transforms.functional.hflip(img)
            if random.random() < 0.3:
                img = transforms.functional.rotate(img, angle=random.uniform(-15, 15))
        
        if self.transform:
            img = self.transform(img)
            
        return img, label