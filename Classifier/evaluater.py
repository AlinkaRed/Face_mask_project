import os


class AccuracyEvaluator:
    """Класс для оценки точности классификации"""
    
    @staticmethod
    def evaluate_accuracy():
        """
        Сравнивает результаты классификации с правильным распределением
        и вычисляет точность в процентах
        """
        print("=== ОЦЕНКА ТОЧНОСТИ КЛАССИФИКАЦИИ ===")
        
        # Папки для сравнения
        results_dir = "classification_results"
        ground_truth_dir = r"C:\Users\USER\Desktop\Face Mask Dataset\Test_r"
        
        if not os.path.exists(results_dir):
            print(f"Ошибка: папка с результатами {results_dir} не найдена")
            return
        
        if not os.path.exists(ground_truth_dir):
            print(f"Ошибка: папка с правильными ответами {ground_truth_dir} не найдена")
            return
        
        # Собираем информацию о правильной классификации
        ground_truth = {}
        for class_name in ['WithMask', 'WithoutMask']:
            class_dir = os.path.join(ground_truth_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        ground_truth[filename] = class_name
        
        # Собираем информацию о нашей классификации
        our_results = {}
        for class_name in ['WithMask', 'WithoutMask']:
            class_dir = os.path.join(results_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        our_results[filename] = class_name
        
        # Сравниваем результаты
        correct = 0
        total = 0
        mismatched_files = []
        
        for filename, true_class in ground_truth.items():
            if filename in our_results:
                total += 1
                if our_results[filename] == true_class:
                    correct += 1
                else:
                    mismatched_files.append((filename, true_class, our_results[filename]))
        
        # Вычисляем точность
        if total > 0:
            accuracy = (correct / total) * 100
            print(f"\nРезультаты оценки точности:")
            print(f"Всего изображений: {total}")
            print(f"Правильно классифицировано: {correct}")
            print(f"Неправильно классифицировано: {len(mismatched_files)}")
            print(f"Точность: {accuracy:.2f}%")
            
            if mismatched_files:
                print(f"\nНеправильно классифицированные файлы:")
                for filename, true_class, our_class in mismatched_files[:10]:  # Показываем первые 10
                    print(f"  {filename}: должно быть {true_class}, наш результат {our_class}")
                if len(mismatched_files) > 10:
                    print(f"  ... и еще {len(mismatched_files) - 10} файлов")
        else:
            print("Нет файлов для сравнения")