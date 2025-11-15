import torch


class Config:
    MODEL_NAME = 'resnet18'
    PRETRAINED = True
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    IMG_SIZE = (224, 224)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")