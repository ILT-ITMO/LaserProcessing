import torch
from pinn import PINN


def save_model(model, path='model.pth', optimizer=None, epoch=None, loss=None):
    """
    Сохраняет модель и дополнительную информацию
    
    Args:
        model: модель для сохранения
        path: путь к файлу
        optimizer: оптимизатор (опционально)
        epoch: номер эпохи (опционально)
        loss: значение loss (опционально)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_architecture': [4, 128, 128, 128, 1]  # сохраняем архитектуру
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    
    torch.save(checkpoint, path)
    print(f"Модель сохранена: {path}")

def load_model(path='model.pth', device='cpu', optimizer=None):
    """
    Загружает модель и дополнительную информацию
    
    Args:
        path: путь к файлу
        device: устройство для загрузки
        optimizer: оптимизатор для восстановления состояния (опционально)
    
    Returns:
        model: загруженная модель
        checkpoint: полный чекпоинт с дополнительной информацией
    """
    checkpoint = torch.load(path, map_location=device)
    
    # Создаем модель с сохраненной архитектурой
    model = PINN(checkpoint['model_architecture'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Восстанавливаем состояние оптимизатора если предоставлен
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Модель загружена: {path}")
    if 'epoch' in checkpoint:
        print(f"Эпоха: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"Loss: {checkpoint['loss']:.4e}")
    
    return model, checkpoint