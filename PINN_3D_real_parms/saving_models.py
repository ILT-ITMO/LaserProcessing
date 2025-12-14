import torch
from pinn import PINN


def save_model(model, path='model.pth', optimizer=None, epoch=None, loss=None):
    """
    Saves the model and related training information to a file.
    
    This allows resuming training from a specific point or using a pre-trained model.
    The model's architecture is also saved to ensure consistent reconstruction.
    
    Args:
        model: The PyTorch model instance to be saved.
        path (str, optional): The file path to save the model checkpoint. Defaults to 'model.pth'.
        optimizer (torch.optim.Optimizer, optional): The optimizer instance used during training. 
                                                     If provided, its state is also saved. Defaults to None.
        epoch (int, optional): The current epoch number. If provided, it's saved in the checkpoint. Defaults to None.
        loss (float, optional): The current loss value. If provided, it's saved in the checkpoint. Defaults to None.
    
    Returns:
        None
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
    Loads a pre-trained model and associated information from a checkpoint file.
    
    This method reconstructs the model architecture, loads the trained weights,
    and optionally restores the optimizer state, allowing for resuming training
    or performing inference with a previously saved model.  It facilitates 
    reusing and continuing work from saved states, avoiding redundant training.
    
    Args:
        path (str): The path to the checkpoint file (e.g., 'model.pth').
        device (str): The device to load the model onto (e.g., 'cpu', 'cuda').
        optimizer (torch.optim.Optimizer, optional): An optimizer instance. 
            If provided and the checkpoint contains optimizer state, the 
            optimizer's state is loaded. Defaults to None.
    
    Returns:
        tuple: A tuple containing:
            - model: The loaded PyTorch model.
            - checkpoint: The complete checkpoint dictionary loaded from the file,
              containing model architecture, state dictionary, and potentially
              optimizer state, epoch number, and loss value.
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