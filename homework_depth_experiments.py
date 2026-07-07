import torch
import matplotlib.pyplot as plt
import time
from datasets import get_mnist_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import count_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # у меня CPU

def run_depth_experiments():
    """ эксперимент с разной глубиной сети для определения оптимальной архитектуры"""
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    
    # тут конфигурации с разной глубиной (от 1 до 5 скрытых слоев)
    configs = {
        '1_layer': {  # 0 скрытых слоев, только выходной слой
            'layers': [
                {'type': 'linear', 'size': 10}  # 
            ]
        },
        '2_layers': {  # 1 скрытый слой
            'layers': [
                {'type': 'linear', 'size': 128},  # скрытый слой
                {'type': 'relu'},
                {'type': 'linear', 'size': 10}     # выходной слой
            ]
        },
        '3_layers': {  # 2 скрытых слоя
            'layers': [
                {'type': 'linear', 'size': 256},   # скрытый слой 1
                {'type': 'relu'},
                {'type': 'linear', 'size': 128},   # скрытый слой 2
                {'type': 'relu'},
                {'type': 'linear', 'size': 10}     # выходной слой
            ]
        },
        '5_layers': {  # 4 скрытых слоя и т.д
            'layers': [
                {'type': 'linear', 'size': 512},
                {'type': 'relu'},
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 128},
                {'type': 'relu'},
                {'type': 'linear', 'size': 64},
                {'type': 'relu'},
                {'type': 'linear', 'size': 10}
            ]
        },
        '7_layers': {
            'layers': [
                {'type': 'linear', 'size': 512},
                {'type': 'relu'},
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 128},
                {'type': 'relu'},
                {'type': 'linear', 'size': 64},
                {'type': 'relu'},
                {'type': 'linear', 'size': 32},
                {'type': 'relu'},
                {'type': 'linear', 'size': 16},
                {'type': 'relu'},
                {'type': 'linear', 'size': 10}
            ]
        }
    }
    
    results = {}
    
    # Обучаем каждую модель
    for name, config in configs.items():
        print(f"Модель: {name}")
        
        # задаём архитектуру
        model = FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers=config['layers']
        ).to(device)
        
        start = time.time()
        history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))
        train_time = time.time() - start
        
        results[name] = {
            'history': history,
            'params': count_parameters(model),
            'time': train_time
        }
    
    """ Визуализация, графики """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    names = list(results.keys())
    
    # точность тестов
    for name in names:
        ax1.plot(results[name]['history']['test_accs'], label=name)
        ax1.set_title('Точность')

    ax1.set_xlabel('Эпоха')
    ax1.legend()
    ax1.grid(True)
    
    # Разница train-test (переобучение)
    for name in names:
        h = results[name]['history']
        diff = [t - s for t, s in zip(h['train_accs'], h['test_accs'])]
        ax2.plot(diff, label=name)

    ax2.set_title('Переобучение (разница Train - Test)')
    ax2.set_xlabel('Эпоха')
    ax2.legend()
    ax2.grid(True)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Параметры
    params = [results[n]['params'] for n in names]
    bars = ax3.bar(names, params)
    ax3.set_title('Параметры')
    ax3.set_ylabel('Количество')
    ax3.tick_params(axis='x', rotation=45)
    for bar, p in zip(bars, params):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{p:,}', ha='center', va='bottom')
    
    # Время обучения
    times = [results[n]['time'] for n in names]
    bars = ax4.bar(names, times)
    ax4.set_title('Время обучения')
    ax4.set_ylabel('Секунды')
    ax4.tick_params(axis='x', rotation=45)
    for bar, t in zip(bars, times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{t:.1f}s', ha='center', va='bottom')
    
    plt.suptitle('DepthExperiments')
    plt.tight_layout()
    plt.savefig('plots/depthExperiments.png', dpi=150)
    plt.show()
    
    return results

def run_depth_with_regularization(): # с регуляризацией
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    
    # 5 слоев
    configs = {
        'reg': {
            'layers': [
                {'type': 'linear', 'size': 512},
                {'type': 'relu'},
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 128},
                {'type': 'relu'},
                {'type': 'linear', 'size': 64},
                {'type': 'relu'},
                {'type': 'linear', 'size': 10}
            ]
        },
        'dropout': {
            'layers': [
                {'type': 'linear', 'size': 512},
                {'type': 'relu'},
                {'type': 'dropout', 'rate': 0.3},
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'dropout', 'rate': 0.3},
                {'type': 'linear', 'size': 128},
                {'type': 'relu'},
                {'type': 'dropout', 'rate': 0.3},
                {'type': 'linear', 'size': 64},
                {'type': 'relu'},
                {'type': 'dropout', 'rate': 0.3},
                {'type': 'linear', 'size': 10}
            ]
        },
        'batchnorm': {
            'layers': [
                {'type': 'linear', 'size': 512},
                {'type': 'batch_norm'},
                {'type': 'relu'},
                {'type': 'linear', 'size': 256},
                {'type': 'batch_norm'},
                {'type': 'relu'},
                {'type': 'linear', 'size': 128},
                {'type': 'batch_norm'},
                {'type': 'relu'},
                {'type': 'linear', 'size': 64},
                {'type': 'batch_norm'},
                {'type': 'relu'},
                {'type': 'linear', 'size': 10}
            ]
        },
        'both': {
            'layers': [
                {'type': 'linear', 'size': 512},
                {'type': 'batch_norm'},
                {'type': 'relu'},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'linear', 'size': 256},
                {'type': 'batch_norm'},
                {'type': 'relu'},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'linear', 'size': 128},
                {'type': 'batch_norm'},
                {'type': 'relu'},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'linear', 'size': 64},
                {'type': 'batch_norm'},
                {'type': 'relu'},
                {'type': 'dropout', 'rate': 0.2},
                {'type': 'linear', 'size': 10}
            ]
        }
    }
    
    results = {}

    for name, config in configs.items():
        print(f"Модель: {name}")
        
        model = FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers=config['layers']
        ).to(device)
        
        # Обучаем модель
        history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))
        results[name] = history
    
    # Визуализация с plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 2 Графика: Test и Train точность
    for name, history in results.items():
        ax1.plot(history['test_accs'], label=name)
        ax2.plot(history['test_losses'], label=name)
    
    ax1.set_title('Accuracy с Регуляризацией')
    ax1.set_xlabel('Эпоха')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Loss с Регуляризацией')
    ax2.set_xlabel('Эпоха')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle('RegularizationForDeep')
    plt.tight_layout()
    plt.savefig('plots/regularizationForDeep.png', dpi=150)
    plt.show()
    
    return results

if __name__ == "__main__":
    import os
    os.makedirs('plots', exist_ok=True)
    print("\n" + "="*60)
    print("Задание 1.1: Сравнение моделей разной глубины")
    print("="*60)
    
    depth_results = run_depth_experiments()
    
    print("\n" + "="*60)
    print("Задание 1.2: Анализ переобучения с регуляризацией")
    print("="*60)

    reg_results = run_depth_with_regularization()