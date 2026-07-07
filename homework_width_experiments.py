import torch
import matplotlib.pyplot as plt
import time
import numpy as np
from datasets import get_mnist_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import count_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_width_experiments():
    """эксперимент с разной шириной сети"""
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    
    configs = {
        'narrow': {
            'layers': [
                {'type': 'linear', 'size': 64},
                {'type': 'relu'},
                {'type': 'linear', 'size': 32},
                {'type': 'relu'},
                {'type': 'linear', 'size': 16},
                {'type': 'relu'},
                {'type': 'linear', 'size': 10}
            ]
        },
        'medium': {
            'layers': [
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 128},
                {'type': 'relu'},
                {'type': 'linear', 'size': 64},
                {'type': 'relu'},
                {'type': 'linear', 'size': 10}
            ]
        },
        'wide': {
            'layers': [
                {'type': 'linear', 'size': 1024},
                {'type': 'relu'},
                {'type': 'linear', 'size': 512},
                {'type': 'relu'},
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 10}
            ]
        },
        'very_wide': {
            'layers': [
                {'type': 'linear', 'size': 2048},
                {'type': 'relu'},
                {'type': 'linear', 'size': 1024},
                {'type': 'relu'},
                {'type': 'linear', 'size': 512},
                {'type': 'relu'},
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
        
        params = count_parameters(model)
        
        start = time.time()
        history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))
        train_time = time.time() - start
        
        results[name] = {
            'history': history,
            'params': params,
            'time': train_time,
            'test_acc': history['test_accs'][-1]
        }
    
    # Визуализация
    plot_width_comparison(results)
    return results

def plot_width_comparison(results):
    """cравнение моделей разной ширины"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    names = list(results.keys())
    
    # test accuracy
    for name in names:
        ax1.plot(results[name]['history']['test_accs'], label=name)
    ax1.set_title('Тесты на Accuracy')
    ax1.set_xlabel('Эпоха')
    ax1.legend()
    ax1.grid(True)
    
    # test loss
    for name in names:
        ax2.plot(results[name]['history']['test_losses'], label=name)
    ax2.set_title('Тесты на Loss')
    ax2.set_xlabel('Эпоха')
    ax2.legend()
    ax2.grid(True)
    
    # params vs accuracy
    params = [results[n]['params'] for n in names]
    accs = [results[n]['test_acc'] for n in names]
    ax3.scatter(params, accs, s=100)
    for i, name in enumerate(names):
        ax3.annotate(name, (params[i], accs[i]), xytext=(5, 5), textcoords='offset points')
    ax3.set_xlabel('Параметры')
    ax3.set_ylabel('Тесты на Accuracy')
    ax3.set_title('Parameters vs Accuracy')
    ax3.grid(True)
    
    # время обучения
    times = [results[n]['time'] for n in names]
    bars = ax4.bar(names, times)
    ax4.set_title('Время обучения')
    ax4.set_ylabel('Секунды')
    ax4.tick_params(axis='x', rotation=45)
    for bar, t in zip(bars, times):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{t:.1f}s', ha='center', va='bottom')
    
    plt.suptitle('WidthExperiments')
    plt.tight_layout()
    plt.savefig('plots/widthExperiments.png', dpi=150)
    plt.show()

def optimize_architecture():
    """Оптимальная архитектурф через grid search"""
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    
    # разные схемы изменения ширины
    configs = {
        'expanding': {
            'layers': [
                {'type': 'linear', 'size': 128},
                {'type': 'relu'},
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 512},
                {'type': 'relu'},
                {'type': 'linear', 'size': 10}
            ]
        },
        'constant': {
            'layers': [
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 10}
            ]
        },
        'narrowing': {
            'layers': [
                {'type': 'linear', 'size': 512},
                {'type': 'relu'},
                {'type': 'linear', 'size': 256},
                {'type': 'relu'},
                {'type': 'linear', 'size': 128},
                {'type': 'relu'},
                {'type': 'linear', 'size': 10}
            ]
        },
        'bottleneck': {
            'layers': [
                {'type': 'linear', 'size': 512},
                {'type': 'relu'},
                {'type': 'linear', 'size': 64},
                {'type': 'relu'},
                {'type': 'linear', 'size': 512},
                {'type': 'relu'},
                {'type': 'linear', 'size': 10}
            ]
        }
    }
    
    results = {}
    
    for name, config in configs.items():
        print(f"\nАрхитектура: {name}")
        
        model = FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers=config['layers']
        ).to(device)
        
        history = train_model(model, train_loader, test_loader, epochs=3, device=str(device))
        results[name] = {
            'history': history,
            'params': count_parameters(model)
        }
    
    # heatmap визуализация
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(results.keys())
    accs = [results[n]['history']['test_accs'][-1] for n in names]
    params = [results[n]['params'] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='skyblue')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, np.array(params)/1000, width, label='Parameters (k)', 
                    color='lightcoral', alpha=0.7)
    
    ax.set_xlabel('Архитектура')
    ax.set_ylabel('Точность')
    ax2.set_ylabel('Параметры (тыс.)')
    ax.set_title('ArchitectureComparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45)
    ax.grid(True, alpha=0.3)
    
    # добавляем сами значения
    for bar, acc in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('plots/architectureOptimization.png', dpi=150)
    plt.show()
    
    # поиск лучшей архитектуры
    best = max(results.items(), key=lambda x: x[1]['history']['test_accs'][-1])
    print(f"\nЛучшая архитектура: {best[0]}")
    
    return results

if __name__ == "__main__":
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Задание 2.1
    print("\n" + "="*60)
    print("Задание 2.1: Сравнение моделей разной ширины")
    print("="*60)
    width_results = run_width_experiments()
    
    # Задание 2.2
    print("\n" + "="*60)
    print("Задание 2.2: Оптимизация архитектуры")
    print("="*60)
    opt_results = optimize_architecture()