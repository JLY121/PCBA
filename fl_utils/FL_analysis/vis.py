import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Optional, Tuple

class FLVisualizer:
    """
    Federated Learning Process Visualizer
    Used for real-time plotting of loss and accuracy curves, and marking key moments
    """
    
    def __init__(self, save_dir: str = "../saved/TestResult"):
        """
        Initialize the visualizer
        
        Args:
            save_dir: Directory to save images
        """
        self.save_dir = save_dir
        self.epochs = []
        self.losses = []
        self.accuracies = []
        self.attack_epoch = None
        self.convergence_epoch = None
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10))
        self.fig.suptitle('Federated Learning Process Visualization', fontsize=16, fontweight='bold')
        
        # Set subplot titles
        self.ax1.set_title('Loss Curve', fontsize=14, fontweight='bold')
        self.ax2.set_title('Accuracy Curve', fontsize=14, fontweight='bold')
        
        # Set axis labels
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        
        # Set grid
        self.ax1.grid(True, alpha=0.3)
        self.ax2.grid(True, alpha=0.3)
        
        # Initialize data lines
        self.loss_line, = self.ax1.plot([], [], 'b-', linewidth=2, label='Loss')
        self.acc_line, = self.ax2.plot([], [], 'g-', linewidth=2, label='Accuracy')
        
        # Initialize marker lines
        self.attack_line_loss = None
        self.attack_line_acc = None
        self.convergence_line_loss = None
        self.convergence_line_acc = None
        
        # Set legends
        self.ax1.legend()
        self.ax2.legend()
        
        # Adjust subplot spacing
        plt.tight_layout()
    
    def add_data_point(self, epoch: int, loss: float, accuracy: float):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)
    
    def mark_attack_time(self, epoch: int):
        self.attack_epoch = epoch
        print(f"[VISUALIZATION] Marked attack time: Epoch {epoch}")
    
    def mark_convergence(self, epoch: int):
        self.convergence_epoch = epoch
        print(f"[VISUALIZATION] Marked convergence time: Epoch {epoch}")
    
    def update_plot(self, save_figure: bool = True):
        """
        Update and draw the figure
        
        Args:
            save_figure: Whether to save the figure
        """
        if len(self.epochs) == 0:
            return
        
        # Clear previous marker lines
        try:
            if self.attack_line_loss is not None:
                self.attack_line_loss.remove()
                self.attack_line_acc.remove()
        except:
            pass
        try:
            if self.convergence_line_loss is not None:
                self.convergence_line_loss.remove()
                self.convergence_line_acc.remove()
        except:
            pass
        
        # Update data lines
        self.loss_line.set_data(self.epochs, self.losses)
        self.acc_line.set_data(self.epochs, self.accuracies)
        
        # Set axis ranges
        self.ax1.set_xlim(min(self.epochs), max(self.epochs))
        self.ax2.set_xlim(min(self.epochs), max(self.epochs))
        
        # Set y-axis ranges (with some margin)
        loss_margin = (max(self.losses) - min(self.losses)) * 0.1
        acc_margin = (max(self.accuracies) - min(self.accuracies)) * 0.1
        
        self.ax1.set_ylim(min(self.losses) - loss_margin, max(self.losses) + loss_margin)
        self.ax2.set_ylim(min(self.accuracies) - acc_margin, max(self.accuracies) + acc_margin)
        
        # Add attack time marker lines
        if self.attack_epoch is not None and self.attack_epoch in self.epochs:
            y_min_loss, y_max_loss = self.ax1.get_ylim()
            y_min_acc, y_max_acc = self.ax2.get_ylim()
            
            self.attack_line_loss = self.ax1.axvline(
                x=self.attack_epoch, color='red', linestyle='--', 
                linewidth=2, alpha=0.8, label=f'Attack Time (Epoch {self.attack_epoch})'
            )
            self.attack_line_acc = self.ax2.axvline(
                x=self.attack_epoch, color='red', linestyle='--', 
                linewidth=2, alpha=0.8, label=f'Attack Time (Epoch {self.attack_epoch})'
            )
            
            # 添加文本标注
            self.ax1.text(self.attack_epoch, y_max_loss * 0.9, 
                         f'Attack\nEpoch {self.attack_epoch}', 
                         color='red', fontweight='bold', ha='center', va='top')
            self.ax2.text(self.attack_epoch, y_max_acc * 0.9, 
                         f'Attack\nEpoch {self.attack_epoch}', 
                         color='red', fontweight='bold', ha='center', va='top')
        
        # Add convergence time marker lines
        if self.convergence_epoch is not None and self.convergence_epoch in self.epochs:
            y_min_loss, y_max_loss = self.ax1.get_ylim()
            y_min_acc, y_max_acc = self.ax2.get_ylim()
            
            self.convergence_line_loss = self.ax1.axvline(
                x=self.convergence_epoch, color='blue', linestyle='--', 
                linewidth=2, alpha=0.8, label=f'Convergence Time (Epoch {self.convergence_epoch})'
            )
            self.convergence_line_acc = self.ax2.axvline(
                x=self.convergence_epoch, color='blue', linestyle='--', 
                linewidth=2, alpha=0.8, label=f'Convergence Time (Epoch {self.convergence_epoch})'
            )
            
            # 添加文本标注
            self.ax1.text(self.convergence_epoch, y_max_loss * 0.8, 
                         f'Convergence\nEpoch {self.convergence_epoch}', 
                         color='blue', fontweight='bold', ha='center', va='top')
            self.ax2.text(self.convergence_epoch, y_max_acc * 0.8, 
                         f'Convergence\nEpoch {self.convergence_epoch}', 
                         color='blue', fontweight='bold', ha='center', va='top')
        
        # Update legends
        self.ax1.legend()
        self.ax2.legend()
        
        # Refresh figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Save figure
        if save_figure:
            self.save_figure()
    
    def save_figure(self, filename: str = "fl_training_process.png"):
        filepath = os.path.join(self.save_dir, filename)
        self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[VISUALIZATION] Figure saved: {filepath}")
    
    def close(self):
        plt.close(self.fig)
    
    def get_summary(self) -> dict:
        """
        Get visualization summary
        
        Returns:
            dict: Dictionary containing key information
        """
        summary = {
            'total_epochs': len(self.epochs),
            'final_loss': self.losses[-1] if self.losses else None,
            'final_accuracy': self.accuracies[-1] if self.accuracies else None,
            'attack_epoch': self.attack_epoch,
            'convergence_epoch': self.convergence_epoch,
            'min_loss': min(self.losses) if self.losses else None,
            'max_accuracy': max(self.accuracies) if self.accuracies else None
        }
        return summary 