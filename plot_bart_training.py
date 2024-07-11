import os
import pandas as pd
import matplotlib.pyplot as plt
#untested


def plot_metrics(log_dir, output_dir):
    # Load the log data
    log_file = os.path.join(log_dir, 'events.out.tfevents.*')  # Adjust the pattern as needed
    logs = pd.read_csv(log_file, sep='\t')

    # Plot training and evaluation loss
    plt.figure(figsize=(12, 6))
    plt.plot(logs['step'], logs['loss'], label='Training Loss')
    plt.plot(logs['step'], logs['eval_loss'], label='Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.show()

    # Plot other metrics if available
    if 'eval_accuracy' in logs.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(logs['step'], logs['eval_accuracy'], label='Validation Accuracy')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
        plt.show()

log_dir = '/scratch4/lhyman6/OCR/work/tuning_results_robust/logs'
output_dir = '/scratch4/lhyman6/OCR/work/tuning_results_robust/plots'
os.makedirs(output_dir, exist_ok=True)
plot_metrics(log_dir, output_dir)
