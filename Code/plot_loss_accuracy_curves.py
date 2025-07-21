import matplotlib.pyplot as plt

def plot_loss_accuracy_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path="loss_accuracy_curve.png"):
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_loss_train = 'tab:blue'
    color_loss_val = 'tab:cyan'
    color_acc_train = 'tab:green'
    color_acc_val = 'tab:orange'

    # Plot Loss (left Y-axis)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color_loss_train)
    ax1.plot(epochs, train_losses, label='Train Loss', color=color_loss_train, linewidth=2)
    ax1.plot(epochs, val_losses, label='Val Loss', color=color_loss_val, linewidth=2, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color_loss_train)
    ax1.legend(loc='upper left')

    # Plot Accuracy (right Y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy (%)', color=color_acc_train)
    ax2.plot(epochs, train_accuracies, label='Train Acc', color=color_acc_train, linewidth=2)
    ax2.plot(epochs, val_accuracies, label='Val Acc', color=color_acc_val, linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color_acc_train)
    ax2.legend(loc='lower right')

    # Final formatting
    plt.title('Loss and Accuracy vs. Epochs')
    fig.tight_layout()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.savefig(save_path, dpi=400)
    plt.show()
