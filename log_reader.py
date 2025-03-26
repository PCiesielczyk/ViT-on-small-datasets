import torch
import matplotlib.pyplot as plt

checkpoint_path = "path_here"

checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=torch.device('cpu'))

train_loss_history = checkpoint['train_loss']
test_loss_history = checkpoint['test_loss']
epochs = range(1, len(train_loss_history) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss_history, label='Train Loss', marker='o')
plt.plot(epochs, test_loss_history, label='Test Loss', marker='o')
plt.title(f"Train vs Test Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("loss_plot.png")
plt.show()
