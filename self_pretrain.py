import wandb
from bend.models.dilated_cnn import ConvNetConfig, ConvNetForMaskedLM
import torch
from torch.optim import AdamW
from bend.utils.masked_data_downstream import get_data
from tqdm import tqdm

# Initialize wandb
wandb.init(project="gene_mlm", name="convnet_masked_lm")

# Config
config = ConvNetConfig(
    vocab_size=7,  # e.g. A,C,G,T,N,mask,pad
    hidden_size=512,
    n_layers=30,
    kernel_size=9,
    dilation_double_every=1,
    dilation_max=32,
    dilation_cycle=6,
    initializer_range=0.02,
)

# Model, data, optimizer
mlm_model = ConvNetForMaskedLM(config)
train_loader, val_loader, test_loader = get_data("./data/gene_finding/onehot")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlm_model.to(device)
optimizer = AdamW(mlm_model.parameters(), lr=1e-4)

# Training loop
best_val_loss = float("inf")
patience_counter = 0
max_patience = 5
num_epochs = 50

def compute_accuracy(preds, labels):
    pred_ids = preds.argmax(dim=-1)
    mask = labels != -100  # ignore padding/masked positions
    correct = (pred_ids == labels) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    return accuracy

for epoch in tqdm(range(num_epochs), desc="Epochs"):
    mlm_model.train()
    train_loss, train_acc, train_steps = 0, 0, 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = mlm_model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        acc = compute_accuracy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += acc
        train_steps += 1

    avg_train_loss = train_loss / train_steps
    avg_train_acc = train_acc / train_steps

    # Validation
    mlm_model.eval()
    val_loss, val_acc, val_steps = 0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = mlm_model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            acc = compute_accuracy(logits, labels)

            val_loss += loss.item()
            val_acc += acc
            val_steps += 1

    avg_val_loss = val_loss / val_steps
    avg_val_acc = val_acc / val_steps

    # Logging to wandb
    wandb.log({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "train_accuracy": avg_train_acc,
        "val_loss": avg_val_loss,
        "val_accuracy": avg_val_acc,
    })

    print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_acc:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_acc:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch} due to no improvement in validation loss.")
            break

# Save final model
# torch.save(mlm_model.state_dict(), "convnet_mlm_final.pt")
mlm_model.save_pretrained("./checkpoints/final")
print("Final model saved.")

# Final test evaluation
mlm_model.eval()
test_loss, test_acc, test_steps = 0, 0, 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = mlm_model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        acc = compute_accuracy(logits, labels)

        test_loss += loss.item()
        test_acc += acc
        test_steps += 1

avg_test_loss = test_loss / test_steps
avg_test_acc = test_acc / test_steps

wandb.log({
    "test_loss": avg_test_loss,
    "test_accuracy": avg_test_acc,
})

print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.4f}")
