import torch
import torch.nn as nn

#validation loop
def evaluate_model(model, train_loader, val_loader, criterion, device, eval_iter=50):
    """Evaluate model on train & val sets (few batches for speed)."""
    model.eval()
    losses = {"train": [], "val": []}

    # Evaluation without gradients
    with torch.no_grad():
        # Training set loss (few batches only)
        for i, (input_batch, target_batch) in enumerate(train_loader):
            if i >= eval_iter: break
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            logits = model(input_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), target_batch.view(-1))
            losses["train"].append(loss.item())

        # Validation set loss
        for i, (input_batch, target_batch) in enumerate(val_loader):
            if i >= eval_iter: break
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            logits = model(input_batch)
            loss = criterion(logits.view(-1, logits.size(-1)), target_batch.view(-1))
            losses["val"].append(loss.item())

    # Return average
    return sum(losses["train"]) / len(losses["train"]), sum(losses["val"]) / len(losses["val"])


# Training loop with early stopping
def train_model(model, train_dataloader, val_dataloader, optimizer,
                num_epochs, eval_freq, device, patience=3):
    loss_function = nn.CrossEntropyLoss()
    train_loss, val_loss = [], []
    global_step = -1
    model = model.to(device)

    # Early stopping vars
    best_val_loss = float("inf")
    counter = 0
    stop_training = False

    # Looping for every epoch
    for epoch in range(num_epochs):
        if stop_training:
            break
        model.train()
        # looping for every batch in our dataloader
        for batch_idx, (input_batch, target_batch) in enumerate(train_dataloader):
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            optimizer.zero_grad()
            # batch prediction
            logits = model(input_batch)
            # loss calculation
            loss = loss_function(logits.view(-1, logits.size(-1)), target_batch.view(-1))
            # backprop
            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % eval_freq == 0:
                train_loss_value, val_loss_value = evaluate_model(
                    model, train_dataloader, val_dataloader, loss_function, device
                )
                train_loss.append(train_loss_value)
                val_loss.append(val_loss_value)

                print(f"Epoch {epoch+1} | Step {global_step} | "
                      f"Train Loss: {train_loss_value:.4f} | Val Loss: {val_loss_value:.4f}")

                # --- Early stopping check ---
                if val_loss_value < best_val_loss:
                    best_val_loss = val_loss_value
                    counter = 0  # reset patience counter
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping triggered at step {global_step} "
                              f"(no improvement for {patience} evals).")
                        stop_training = True
                        break

    return train_loss, val_loss
