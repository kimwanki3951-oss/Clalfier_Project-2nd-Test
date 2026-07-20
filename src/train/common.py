from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


def train_model(
    model,
    train_loader,
    valid_loader,
    train_dataset,
    valid_dataset,
    device,
    epochs: int,
    lr: float,
    save_name: str | Path | None = None,
):
    """
    사용자가 기존 Notebook에서 사용하던 train_model() 흐름을 유지한 공통 함수.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_dataset)

        model.eval()
        valid_loss = 0.0
        valid_correct = 0

        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()
                valid_correct += (
                    outputs.argmax(1) == labels
                ).sum().item()

        valid_acc = valid_correct / len(valid_dataset)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss / len(train_loader):.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Valid Loss: {valid_loss / len(valid_loader):.4f} "
            f"Valid Acc: {valid_acc:.4f}"
        )

    if save_name:
        save_path = Path(save_name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved: {save_path}")

    return model
