import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import pandas as pd
import os

from model import PrunableNet
from utils import compute_sparsity, compute_l1_loss, collect_all_gates

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)


def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return 100 * correct / total


def train_model(lambda_val):
    model = PrunableNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 5  # keep small for demo

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader)

        for x, y in loop:
            x, y = x.to(device), y.to(device)

            outputs = model(x)

            classification_loss = criterion(outputs, y)
            sparsity_loss = compute_l1_loss(model)

            loss = classification_loss + lambda_val * sparsity_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

    acc = evaluate(model)
    sparsity = compute_sparsity(model)

    return model, acc, sparsity


if __name__ == "__main__":
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    lambdas = [0.0001, 0.001, 0.01]
    results = []

    best_model = None
    best_acc = 0

    for lam in lambdas:
        print(f"\nTraining with lambda = {lam}")
        model, acc, sparsity = train_model(lam)

        results.append([lam, acc, sparsity])

        if acc > best_acc:
            best_acc = acc
            best_model = model

    df = pd.DataFrame(results, columns=["Lambda", "Accuracy", "Sparsity"])
    df.to_csv("experiments/results.csv", index=False)

    print("\nResults:")
    print(df)

    import matplotlib.pyplot as plt

    gates = collect_all_gates(best_model).numpy()
    plt.hist(gates, bins=50)
    plt.title("Gate Distribution")
    plt.savefig("outputs/gate_distribution.png")

    print("\nSaved plot in outputs/")
