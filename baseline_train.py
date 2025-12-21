# baseline_train.py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from load_dataset import load_dataset
from dataset_split import split_dataset
from baseline_model import Baseline1DCNN   


# --------------------------------------------------------
# 参数
# --------------------------------------------------------
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------
# numpy → tensor
# --------------------------------------------------------
def to_tensor(x, y):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # [N, 1, 1024]
    y = torch.tensor(y, dtype=torch.long)
    return x, y


# --------------------------------------------------------
# 训练 1 epoch
# --------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(batch_y).sum().item()
        total += batch_y.size(0)

    return total_loss / total, correct / total


# --------------------------------------------------------
# 验证
# --------------------------------------------------------
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(batch_y).sum().item()
            total += batch_y.size(0)

    return total_loss / total, correct / total


# --------------------------------------------------------
# 主程序
# --------------------------------------------------------
def main():

    print("Loading raw CWRU dataset...")

    # 路径
    # D:/design/data/CWRU/
    data = load_dataset("data/CWRU")
    print("Loaded items:", len(data))
    print("Example item:", data[0])

    print("Dataset loaded. Now splitting...")

    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(
        input_dataset=data
    )

    # 转 tensor
    x_train, y_train = to_tensor(x_train, y_train)
    x_val, y_val = to_tensor(x_val, y_val)
    x_test, y_test = to_tensor(x_test, y_test)

    # Dataloader
    train_loader = DataLoader(TensorDataset(x_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val),
                            batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test),
                             batch_size=BATCH_SIZE, shuffle=False)

    # 模型
    model = Baseline1DCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Start training...")
    best_val_acc = 0.0

    # ----------------------------------------------------
    # Epoch 训练循环
    # ----------------------------------------------------
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        # 保存最好的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "baseline_best.pth")

    # ----------------------------------------------------
    # Test
    # ----------------------------------------------------
    print("Training finished.")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")

    print("Evaluating on test set...")
    model.load_state_dict(torch.load("baseline_best.pth"))
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print(f"Test Accuracy: {test_acc*100:.2f}%")



if __name__ == "__main__":
    main()
