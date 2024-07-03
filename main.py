import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import seaborn as sns
from sklearn.model_selection import KFold
import torchvision.models as models
from torchvision.datasets import ImageFolder
import os
import torch.nn.functional as F
from PIL import Image

# Transformări pentru datele de antrenare (augmentări + normalizare)
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Flipping orizontal aleatoriu
    transforms.RandomCrop(224, padding=4),  # Decupare aleatorie cu padding
    transforms.ToTensor(),  # Conversie la tensor
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalizare
])

# Transformări pentru datele de validare și testare (doar redimensionare și normalizare)
transform_test = transforms.Compose([
    transforms.Resize(224),  # Redimensionare la 224x224
    transforms.ToTensor(),  # Conversie la tensor
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalizare
])

# Funcție pentru încărcarea datelor dintr-un director, cu transformările specificate
def load_data(root_dir, transform):
    dataset = ImageFolder(root=root_dir, transform=transform)
    return dataset

# Funcție pentru pregătirea datelor (antrenare, validare, testare) folosind DataLoader și transformări
def prepare_data(root_dir='./data', batch_size=64, num_workers=4):
    # Încărcarea dataset-urilor folosind transformările definite
    train_dataset = load_data(root_dir + '/train', transform_train)
    valid_dataset = load_data(root_dir + '/valid', transform_test)
    test_dataset = load_data(root_dir + '/test', transform_test)

    # Crearea DataLoader-elor pentru fiecare dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

# Funcție pentru vizualizarea datelor și distribuției claselor
def visualize_data(loader, classes):
    dataiter = iter(loader)  # Crearea unui iterator pentru loader
    images, labels = next(dataiter)  # Obținerea primului batch de imagini și etichete
    img = torchvision.utils.make_grid(images)  # Crearea unui grid de imagini
    img = img / 2 + 0.5  # Denormalizare
    img = torch.clamp(img, 0, 1)  # Limitare la intervalul [0, 1]
    npimg = img.numpy()  # Conversie la numpy array
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Afișare imagine
    plt.show()

    # Calcularea și afișarea distribuției claselor
    class_counts = [0] * len(classes)
    for _, label in loader.dataset:
        class_counts[label] += 1

    plt.bar(classes, class_counts)  # Crearea unui grafic de bare pentru distribuția claselor
    plt.title('Class Distribution')
    plt.show()

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Primul strat convoluțional
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Al doilea strat convoluțional
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Al treilea strat convoluțional
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Stratul de max pooling
        self.fc1 = nn.Linear(128 * 28 * 28, 256)  # Primul strat complet conectat
        self.fc2 = nn.Linear(256, num_classes)  # Al doilea strat complet conectat

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Aplicarea primei convoluții, urmată de ReLU și max pooling
        x = self.pool(F.relu(self.conv2(x)))  # Aplicarea celei de-a doua convoluții, urmată de ReLU și max pooling
        x = self.pool(F.relu(self.conv3(x)))  # Aplicarea celei de-a treia convoluții, urmată de ReLU și max pooling
        x = x.view(-1, 128 * 28 * 28)  # Flattening
        x = F.relu(self.fc1(x))  # Aplicarea primului strat complet conectat cu ReLU
        x = self.fc2(x)  # Aplicarea celui de-al doilea strat complet conectat (output)
        return x

# Funcție pentru antrenarea modelului
def train_model(model, device, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs, writer=None):
    train_losses = []  # Listă pentru stocarea pierderilor din antrenare
    valid_losses = []  # Listă pentru stocarea pierderilor din validare

    for epoch in range(num_epochs):
        model.train()  # Mod de antrenare
        running_train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Mutarea datelor pe device
            optimizer.zero_grad()  # Resetarea gradientului
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calcularea pierderii
            loss.backward()  # Backward pass
            optimizer.step()  # Actualizarea parametrilor
            running_train_loss += loss.item()  # Adăugarea pierderii curente

        train_loss = running_train_loss / len(train_loader)  # Pierderea medie pe epocă
        train_losses.append(train_loss)  # Stocarea pierderii

        model.eval()  # Mod de evaluare
        running_valid_loss = 0.0

        with torch.no_grad():  # Fără calcularea gradientului
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # Mutarea datelor pe device
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Calcularea pierderii
                running_valid_loss += loss.item()  # Adăugarea pierderii curente

        valid_loss = running_valid_loss / len(valid_loader)  # Pierderea medie pe epocă
        valid_losses.append(valid_loss)  # Stocarea pierderii

        # Scrierea pierderilor în TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/valid', valid_loss, epoch)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')

        if scheduler:
            scheduler.step()  # Actualizarea learning rate-ului

    return train_losses, valid_losses


# Funcție pentru testarea modelului și calcularea metricilor
def test_model(model, device, test_loader, classes):
    y_true = []  # Lista pentru etichetele reale
    y_pred = []  # Lista pentru predicții

    model.eval()  # Mod de evaluare
    with torch.no_grad():  # Fără calcularea gradientului
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Mutarea datelor pe device
            outputs = model(inputs)  # Forward pass
            _, preds = torch.max(outputs, 1)  # Obținerea predicțiilor
            y_true.extend(labels.cpu().numpy())  # Adevăratele etichete
            y_pred.extend(preds.cpu().numpy())  # Predicțiile

    # Calcularea metricilor
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Recall: {recall:.4f}')

    # Afișarea matricei de confuzie
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()


# Funcție pentru clasificarea unei imagini individuale
def predict_image(image_path, model, device, classes, transform):
    # Încărcarea imaginii și aplicarea transformărilor
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Adăugarea dimensiunii batch-ului și mutarea pe device

    # Obținerea predicției
    model.eval()  # Mod de evaluare
    with torch.no_grad():  # Fără calcularea gradientului
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Returnarea numelui clasei prezise
    return classes[predicted.item()]

def main():
    # Pregătirea datelor
    train_loader, valid_loader, test_loader = prepare_data()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Vizualizare date
    classes = train_loader.dataset.classes
    visualize_data(train_loader, classes)

    # Crearea modelului
    model = SimpleCNN(len(classes))
    model.to(device)  # Mutarea modelului pe device-ul specificat

    # Parametrii de antrenare
    criterion = nn.CrossEntropyLoss()  # Funcția de pierdere
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizatorul
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Scheduler pentru learning rate

    # Verificare existență fișier de checkpoint
    if os.path.exists('checkpoint.pth'):
        checkpoint = torch.load('checkpoint.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        valid_losses = checkpoint['valid_losses']
        print(f"Checkpoint loaded. Resuming training from epoch {epoch + 1}.")
    else:
        epoch = 0
        train_losses = []
        valid_losses = []
        print("No checkpoint found. Starting training from scratch.")

    # Antrenarea modelului
    writer = SummaryWriter()  # Inițializarea writer-ului pentru TensorBoard
    train_losses, valid_losses = train_model(model, device, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=0, writer=writer)

    # Testarea modelului
    test_model(model, device, test_loader, classes)

    # Salvarea checkpoint-ului
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'valid_losses': valid_losses
    }
    torch.save(checkpoint, 'checkpoint.pth')

    # Închiderea writer-ului pentru TensorBoard
    writer.close()

    # Generarea graficului pentru pierderi
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Clasificarea unei imagini individuale
    test_image_path = './data/test/ace of clubs/1.jpg'
    predicted_class = predict_image(test_image_path, model, device, classes, transform_test)
    print(f'The predicted class for the image is: {predicted_class}')

    test_image_path = './data/test/eight of clubs/5.jpg'
    predicted_class = predict_image(test_image_path, model, device, classes, transform_test)
    print(f'The predicted class for the image is: {predicted_class}')

    test_image_path = './data/test/four of clubs/3.jpg'
    predicted_class = predict_image(test_image_path, model, device, classes, transform_test)
    print(f'The predicted class for the image is: {predicted_class}')

    test_image_path = './data/test/joker/3.jpg'
    predicted_class = predict_image(test_image_path, model, device, classes, transform_test)
    print(f'The predicted class for the image is: {predicted_class}')

    test_image_path = './data/test/queen of diamonds/3.jpg'
    predicted_class = predict_image(test_image_path, model, device, classes, transform_test)
    print(f'The predicted class for the image is: {predicted_class}')

    test_image_path = './data/test/six of hearts/3.jpg'
    predicted_class = predict_image(test_image_path, model, device, classes, transform_test)
    print(f'The predicted class for the image is: {predicted_class}')

    test_image_path = './data/test/two of spades/1.jpg'
    predicted_class = predict_image(test_image_path, model, device, classes, transform_test)
    print(f'The predicted class for the image is: {predicted_class}')


if __name__ == "__main__":
    main()
