import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from skimage import io
from PIL import Image, ImageOps
from torchvision import transforms
from torch import optim
from torch.autograd import Variable

plt.rcParams['figure.figsize'] = (8, 8)

# fancy
# class Number16(Dataset):
    # def __init__(self, csv_file, dir, transform=None):
    #     self.annotations = pd.read_csv(csv_file, delimiter=";")
    #     self.root_dir = dir
    #     self.transform = transform

    # def __len__(self):
    #     return len(self.annotations)

    # def __getitem__(self, index):
    #     img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])

    #     original = cv2.imread(img_path)

    #     gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    #     kernel_3x3 = np.ones((6, 6), np.uint8) / 36
    #     blurred = cv2.filter2D(gray, -1, kernel_3x3)

    #     sobelx = cv2.Sobel(src=blurred, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)

    #     # Normalize the sobelx image to the range [0, 255]
    #     sobelx = cv2.normalize(sobelx, None, 0, 255, cv2.NORM_MINMAX)

    #     # Convert sobelx to 8-bit image
    #     sobelx = np.uint8(sobelx)

    #     image = Image.fromarray(sobelx)

    #     y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

    #     if self.transform:
    #         image = self.transform(image)

    #     return image, y_label

class Number16(Dataset):
    def __init__(self, csv_file, dir, transform=None):
        self.annotations = pd.read_csv(csv_file, delimiter=";")
        self.root_dir = dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])

        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert('L')

        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, y_label



dataset = Number16(
    csv_file="numbers555.csv",
    dir='./datasetFinal',
    transform=transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
)

print(dataset.annotations)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

loaders = {
    'train': torch.utils.data.DataLoader(train_dataset,
                                         batch_size=164,
                                         shuffle=True),

    'test': torch.utils.data.DataLoader(test_dataset,
                                        batch_size=164,
                                        shuffle=False)
}

print(f"testovaci {len(test_dataset)}")
print(f'trenovaci {len(train_dataset)}')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=4,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 64, 4, 1, 1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 72, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(72, 128, 4, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 1, 1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=4, padding=1, stride=1),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=4, padding=1, stride=1)

        )
        self.fc1 = nn.Sequential(
            nn.Linear(11 * 11 * 256, 472),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(472, 16),
            nn.BatchNorm1d(16),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        outputs = []
        x = self.conv1(x)
        outputs.append(x)
        x = self.conv2(x)
        outputs.append(x)
        x = self.conv3(x)
        outputs.append(x)
        x = self.conv4(x)
        # print("After conv4:", x.shape)
        outputs.append(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        outputs.append(x)
        output = self.out(x)
        outputs.append(output)
        return outputs


def plot_feature_maps(feature_maps, layer_name, n_cols=6):
    n_feature_maps = feature_maps.shape[1]
    n_rows = (n_feature_maps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    fig.suptitle(layer_name)
    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        if i < n_feature_maps:
            ax.imshow(feature_maps[0, i].detach().cpu().numpy(), cmap='gray')
        ax.axis('off')
    plt.show()


cnn = CNN()

loss_func = nn.CrossEntropyLoss()

optimizer = optim.Adam(cnn.parameters(), lr=0.0022)
print(optimizer)

print("---------------")
print("Trenovanie")
print("---------------")
num_epochs = 65
loss_iter = []


def train(num_epochs, cnn, loaders):
    cnn.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        total_step = len(loaders['train'])

        for i, (images, labels) in enumerate(loaders['train']):
            b_x = Variable(images)  # batch x
            b_y = Variable(labels)  # batch y
            output = cnn(b_x)
            loss = loss_func(output[-1], b_y)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch + 1}/{num_epochs}] finished with average loss: {running_loss / total_step:.4f}')


train(num_epochs, cnn, loaders)
print("Training complete.")

torch.save(cnn.state_dict(), 'cnn_model.pt')
print("Model saved as cnn_model.pt")


def test():
    cnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output = cnn(images)
            pred_y = torch.max(test_output[-1], 1)[1].data.squeeze()  # Use the final output for prediction
            correct += (pred_y == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / total
    print('Test Accuracy of the model on the {} test images: {:.2f}%'.format(total, accuracy * 100))


test()

# # Visualize feature maps
# image, label = dataset[2000]  # Take the first image from the dataset
# image = image.unsqueeze(0)  # Add batch dimension

# # Get the intermediate outputs
# outputs = cnn(image)

# # Plot the feature maps for each layer
# layer_names = ["conv1", "conv2", "conv3", "conv4", "fc1", "out"]
# for idx, output in enumerate(outputs):
#     if output.dim() == 4:  # Only plot for feature maps with 4 dimensions (batch_size, channels, height, width)
#         plot_feature_maps(output, layer_names[idx])
