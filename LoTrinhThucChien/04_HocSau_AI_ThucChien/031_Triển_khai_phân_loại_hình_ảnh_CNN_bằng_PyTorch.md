## Triển khai phân loại hình ảnh CNN bằng PyTorch
Slide 1: Giới thiệu về CNN và PyTorch

Mạng thần kinh chuyển đổi (CNN) là một lớp mô hình học sâu mạnh mẽ, đặc biệt hiệu quả cho các nhiệm vụ phân loại hình ảnh. PyTorch, một framework deep learning phổ biến, cung cấp một cách trực quan để triển khai CNN. Trình chiếu này sẽ hướng dẫn bạn trong quá trình tạo CNN để phân loại hình ảnh bằng PyTorch và Python.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

Slide 2: Chuẩn bị bộ dữ liệu

Trước khi xây dựng CNN, chúng ta cần chuẩn bị tập dữ liệu của mình. Chúng tôi sẽ sử dụng tập dữ liệu CIFAR-10, chứa 60.000 hình ảnh màu 32x32 trong 10 lớp. PyTorch cung cấp các phương pháp thuận tiện để tải và xử lý trước tập dữ liệu này.

```python
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')
```

Trang trình bày 3: Xác định kiến ​​trúc CNN

Bây giờ, hãy xác định kiến ​​trúc CNN của chúng ta. Chúng ta sẽ tạo một CNN đơn giản với hai lớp chập theo sau là ba lớp được kết nối đầy đủ.

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().to(device)
```

Slide 4: Chức năng mất mát và tối ưu hóa

Để huấn luyện CNN, chúng ta cần xác định hàm mất mát và trình tối ưu hóa. Chúng tôi sẽ sử dụng trình tối ưu hóa Giảm Entropy chéo và Giảm độ dốc ngẫu nhiên (SGD).

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

Slide 5: Training the CNN

Now, let's train our CNN. We'll iterate over our dataset multiple times (epochs), and in each epoch, we'll perform forward and backward passes to update our model's parameters.

```python
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
```

Slide 6: Đánh giá mô hình

Sau khi đào tạo, chúng ta cần đánh giá hiệu suất của mô hình trên tập kiểm tra để xem nó khái quát hóa dữ liệu chưa nhìn thấy tốt như thế nào.

```python
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
```

Trang trình bày 7: Độ chính xác theo cấp độ

Hãy phân tích hiệu suất của mô hình cho từng lớp để xác định bất kỳ sai lệch hoặc điểm yếu nào.

```python
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print(f'Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]}%')
```

Trang trình bày 8: Trực quan hóa các bộ lọc tích chập

Việc hiểu những gì CNN của chúng tôi đã học được có thể là một thách thức. Một cách để hiểu rõ hơn là trực quan hóa các bộ lọc trong các lớp tích chập.

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_filters(model, layer_num, single_channel=True, collated=False):
    filters = model.conv1.weight.data.cpu().numpy()
    if single_channel:
        if collated:
            filters = filters.reshape(filters.shape[0]*filters.shape[1], filters.shape[2], filters.shape[3])
        else:
            filters = filters[:,0,:,:]
    n_filters = filters.shape[0]
    ix = 1
    for i in range(n_filters):
        f = filters[i]
        ax = plt.subplot(n_filters//8 + 1, 8, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f, cmap='gray')
        ix += 1
    plt.show()

plot_filters(net, 0)
```

Slide 9: Trực quan hóa bản đồ đặc điểm

Một cách khác để hiểu CNN của chúng tôi là bằng cách trực quan hóa các bản đồ đặc trưng, ​​cho thấy cách đầu vào được chuyển đổi khi nó đi qua mạng.

```python
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

activation = {}
net.conv1.register_forward_hook(get_activation('conv1'))
net.conv2.register_forward_hook(get_activation('conv2'))

dataiter = iter(testloader)
images, labels = next(dataiter)

output = net(images.to(device))

plt.imshow(images[0].permute(1, 2, 0))
plt.show()

plt.imshow(activation['conv1'][0, 0].cpu(), cmap='viridis')
plt.show()

plt.imshow(activation['conv2'][0, 0].cpu(), cmap='viridis')
plt.show()
```

Trang trình bày 10: Học chuyển tiếp

Học chuyển giao cho phép chúng tôi tận dụng các mô hình được đào tạo trước trên các tập dữ liệu lớn để cải thiện hiệu suất trên các tập dữ liệu tương tự, nhỏ hơn. Hãy sử dụng mô hình ResNet được đào tạo trước cho nhiệm vụ phân loại CIFAR-10 của chúng tôi.

```python
import torchvision.models as models

# Load pre-trained ResNet
resnet = models.resnet18(pretrained=True)

# Freeze all layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace the last fully connected layer
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)

# Move model to device
resnet = resnet.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.fc.parameters(), lr=0.001, momentum=0.9)

# Train the model
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
```

Trang trình bày 11: Tăng cường dữ liệu

Tăng cường dữ liệu là một kỹ thuật nhằm tăng tính đa dạng của tập huấn luyện của bạn bằng cách áp dụng các phép biến đổi ngẫu nhiên. Điều này có thể giúp cải thiện việc khái quát hóa mô hình và giảm việc trang bị quá mức.

```python
# Define augmented transformations
augmented_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset with augmented transformations
augmented_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                  download=True, transform=augmented_transform)
augmented_trainloader = torch.utils.data.DataLoader(augmented_trainset, batch_size=4,
                                                    shuffle=True, num_workers=2)

# Visualize augmented images
dataiter = iter(augmented_trainloader)
images, labels = next(dataiter)

plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i].permute(1, 2, 0))
    plt.title(classes[labels[i]])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

Trang trình chiếu 12: Ví dụ thực tế: Phân loại giống vật nuôi

Một ứng dụng thực tế của CNN là phân loại giống vật nuôi. Điều này có thể được sử dụng trong các nơi trú ẩn động vật để tự động xác định giống chó hoặc mèo từ ảnh, hỗ trợ quá trình nhận nuôi.

```python
# Assume we have a pre-trained model for pet breed classification
class PetBreedClassifier(nn.Module):
    def __init__(self, num_breeds):
        super(PetBreedClassifier, self).__init__()
        self.features = models.resnet50(pretrained=True)
        num_ftrs = self.features.fc.in_features
        self.features.fc = nn.Linear(num_ftrs, num_breeds)

    def forward(self, x):
        return self.features(x)

# Load the model
num_breeds = 120  # Example: 120 dog breeds
model = PetBreedClassifier(num_breeds).to(device)
model.load_state_dict(torch.load('pet_breed_classifier.pth'))
model.eval()

# Function to predict breed
def predict_breed(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Example usage
image_path = 'golden_retriever.jpg'
breed_index = predict_breed(image_path, model, device)
print(f"Predicted breed: {breeds[breed_index]}")
```

Trang trình chiếu 13: Ví dụ thực tế: Phát hiện bệnh cây

Một ứng dụng thực tế khác của CNN là trong nông nghiệp để phát hiện bệnh cây trồng. Điều này có thể giúp nông dân nhanh chóng xác định và giải quyết các bệnh cây trồng, có khả năng cứu được toàn bộ vụ thu hoạch.

```python
# Assume we have a pre-trained model for plant disease detection
class PlantDiseaseDetector(nn.Module):
    def __init__(self, num_diseases):
        super(PlantDiseaseDetector, self).__init__()
        self.features = models.densenet121(pretrained=True)
        num_ftrs = self.features.classifier.in_features
        self.features.classifier = nn.Linear(num_ftrs, num_diseases)

    def forward(self, x):
        return self.features(x)

# Load the model
num_diseases = 38  # Example: 38 different plant diseases
model = PlantDiseaseDetector(num_diseases).to(device)
model.load_state_dict(torch.load('plant_disease_detector.pth'))
model.eval()

# Function to detect disease
def detect_disease(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Example usage
image_path = 'tomato_leaf.jpg'
disease_index = detect_disease(image_path, model, device)
print(f"Detected disease: {diseases[disease_index]}")
```

Trang trình bày 14: Tài nguyên bổ sung

Để khám phá thêm về CNN và PyTorch, hãy xem xét các tài nguyên sau:

1. "Phân loại ImageNet với Mạng lưới thần kinh chuyển đổi sâu" của Krizhevsky và cộng sự. (2012): Một bài báo chuyên đề về phân loại hình ảnh của CNN. (arXiv:1207.0580)
2. "Mạng chuyển đổi rất sâu để nhận dạng hình ảnh quy mô lớn" của Simonyan và Zisserman (2014): Giới thiệu kiến trúc mạng VGG. (arXiv:1409.1556)
3. "Học tập sâu để nhận dạng hình ảnh" của He et al. (2015): Trình bày kiến ​​trúc ResNet, cho phép đào tạo các mạng rất sâu. (arXiv:1512.03385)
4. Tài liệu về PyTorch ([https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)): Hướng dẫn toàn diện về các tính năng và API của PyTorch.
5. "Trực quan hóa và hiểu biết về mạng tích chập" của Zeiler và Fergus (2013): Cung cấp các kỹ thuật trực quan hóa các tính năng CNN. (arXiv:1311.2901)
