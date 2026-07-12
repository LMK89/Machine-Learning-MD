## Giới thiệu về Mạng nơ-ron tích chập trong Python
Trang trình bày 1: Giới thiệu về Mạng thần kinh chuyển đổi

Mạng thần kinh chuyển đổi (CNN) là một lớp mô hình học sâu chủ yếu được sử dụng để xử lý dữ liệu dạng lưới, chẳng hạn như hình ảnh. Chúng được thiết kế để học một cách tự động và thích ứng các hệ thống phân cấp không gian của các tính năng từ dữ liệu đầu vào.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Creating a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.summary()
```

Slide 2: Các thành phần cốt lõi của CNN

Các thành phần chính của CNN là các lớp tích chập, các lớp gộp và các lớp được kết nối đầy đủ. Các lớp tích chập áp dụng các bộ lọc để phát hiện các tính năng, các lớp gộp làm giảm kích thước không gian và các lớp được kết nối đầy đủ thực hiện phân loại.

```python
# Convolutional Layer
conv_layer = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))

# Pooling Layer
pool_layer = layers.MaxPooling2D((2, 2))

# Fully Connected Layer
fc_layer = layers.Dense(64, activation='relu')

# Visualizing the output shape of each layer
input_shape = (28, 28, 1)
print(f"Input shape: {input_shape}")
print(f"Conv2D output shape: {conv_layer(tf.zeros(input_shape)).shape}")
print(f"MaxPooling2D output shape: {pool_layer(conv_layer(tf.zeros(input_shape))).shape}")
```

Trang trình bày 3: Lớp chập

Các lớp tích chập là các khối xây dựng cốt lõi của CNN. Họ sử dụng các bộ lọc để phát hiện các đặc điểm trong dữ liệu đầu vào, chẳng hạn như các cạnh, họa tiết và mẫu. Các bộ lọc trượt qua đầu vào, thực hiện phép nhân và tổng theo từng phần tử.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 5x5 image
image = np.array([
    [0, 0, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 0]
])

# Define a 3x3 filter for edge detection
filter = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# Perform convolution
output = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        output[i, j] = np.sum(image[i:i+3, j:j+3] * filter)

# Visualize the input, filter, and output
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Input Image')
ax2.imshow(filter, cmap='gray')
ax2.set_title('Filter')
ax3.imshow(output, cmap='gray')
ax3.set_title('Output')
plt.show()
```

Slide 4: Chức năng kích hoạt

Các chức năng kích hoạt đưa tính phi tuyến tính vào mạng, cho phép mạng tìm hiểu các mẫu phức tạp. Các hàm kích hoạt phổ biến bao gồm ReLU, sigmoid và tanh. ReLU được sử dụng rộng rãi do tính đơn giản và hiệu quả trong việc giảm thiểu vấn đề biến mất gradient.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

x = np.linspace(-5, 5, 100)

plt.figure(figsize=(12, 4))
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.plot(x, tanh(x), label='Tanh')
plt.legend()
plt.title('Activation Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
```

Trang trình bày 5: Các lớp gộp

Các lớp gộp làm giảm kích thước không gian của bản đồ đặc trưng, ​​giảm tải tính toán và giúp đạt được tính bất biến về không gian. Các hoạt động gộp chung bao gồm gộp tối đa và gộp trung bình.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a 4x4 input
input_data = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# Max pooling
def max_pool(input_data, pool_size):
    output_shape = input_data.shape[0] // pool_size
    output = np.zeros((output_shape, output_shape))
    for i in range(output_shape):
        for j in range(output_shape):
            output[i, j] = np.max(input_data[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size])
    return output

# Apply max pooling
max_pooled = max_pool(input_data, 2)

# Visualize input and output
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(input_data, cmap='viridis')
ax1.set_title('Input')
ax2.imshow(max_pooled, cmap='viridis')
ax2.set_title('Max Pooled Output')
plt.show()
```

Slide 6: Các lớp được kết nối đầy đủ

Các lớp được kết nối đầy đủ thường được sử dụng ở phần cuối của kiến ​​trúc CNN cho các nhiệm vụ phân loại. Họ lấy đầu ra đã được làm phẳng của các lớp tích chập và lớp gộp và tạo ra các dự đoán đầu ra cuối cùng.

```python
import numpy as np

# Simulating the output of convolutional and pooling layers
flattened_input = np.random.rand(1, 64)  # 64 features

# Weights and biases for a fully connected layer
weights = np.random.rand(64, 10)  # 10 output classes
biases = np.random.rand(10)

# Forward pass through the fully connected layer
output = np.dot(flattened_input, weights) + biases

# Apply softmax activation for classification
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

probabilities = softmax(output)

print("Output probabilities:")
print(probabilities)
print("\nPredicted class:", np.argmax(probabilities))
```

Slide 7: Đào tạo CNN

Việc đào tạo CNN liên quan đến việc truyền dữ liệu về phía trước, tính toán tổn hao, truyền ngược và cập nhật tham số. Quá trình này nhằm mục đích giảm thiểu sự khác biệt giữa đầu ra dự đoán và đầu ra thực tế.

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Create the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, validation_split=0.2)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

Trang trình bày 8: Kiến trúc CNN

Nhiều kiến ​​trúc CNN khác nhau đã được phát triển theo thời gian, mỗi kiến ​​trúc đều có những đặc điểm riêng. Một số kiến ​​trúc phổ biến bao gồm LeNet, AlexNet, VGGNet và ResNet.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_lenet():
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1)),
        layers.AveragePooling2D((2, 2)),
        layers.Conv2D(16, (5, 5), activation='relu'),
        layers.AveragePooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation='relu'),
        layers.Dense(84, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

lenet = create_lenet()
lenet.summary()
```

Slide 9: Chuyển giao học tập

Học chuyển giao cho phép chúng tôi tận dụng các mô hình được đào tạo trước trên các tập dữ liệu lớn và tinh chỉnh chúng cho các nhiệm vụ cụ thể. Cách tiếp cận này đặc biệt hữu ích khi làm việc với dữ liệu hạn chế.

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom layers for fine-tuning
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

Trang trình bày 10: Tăng cường dữ liệu

Tăng cường dữ liệu là một kỹ thuật được sử dụng để tăng kích thước của tập dữ liệu huấn luyện một cách giả tạo bằng cách áp dụng các phép biến đổi khác nhau cho các hình ảnh hiện có. Điều này giúp cải thiện việc khái quát hóa mô hình và giảm việc trang bị quá mức.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Create an instance of ImageDataGenerator with augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Load a sample image
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
image = train_images[0].reshape((1, 28, 28, 1)).astype('float32') / 255

# Generate augmented images
aug_iter = datagen.flow(image, batch_size=1)

# Display original and augmented images
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
axs[0].imshow(image[0, :, :, 0], cmap='gray')
axs[0].set_title('Original')
for i in range(4):
    aug_image = next(aug_iter)[0, :, :, 0]
    axs[i+1].imshow(aug_image, cmap='gray')
    axs[i+1].set_title(f'Augmented {i+1}')
plt.show()
```

Slide 11: Ví dụ thực tế: Phân loại hình ảnh

CNN được sử dụng rộng rãi trong các nhiệm vụ phân loại hình ảnh. Hãy chứng minh điều này bằng một ví dụ đơn giản về phân loại các chữ số viết tay bằng bộ dữ liệu MNIST.

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Create and train the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Make predictions
predictions = model.predict(test_images[:5])
print("\nPredictions:")
print(predictions)

# Display some test images and their predicted labels
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    axs[i].imshow(test_images[i, :, :, 0], cmap='gray')
    axs[i].set_title(f'Predicted: {predictions[i].argmax()}')
    axs[i].axis('off')
plt.show()
```

Trang trình bày 12: Ví dụ thực tế: Phát hiện đối tượng

CNN cũng được sử dụng trong các nhiệm vụ phát hiện đối tượng, trong đó mục tiêu là xác định và định vị các đối tượng trong một hình ảnh. Đây là một ví dụ đơn giản sử dụng mô hình được đào tạo trước để phát hiện đối tượng.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub

# Load a pre-trained object detection model
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Function to load and preprocess an image
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, scores, classes, threshold=0.5):
    for i in range(len(boxes)):
        if scores[i] > threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                              fill=False, edgecolor='red', linewidth=2))
            plt.text(xmin, ymin, f'Class {classes[i]}: {scores[i]:.2f}',
                     bbox=dict(facecolor='red', alpha=0.5))

# Load and preprocess a sample image
image_path = tf.keras.utils.get_file("example_image.jpg", "https://example.com/image.jpg")
input_tensor = load_image(image_path)

# Perform object detection
output = model(input_tensor)

# Process the output
boxes = output["detection_boxes"][0].numpy()
scores = output["detection_scores"][0].numpy()
classes = output["detection_classes"][0].numpy().astype(int)

# Visualize the results
plt.figure(figsize=(12, 8))
plt.imshow(input_tensor[0])
draw_boxes(input_tensor[0], boxes, scores, classes)
plt.axis('off')
plt.show()
```

Slide 13: Ứng dụng CNN trong chẩn đoán hình ảnh y tế

CNN đã tìm thấy những ứng dụng quan trọng trong hình ảnh y tế, đặc biệt là trong phân tích tia X, MRI và quét CT. Chúng có thể hỗ trợ phát hiện những bất thường, phân loại bệnh và phân chia các cơ quan hoặc khối u.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define a simple CNN for medical image classification
def create_medical_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Example usage
input_shape = (256, 256, 1)  # Grayscale medical images
num_classes = 2  # Binary classification (e.g., normal vs. abnormal)

model = create_medical_cnn(input_shape, num_classes)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
```

Slide 14: Ứng dụng CNN trong xử lý ngôn ngữ tự nhiên

Mặc dù CNN chủ yếu liên quan đến xử lý hình ảnh nhưng chúng cũng đã được áp dụng cho các tác vụ xử lý ngôn ngữ tự nhiên. Trong NLP, CNN có thể được sử dụng để phân loại văn bản, phân tích tình cảm và thậm chí cả dịch máy.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Define a simple CNN for text classification
def create_text_cnn(max_words, embedding_dim, max_length, num_classes):
    model = models.Sequential([
        layers.Embedding(max_words, embedding_dim, input_length=max_length),
        layers.Conv1D(128, 5, activation='relu'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Example usage
max_words = 10000  # Vocabulary size
embedding_dim = 100  # Embedding dimension
max_length = 100  # Maximum sequence length
num_classes = 3  # Number of classes for classification

model = create_text_cnn(max_words, embedding_dim, max_length, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người quan tâm đến việc tìm hiểu sâu hơn về Mạng thần kinh chuyển đổi, đây là một số tài nguyên có giá trị:

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Học sâu. Thiên nhiên, 521(7553), 436-444. ArXiv: [https://arxiv.org/abs/1807.07987](https://arxiv.org/abs/1807.07987)
2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Phân loại ImageNet với mạng lưới thần kinh tích chập sâu. Những tiến bộ trong hệ thống xử lý thông tin thần kinh, 25. ArXiv: [https://arxiv.org/abs/1404.5997](https://arxiv.org/abs/1404.5997)
3. Simonyan, K., & Zisserman, A. (2014). Mạng tích chập rất sâu để nhận dạng hình ảnh quy mô lớn. ArXiv: [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Học dư sâu để nhận dạng hình ảnh. Trong Kỷ yếu của hội nghị IEEE về thị giác máy tính và nhận dạng mẫu (trang 770-778). ArXiv: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

Những bài viết này cung cấp kiến ​​thức nền tảng và các khái niệm nâng cao về kiến ​​trúc và ứng dụng CNN.
