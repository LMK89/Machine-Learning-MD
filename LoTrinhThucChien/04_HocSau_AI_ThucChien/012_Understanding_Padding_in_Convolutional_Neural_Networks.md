## Tìm hiểu về phần đệm trong Mạng thần kinh chuyển đổi
Trang trình bày 1: Tìm hiểu phần đệm trong Mạng thần kinh chuyển đổi (CNN)

Đệm là một khái niệm quan trọng trong CNN liên quan đến việc thêm các pixel bổ sung xung quanh hình ảnh đầu vào trước khi áp dụng các phép tích chập. Kỹ thuật này giúp bảo toàn kích thước không gian và trích xuất các đặc điểm từ các cạnh của hình ảnh. Hãy cùng khám phá phần đệm bằng một ví dụ đơn giản:

```python
import numpy as np
import matplotlib.pyplot as plt

def add_padding(image, pad_width):
    return np.pad(image, pad_width, mode='constant', constant_values=0)

# Create a sample 5x5 image
image = np.random.randint(0, 255, size=(5, 5))

# Add padding of 1 pixel
padded_image = add_padding(image, pad_width=1)

# Visualize the original and padded images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(padded_image, cmap='gray')
ax2.set_title('Padded Image')
plt.show()
```

Slide 2: Các loại Padding

Có hai loại phần đệm chính: phần đệm hợp lệ (không có phần đệm) và phần đệm giống nhau (phần đệm để duy trì kích thước đầu vào). Hãy thực hiện cả hai loại:

```python
import numpy as np

def convolution2d(image, kernel, padding='valid'):
    if padding == 'same':
        pad_height = (kernel.shape[0] - 1) // 2
        pad_width = (kernel.shape[1] - 1) // 2
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    else:
        padded_image = image

    output_height = padded_image.shape[0] - kernel.shape[0] + 1
    output_width = padded_image.shape[1] - kernel.shape[1] + 1
    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)

    return output

# Example usage
image = np.random.randint(0, 255, size=(6, 6))
kernel = np.random.rand(3, 3)

valid_output = convolution2d(image, kernel, padding='valid')
same_output = convolution2d(image, kernel, padding='same')

print("Valid padding output shape:", valid_output.shape)
print("Same padding output shape:", same_output.shape)
```

Slide 3: Tầm quan trọng của phần đệm trong CNN

Phần đệm đóng một vai trò quan trọng trong CNN bằng cách giải quyết các vấn đề sau:

1. Bảo toàn kích thước không gian: Không có phần đệm, mỗi lớp tích chập sẽ giảm kích thước không gian của đầu vào, có khả năng dẫn đến mất thông tin quan trọng.
2. Lưu giữ thông tin ở cạnh: Phần đệm giúp bảo tồn thông tin từ các cạnh của đầu vào, nếu không thì thông tin này sẽ không được thể hiện ở đầu ra.
3. Mạng sâu hơn: Phần đệm cho phép tạo mạng sâu hơn bằng cách duy trì các kích thước không gian thông qua nhiều lớp tích chập.

Hãy hình dung tác dụng của phần đệm trên các kích thước không gian:

```python
import numpy as np
import matplotlib.pyplot as plt

def conv_output_shape(input_shape, kernel_size, padding='valid', stride=1):
    if padding == 'same':
        return input_shape
    else:
        return (input_shape - kernel_size) // stride + 1

input_sizes = range(10, 101, 10)
kernel_size = 3
valid_outputs = [conv_output_shape(size, kernel_size, 'valid') for size in input_sizes]
same_outputs = [conv_output_shape(size, kernel_size, 'same') for size in input_sizes]

plt.figure(figsize=(10, 6))
plt.plot(input_sizes, valid_outputs, label='Valid Padding')
plt.plot(input_sizes, same_outputs, label='Same Padding')
plt.xlabel('Input Size')
plt.ylabel('Output Size')
plt.title('Effect of Padding on Output Size')
plt.legend()
plt.grid(True)
plt.show()
```

Trang trình bày 4: Triển khai phần đệm trong TensorFlow/Keras

TensorFlow và Keras cung cấp các API dễ sử dụng để triển khai phần đệm trong CNN. Hãy tạo một mô hình CNN đơn giản với các tùy chọn đệm khác nhau:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape, padding_type):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding=padding_type, input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding=padding_type),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding=padding_type),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Create models with different padding types
valid_model = create_cnn_model((28, 28, 1), 'valid')
same_model = create_cnn_model((28, 28, 1), 'same')

print("Valid padding model summary:")
valid_model.summary()

print("\nSame padding model summary:")
same_model.summary()
```

Trang trình bày 5: Phần đệm tùy chỉnh trong PyTorch

PyTorch cho phép các tùy chọn đệm linh hoạt hơn, bao gồm cả phần đệm tùy chỉnh. Hãy triển khai chức năng đệm tùy chỉnh và sử dụng nó trong mô hình PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomPadding(nn.Module):
    def __init__(self, padding):
        super(CustomPadding, self).__init__()
        self.padding = padding

    def forward(self, x):
        return F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)

class CustomCNN(nn.Module):
    def __init__(self, padding_size):
        super(CustomCNN, self).__init__()
        self.padding = CustomPadding(padding_size)
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 6, 10)

    def forward(self, x):
        x = self.padding(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.padding(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 6 * 6)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

# Create a model with custom padding
model = CustomCNN(padding_size=1)
print(model)

# Test the model with a sample input
sample_input = torch.randn(1, 1, 28, 28)
output = model(sample_input)
print("Output shape:", output.shape)
```

Trang trình bày 6: Bản đồ đệm và tính năng

Phần đệm ảnh hưởng đến kích thước của bản đồ đặc trưng trong CNN. Hãy hình dung các loại phần đệm khác nhau tác động như thế nào đến kích thước bản đồ đặc điểm:

```python
import numpy as np
import matplotlib.pyplot as plt

def calculate_feature_map_size(input_size, kernel_size, padding, stride):
    if padding == 'same':
        return np.ceil(input_size / stride)
    elif padding == 'valid':
        return np.floor((input_size - kernel_size + 1) / stride)

input_sizes = range(20, 101, 10)
kernel_size = 3
stride = 1

valid_sizes = [calculate_feature_map_size(size, kernel_size, 'valid', stride) for size in input_sizes]
same_sizes = [calculate_feature_map_size(size, kernel_size, 'same', stride) for size in input_sizes]

plt.figure(figsize=(10, 6))
plt.plot(input_sizes, valid_sizes, label='Valid Padding', marker='o')
plt.plot(input_sizes, same_sizes, label='Same Padding', marker='s')
plt.xlabel('Input Size')
plt.ylabel('Feature Map Size')
plt.title('Effect of Padding on Feature Map Size')
plt.legend()
plt.grid(True)
plt.show()
```

Trang trình bày 7: Phần đệm và trường tiếp nhận

Trường tiếp nhận là vùng đầu vào ảnh hưởng đến một tính năng CNN cụ thể. Lớp đệm ảnh hưởng đến sự phát triển của trường tiếp nhận. Hãy hình dung mối quan hệ này:

```python
import numpy as np
import matplotlib.pyplot as plt

def calculate_receptive_field(num_layers, kernel_size, padding):
    if padding == 'same':
        return kernel_size + (kernel_size - 1) * (num_layers - 1)
    elif padding == 'valid':
        return num_layers * (kernel_size - 1) + 1

num_layers = range(1, 11)
kernel_size = 3

valid_rf = [calculate_receptive_field(n, kernel_size, 'valid') for n in num_layers]
same_rf = [calculate_receptive_field(n, kernel_size, 'same') for n in num_layers]

plt.figure(figsize=(10, 6))
plt.plot(num_layers, valid_rf, label='Valid Padding', marker='o')
plt.plot(num_layers, same_rf, label='Same Padding', marker='s')
plt.xlabel('Number of Layers')
plt.ylabel('Receptive Field Size')
plt.title('Effect of Padding on Receptive Field Growth')
plt.legend()
plt.grid(True)
plt.show()
```

Slide 8: Ví dụ thực tế: Phân loại hình ảnh

Hãy triển khai một tác vụ phân loại hình ảnh đơn giản bằng cách sử dụng tập dữ liệu MNIST để chứng minh tác động của phần đệm đến hiệu suất mô hình:

```python
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

def create_model(padding_type):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding=padding_type, input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding=padding_type),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding=padding_type),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Create and train models with different padding types
valid_model = create_model('valid')
same_model = create_model('same')

valid_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
same_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

valid_history = valid_model.fit(train_images, train_labels, epochs=5, validation_split=0.2, verbose=0)
same_history = same_model.fit(train_images, train_labels, epochs=5, validation_split=0.2, verbose=0)

# Evaluate models
valid_test_loss, valid_test_acc = valid_model.evaluate(test_images, test_labels, verbose=0)
same_test_loss, same_test_acc = same_model.evaluate(test_images, test_labels, verbose=0)

print(f"Valid padding - Test accuracy: {valid_test_acc:.4f}")
print(f"Same padding - Test accuracy: {same_test_acc:.4f}")
```

Trang trình bày 9: Phần đệm và độ phức tạp của mô hình

Phần đệm ảnh hưởng đến số lượng tham số trong mô hình CNN. Hãy so sánh độ phức tạp của mô hình đối với các loại phần đệm khác nhau:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_model(input_shape, padding_type):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding=padding_type, input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding=padding_type),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding=padding_type),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

input_shapes = [(28, 28, 1), (32, 32, 1), (64, 64, 1)]
padding_types = ['valid', 'same']

for shape in input_shapes:
    print(f"\nInput shape: {shape}")
    for padding in padding_types:
        model = create_model(shape, padding)
        print(f"{padding.capitalize()} padding - Total parameters: {model.count_params():,}")
```

Trang trình bày 10: Đệm và hiệu quả tính toán

Phần đệm có thể ảnh hưởng đến hiệu quả tính toán của CNN. Hãy đo thời gian suy luận cho các mô hình với các loại phần đệm khác nhau:

```python
import tensorflow as tf
import time
import numpy as np
from tensorflow.keras import layers, models

def create_model(input_shape, padding_type):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding=padding_type, input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding=padding_type),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding=padding_type),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

input_shape = (224, 224, 3)
batch_size = 32
num_iterations = 100

valid_model = create_model(input_shape, 'valid')
same_model = create_model(input_shape, 'same')

# Warm-up
for _ in range(10):
    _ = valid_model.predict(np.random.rand(1, *input_shape))
    _ = same_model.predict(np.random.rand(1, *input_shape))

# Measure inference time
valid_time = 0
same_time = 0

for _ in range(num_iterations):
    input_data = np.random.rand(batch_size, *input_shape)

    start = time.time()
    _ = valid_model.predict(input_data)
    valid_time += time.time() - start

    start = time.time()
    _ = same_model.predict(input_data)
    same_time += time.time() - start

print(f"Average inference time (Valid padding): {valid_time/num_iterations*1000:.2f} ms")
print(f"Average inference time (Same padding): {same_time/num_iterations*1000:.2f} ms")
```

Trang trình bày 11: Khoảng đệm và Dòng chuyển màu

Phần đệm ảnh hưởng đến luồng gradient trong CNN. Hãy trực quan hóa độ lớn độ dốc cho các lớp khác nhau với các loại phần đệm khác nhau:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def create_model(padding):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, padding=padding, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, 3, padding=padding, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

def get_gradients(model, inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = tf.keras.losses.sparse_categorical_crossentropy(targets, predictions)
    return tape.gradient(loss, model.trainable_variables)

# Create sample data
inputs = np.random.rand(1, 28, 28, 1).astype(np.float32)
targets = np.array([5])

# Create models with different padding
valid_model = create_model('valid')
same_model = create_model('same')

# Compute gradients
valid_grads = get_gradients(valid_model, inputs, targets)
same_grads = get_gradients(same_model, inputs, targets)

# Plot gradient magnitudes
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Valid Padding Gradients')
plt.bar(range(len(valid_grads)), [np.mean(np.abs(g)) for g in valid_grads])
plt.subplot(1, 2, 2)
plt.title('Same Padding Gradients')
plt.bar(range(len(same_grads)), [np.mean(np.abs(g)) for g in same_grads])
plt.tight_layout()
plt.show()
```

Slide 12: Ví dụ thực tế: Phân đoạn hình ảnh

Phân đoạn hình ảnh là một nhiệm vụ trong đó phần đệm đóng vai trò quan trọng. Hãy triển khai kiến ​​trúc U-Net đơn giản để phân đoạn hình ảnh:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(inputs, filters, padding='same'):
    x = layers.Conv2D(filters, 3, padding=padding, activation='relu')(inputs)
    x = layers.Conv2D(filters, 3, padding=padding, activation='relu')(x)
    return x

def unet(input_size=(256, 256, 1), padding='same'):
    inputs = layers.Input(input_size)

    # Encoder
    conv1 = conv_block(inputs, 64, padding)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, 128, padding)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bridge
    conv3 = conv_block(pool2, 256, padding)

    # Decoder
    up4 = layers.UpSampling2D(size=(2, 2))(conv3)
    up4 = layers.concatenate([up4, conv2])
    conv4 = conv_block(up4, 128, padding)
    up5 = layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = layers.concatenate([up5, conv1])
    conv5 = conv_block(up5, 64, padding)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv5)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Create U-Net models with different padding
valid_unet = unet(padding='valid')
same_unet = unet(padding='same')

print("U-Net with valid padding:")
valid_unet.summary()

print("\nU-Net with same padding:")
same_unet.summary()
```

Trang trình bày 13: Đệm và tăng cường dữ liệu

Phần đệm có thể được sử dụng trong các kỹ thuật tăng cường dữ liệu để cải thiện khả năng khái quát hóa mô hình. Hãy triển khai phần mở rộng dựa trên phần đệm tùy chỉnh:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def random_padding_augmentation(image, max_pad=10):
    pad_top = np.random.randint(0, max_pad)
    pad_bottom = np.random.randint(0, max_pad)
    pad_left = np.random.randint(0, max_pad)
    pad_right = np.random.randint(0, max_pad)

    padded_image = tf.pad(image, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='CONSTANT', constant_values=0)

    crop_height = tf.shape(image)[0]
    crop_width = tf.shape(image)[1]

    return tf.image.random_crop(padded_image, [crop_height, crop_width, 3])

# Load a sample image
sample_image = plt.imread('sample_image.jpg')

# Apply augmentation
augmented_images = [random_padding_augmentation(sample_image) for _ in range(4)]

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(augmented_images[i].numpy().astype(np.uint8))
    ax.axis('off')
    ax.set_title(f'Augmented Image {i+1}')
plt.tight_layout()
plt.show()
```

Trang trình bày 14: Phần đệm trong các kết cấu 3D

Phần đệm cũng rất quan trọng trong các cấu trúc 3D, thường được sử dụng trong phân tích video và hình ảnh y tế. Hãy triển khai một CNN 3D đơn giản với phần đệm:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_3d_cnn(input_shape, padding='same'):
    model = models.Sequential([
        layers.Conv3D(32, (3, 3, 3), activation='relu', padding=padding, input_shape=input_shape),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Conv3D(64, (3, 3, 3), activation='relu', padding=padding),
        layers.MaxPooling3D((2, 2, 2)),
        layers.Conv3D(64, (3, 3, 3), activation='relu', padding=padding),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Create 3D CNN models with different padding
input_shape = (64, 64, 64, 1)  # Example input shape for 3D data
valid_3d_cnn = create_3d_cnn(input_shape, padding='valid')
same_3d_cnn = create_3d_cnn(input_shape, padding='same')

print("3D CNN with valid padding:")
valid_3d_cnn.summary()

print("\n3D CNN with same padding:")
same_3d_cnn.summary()
```

Trang trình bày 15: Tài nguyên bổ sung

Để biết thêm thông tin chuyên sâu về phần đệm trong Mạng thần kinh chuyển đổi, hãy xem xét khám phá các tài nguyên sau:

1. "Hướng dẫn về số học tích chập để học sâu" của Vincent Dumoulin và Francesco Visin (arXiv:1603.07285) URL: [https://arxiv.org/abs/1603.07285](https://arxiv.org/abs/1603.07285)
2. "Đi sâu vào bộ chỉnh lưu: Vượt qua hiệu suất cấp độ con người trên phân loại ImageNet" của Kaiming He và cộng sự. (arXiv:1502.01852) URL: [https://arxiv.org/abs/1502.01852](https://arxiv.org/abs/1502.01852)
3. "Mạng trong mạng" của Min Lin và cộng sự. (arXiv:1312.4400) URL: [https://arxiv.org/abs/1312.4400](https://arxiv.org/abs/1312.4400)

Những bài viết này cung cấp những hiểu biết sâu sắc có giá trị về lý thuyết và thực tiễn của mạng lưới thần kinh tích chập, bao gồm cả vai trò của phần đệm trong các kiến ​​trúc và nhiệm vụ khác nhau.
