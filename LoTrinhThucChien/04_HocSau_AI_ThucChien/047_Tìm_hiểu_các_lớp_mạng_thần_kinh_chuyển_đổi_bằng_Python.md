## Tìm hiểu các lớp mạng thần kinh tích chập bằng Python
Trang trình bày 1: Tìm hiểu các lớp của Mạng thần kinh chuyển đổi (CNN)

Mạng thần kinh chuyển đổi (CNN) là một lớp mô hình học sâu chủ yếu được sử dụng cho các tác vụ xử lý hình ảnh. Chúng bao gồm nhiều lớp hoạt động cùng nhau để trích xuất các đặc điểm từ hình ảnh đầu vào và đưa ra dự đoán. Trong trình chiếu này, chúng ta sẽ khám phá các lớp CNN khác nhau và chức năng của chúng, sử dụng các ví dụ về mã Python để minh họa các khái niệm chính.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Creating a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

Trang trình bày 2: Lớp đầu vào

Lớp đầu vào là lớp đầu tiên của CNN, chịu trách nhiệm nhận và xử lý trước dữ liệu hình ảnh thô. Nó xác định kích thước của hình ảnh đầu vào, bao gồm chiều cao, chiều rộng và số lượng kênh màu.

```python
import numpy as np
import matplotlib.pyplot as plt

# Creating a sample input image
input_image = np.random.rand(28, 28, 1)

# Displaying the input image
plt.imshow(input_image[:,:,0], cmap='gray')
plt.title('Input Image')
plt.show()

# Defining the input layer
input_layer = layers.Input(shape=(28, 28, 1))
```

Trang trình bày 3: Lớp chập

Lớp chập là khối xây dựng cốt lõi của CNN. Nó áp dụng một tập hợp các bộ lọc có thể học được cho đầu vào, tạo ra các bản đồ đặc trưng làm nổi bật các đặc điểm quan trọng trong hình ảnh. Mỗi bộ lọc trượt qua đầu vào, thực hiện phép nhân theo từng phần tử và tính tổng kết quả.

```python
# Creating a convolutional layer
conv_layer = layers.Conv2D(32, (3, 3), activation='relu')

# Applying the convolutional layer to the input
feature_maps = conv_layer(input_layer)

# Visualizing a feature map
plt.imshow(feature_maps[0,:,:,0], cmap='viridis')
plt.title('Feature Map')
plt.show()
```

Slide 4: Chức năng kích hoạt

Các chức năng kích hoạt đưa tính phi tuyến tính vào mạng, cho phép mạng tìm hiểu các mẫu phức tạp. Đơn vị tuyến tính chỉnh lưu (ReLU) thường được sử dụng trong CNN vì nó giúp giảm thiểu vấn đề biến mất độ dốc và tăng tốc độ đào tạo.

```python
import tensorflow as tf

# Implementing ReLU activation
def relu_activation(x):
    return tf.maximum(0, x)

# Applying ReLU to sample data
sample_data = tf.constant([-2, -1, 0, 1, 2], dtype=tf.float32)
activated_data = relu_activation(sample_data)

print("Input:", sample_data.numpy())
print("After ReLU:", activated_data.numpy())
```

Trang trình bày 5: Lớp gộp

Các lớp gộp làm giảm kích thước không gian của bản đồ đặc điểm, giảm độ phức tạp tính toán và giúp đạt được tính bất biến về không gian. Tổng hợp tối đa là loại phổ biến nhất, chọn giá trị tối đa trong mỗi cửa sổ tổng hợp.

```python
# Creating a max pooling layer
pool_layer = layers.MaxPooling2D((2, 2))

# Applying max pooling to the feature maps
pooled_features = pool_layer(feature_maps)

# Visualizing a pooled feature map
plt.imshow(pooled_features[0,:,:,0], cmap='viridis')
plt.title('Pooled Feature Map')
plt.show()
```

Slide 6: Làm phẳng lớp

Lớp làm phẳng biến đổi bản đồ tính năng 2D thành vectơ 1D, chuẩn bị dữ liệu để nhập vào các lớp được kết nối đầy đủ. Quá trình này bảo tồn thông tin từ các lớp tích chập và lớp gộp trong khi thay đổi cấu trúc dữ liệu.

```python
# Creating a flattening layer
flatten_layer = layers.Flatten()

# Flattening the pooled features
flattened_features = flatten_layer(pooled_features)

print("Shape before flattening:", pooled_features.shape)
print("Shape after flattening:", flattened_features.shape)
```

Trang trình bày 7: Lớp được kết nối đầy đủ (Dày đặc)

Các lớp được kết nối đầy đủ lấy vectơ đặc trưng phẳng và thực hiện suy luận cấp cao. Mỗi nơ-ron trong lớp dày đặc được kết nối với mọi nơ-ron ở lớp trước, cho phép mạng kết hợp các tính năng và đưa ra các quyết định phức tạp.

```python
# Creating a dense layer
dense_layer = layers.Dense(64, activation='relu')

# Applying the dense layer to flattened features
dense_output = dense_layer(flattened_features)

print("Dense layer output shape:", dense_output.shape)
```

Slide 8: Lớp đầu ra

Lớp đầu ra tạo ra các dự đoán cuối cùng của CNN. Đối với các nhiệm vụ phân loại, nó thường sử dụng hàm kích hoạt softmax để tạo ra phân bố xác suất trên các lớp có thể.

```python
# Creating an output layer for a 10-class classification problem
output_layer = layers.Dense(10, activation='softmax')

# Generating predictions
predictions = output_layer(dense_output)

print("Predictions shape:", predictions.shape)
print("Sample prediction:", predictions[0].numpy())
```

Trang trình bày 9: Kết hợp tất cả lại với nhau

Bây giờ chúng ta đã khám phá các lớp riêng lẻ, hãy xem cách chúng kết hợp với nhau để tạo thành một kiến ​​trúc CNN hoàn chỉnh. Chúng tôi sẽ tạo một CNN đơn giản để phân loại hình ảnh bằng bộ dữ liệu MNIST.

```python
# Building a CNN for MNIST classification
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

Slide 10: Đào tạo CNN

Việc đào tạo CNN bao gồm việc cung cấp dữ liệu được dán nhãn, so sánh các dự đoán của nó với các nhãn thực và điều chỉnh trọng số của nó để giảm thiểu sai sót. Chúng tôi sử dụng lan truyền ngược và giảm độ dốc để tối ưu hóa các tham số của mạng.

```python
# Loading and preprocessing the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Compiling and training the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)
```

Slide 11: Đánh giá CNN

Sau khi đào tạo, chúng tôi đánh giá hiệu suất của CNN trên một bộ thử nghiệm riêng để đánh giá khả năng khái quát hóa của nó. Chúng ta cũng có thể hình dung quá trình đào tạo để phát hiện các vấn đề như trang bị quá mức.

```python
# Evaluating the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# Plotting training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

Slide 12: Ví dụ thực tế: Phân loại hình ảnh

CNN được sử dụng rộng rãi trong các nhiệm vụ phân loại hình ảnh. Hãy sử dụng mô hình đã được đào tạo của chúng tôi để phân loại một chữ số viết tay từ tập dữ liệu MNIST.

```python
import numpy as np

# Select a random test image
test_image = test_images[np.random.randint(0, len(test_images))]

# Make a prediction
prediction = model.predict(test_image.reshape(1, 28, 28, 1))
predicted_class = np.argmax(prediction)

# Display the image and prediction
plt.imshow(test_image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted Digit: {predicted_class}")
plt.show()
```

Slide 13: Ví dụ thực tế: Trực quan hóa tính năng

Việc hiểu những tính năng mà CNN tìm hiểu có thể cung cấp thông tin chi tiết về quá trình ra quyết định của họ. Hãy hình dung các tính năng được học bởi lớp tích chập đầu tiên trong mô hình của chúng tôi.

```python
# Get the weights of the first convolutional layer
first_layer_weights = model.layers[0].get_weights()[0]

# Plot the learned filters
fig, axs = plt.subplots(4, 8, figsize=(20, 10))
for i in range(32):
    axs[i//8, i%8].imshow(first_layer_weights[:,:,0,i], cmap='viridis')
    axs[i//8, i%8].axis('off')
plt.suptitle("First Layer Filters")
plt.show()
```

Slide 14: Kiến trúc CNN nâng cao

Khi CNN phát triển, các kiến ​​trúc phức tạp hơn cũng được phát triển để cải thiện hiệu suất của nhiều nhiệm vụ khác nhau. Một số ví dụ đáng chú ý bao gồm:

1. VGGNet: Được biết đến với tính đơn giản và có chiều sâu, sử dụng các bộ lọc tích chập 3x3 nhỏ.
2. ResNet: Giới thiệu bỏ qua kết nối để cho phép đào tạo các mạng rất sâu.
3. Khởi động: Các mô-đun khởi động được sử dụng với nhiều kích thước bộ lọc để nắm bắt các tính năng ở các tỷ lệ khác nhau.
4. DenseNet: Kết nối từng lớp với mọi lớp khác theo kiểu chuyển tiếp nguồn cấp dữ liệu, thúc đẩy việc tái sử dụng tính năng.

Những kiến ​​trúc này đã vượt qua ranh giới của những gì có thể làm được với CNN, đạt được những kết quả tiên tiến nhất trong nhiều nhiệm vụ thị giác máy tính.

```python
# Example of a ResNet-like skip connection
def residual_block(x, filters, kernel_size=3):
    y = layers.Conv2D(filters, kernel_size, padding='same')(x)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.Conv2D(filters, kernel_size, padding='same')(y)
    y = layers.BatchNormalization()(y)
    out = layers.Add()([x, y])
    return layers.Activation('relu')(out)

# Using the residual block in a model
inputs = layers.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = residual_block(x, 32)
# ... (add more layers as needed)
outputs = layers.Dense(10, activation='softmax')(x)
resnet_model = models.Model(inputs, outputs)
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về CNN và các ứng dụng của chúng, đây là một số tài nguyên có giá trị:

1. "Phân loại ImageNet với Mạng lưới thần kinh chuyển đổi sâu" của Krizhevsky và cộng sự. (2012) - Bài báo phổ biến CNN để phân loại hình ảnh. ArXiv: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
2. "Mạng chuyển đổi rất sâu để nhận dạng hình ảnh quy mô lớn" của Simonyan và Zisserman (2014) - Giới thiệu kiến trúc VGG. ArXiv: [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)
3. "Học tập sâu để nhận dạng hình ảnh" của He et al. (2015) - Trình bày kiến ​​trúc ResNet. ArXiv: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)
4. "Đi sâu hơn với các kết cấu" của Szegedy và cộng sự. (2014) - Mô tả kiến ​​trúc Inception. ArXiv: [https://arxiv.org/abs/1409.4842](https://arxiv.org/abs/1409.4842)

Những bài viết này cung cấp những giải thích sâu sắc về các kiến ​​trúc CNN chính và tác động của chúng đối với lĩnh vực thị giác máy tính.
