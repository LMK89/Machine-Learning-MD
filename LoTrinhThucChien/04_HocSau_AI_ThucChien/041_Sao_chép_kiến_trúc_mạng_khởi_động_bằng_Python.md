## Sao chép kiến trúc mạng khởi động bằng Python
Slide 1: Giới thiệu về InceptionNet

InceptionNet, còn được gọi là GoogLeNet, là kiến ​​trúc mạng nơ-ron tích chập sâu được thiết kế để cải thiện hiệu quả và độ chính xác trong các tác vụ phân loại hình ảnh. Được phát triển bởi các nhà nghiên cứu của Google vào năm 2014, nó đã giới thiệu khái niệm "mô-đun khởi động" cho phép mạng nắm bắt các tính năng ở nhiều quy mô cùng một lúc.

```python
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3

# Load pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet', include_top=True)

# Display model summary
model.summary()
```

Slide 2: Mô-đun khởi động

Điểm đổi mới quan trọng của InceptionNet là mô-đun khởi động. Mô-đun này thực hiện song song các tích chập với nhiều kích thước bộ lọc (1x1, 3x3, 5x5), cho phép mạng nắm bắt cả các tính năng cục bộ và toàn cầu một cách hiệu quả.

```python
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, Input

def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)

    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv_5x5)

    pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(pool_proj)

    output = Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5, pool_proj])
    return output

# Example usage
input_tensor = Input(shape=(299, 299, 3))
inception_output = inception_module(input_tensor, 64, 96, 128, 16, 32, 32)
```

Slide 3: Hệ số tích chập

Hệ số tích chập là một kỹ thuật được sử dụng trong InceptionNet để giảm độ phức tạp tính toán. Nó liên quan đến việc chia nhỏ các tổ hợp lớn hơn thành các hoạt động nhỏ hơn, hiệu quả hơn.

```python
from tensorflow.keras.layers import Conv2D

def factorized_conv(x, filters, kernel_size):
    # Factorize nxn convolution into two consecutive 1xn and nx1 convolutions
    conv_1xn = Conv2D(filters, (1, kernel_size), padding='same', activation='relu')(x)
    conv_nx1 = Conv2D(filters, (kernel_size, 1), padding='same', activation='relu')(conv_1xn)
    return conv_nx1

# Example usage
input_tensor = Input(shape=(299, 299, 3))
factorized_output = factorized_conv(input_tensor, 64, 3)
```

Trang trình bày 4: Phép cuộn 1x1 để giảm kích thước

InceptionNet sử dụng các phép tích chập 1x1 để giảm số lượng bản đồ đặc trưng trước khi áp dụng các phép tích chập lớn hơn, giúp giảm đáng kể chi phí tính toán.

```python
from tensorflow.keras.layers import Conv2D

def dimension_reduction(x, filters_reduce, filters_conv):
    # Apply 1x1 convolution for dimensionality reduction
    x = Conv2D(filters_reduce, (1, 1), padding='same', activation='relu')(x)
    # Apply 3x3 convolution
    x = Conv2D(filters_conv, (3, 3), padding='same', activation='relu')(x)
    return x

# Example usage
input_tensor = Input(shape=(299, 299, 256))
reduced_output = dimension_reduction(input_tensor, 64, 192)
```

Slide 5: Bộ phân loại phụ trợ

InceptionNet kết hợp các bộ phân loại phụ trợ ở các lớp giữa để giải quyết vấn đề biến mất độ dốc và cung cấp khả năng chính quy hóa bổ sung.

```python
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

def auxiliary_classifier(x, num_classes):
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)
    return x

# Example usage
intermediate_output = inception_module(input_tensor, 64, 96, 128, 16, 32, 32)
auxiliary_output = auxiliary_classifier(intermediate_output, 1000)
```

Trang trình bày 6: Tổng hợp trung bình toàn cầu

InceptionNet thay thế các lớp được kết nối đầy đủ ở đầu mạng bằng tính năng gộp chung trung bình toàn cầu, giảm số lượng tham số và giảm thiểu tình trạng trang bị quá mức.

```python
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

def global_avg_pooling_classifier(x, num_classes):
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    return x

# Example usage
final_inception_output = inception_module(input_tensor, 384, 192, 384, 48, 128, 128)
final_output = global_avg_pooling_classifier(final_inception_output, 1000)
```

Trang trình bày 7: Kiến trúc mạng trong mạng

InceptionNet kết hợp khái niệm Network-in-Network, sử dụng các perceptron nhiều lớp trong các lớp tích chập để tăng độ sâu và tính biểu cảm của mạng.

```python
from tensorflow.keras.layers import Conv2D

def network_in_network(x, filters):
    x = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    return x

# Example usage
input_tensor = Input(shape=(299, 299, 3))
nin_output = network_in_network(input_tensor, 64)
```

Trang trình bày 8: Chuẩn hóa hàng loạt

InceptionNet v2 và các phiên bản mới hơn kết hợp chuẩn hóa hàng loạt để cải thiện độ ổn định trong quá trình huấn luyện và tốc độ hội tụ.

```python
from tensorflow.keras.layers import BatchNormalization, Activation

def batch_norm_relu(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

# Example usage
conv_output = Conv2D(64, (3, 3), padding='same')(input_tensor)
normalized_output = batch_norm_relu(conv_output)
```

Trang trình bày 9: Làm mịn nhãn

InceptionNet v2 giới thiệu tính năng làm mịn nhãn, một kỹ thuật chính quy hóa giúp cải thiện tính khái quát hóa bằng cách ngăn mô hình trở nên quá tự tin.

```python
import tensorflow as tf

def label_smoothing(labels, factor=0.1):
    num_classes = tf.shape(labels)[-1]
    smooth_labels = labels * (1.0 - factor) + (factor / tf.cast(num_classes, tf.float32))
    return smooth_labels

# Example usage
true_labels = tf.constant([[0, 1, 0], [1, 0, 0]])
smoothed_labels = label_smoothing(true_labels)
print(smoothed_labels)
```

Trang trình bày 10: Ví dụ thực tế: Phân loại hình ảnh

InceptionNet được sử dụng rộng rãi cho các nhiệm vụ phân loại hình ảnh. Dưới đây là ví dụ về việc sử dụng mô hình InceptionV3 được đào tạo trước để phân loại hình ảnh.

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Load pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

# Load and preprocess an image
img_path = 'path/to/your/image.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make predictions
preds = model.predict(x)
decoded_preds = decode_predictions(preds, top=3)[0]

# Print results
for _, label, score in decoded_preds:
    print(f"{label}: {score:.2f}")
```

Slide 11: Ví dụ thực tế: Học chuyển tiếp

Kiến trúc của InceptionNet thường được sử dụng làm cơ sở cho việc học chuyển giao trong các nhiệm vụ thị giác máy tính khác nhau. Đây là ví dụ về cách sử dụng InceptionV3 cho tác vụ phân loại tùy chỉnh.

```python
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Load pre-trained InceptionV3 model without top layers
base_model = InceptionV3(weights='imagenet', include_top=False)

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(10, activation='softmax')(x)  # 10 classes in this example

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model (assuming you have your data ready)
# model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
```

Trang trình bày 12: Các biến thể khởi đầu

Một số biến thể của kiến ​​trúc Inception đã được đề xuất, mỗi biến thể đều đưa ra những cải tiến và ý tưởng mới.

```python
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2, Xception

# InceptionV3
inceptionv3 = InceptionV3(weights='imagenet', include_top=False)

# Inception-ResNet-V2 (combines Inception with residual connections)
inceptionresnetv2 = InceptionResNetV2(weights='imagenet', include_top=False)

# Xception (extreme version of Inception, replacing Inception modules with depthwise separable convolutions)
xception = Xception(weights='imagenet', include_top=False)

# Print model summaries
print("InceptionV3:")
inceptionv3.summary()

print("\nInception-ResNet-V2:")
inceptionresnetv2.summary()

print("\nXception:")
xception.summary()
```

Slide 13: Kết luận và định hướng tương lai

InceptionNet đã ảnh hưởng đáng kể đến lĩnh vực deep learning và thị giác máy tính. Các khái niệm của nó tiếp tục phù hợp trong các kiến ​​trúc hiện đại và nghiên cứu đang được tiến hành để nâng cao hơn nữa hiệu quả và hiệu suất trong các mạng lưới thần kinh.

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulating performance improvements over time
versions = ['InceptionV1', 'InceptionV2', 'InceptionV3', 'Inception-ResNet-V2', 'Future?']
accuracy = [0.89, 0.915, 0.937, 0.953, 0.97]

plt.figure(figsize=(10, 6))
plt.plot(versions, accuracy, marker='o')
plt.title('Inception Architecture Performance Over Time')
plt.xlabel('Version')
plt.ylabel('Top-5 Accuracy on ImageNet')
plt.ylim(0.88, 0.98)
plt.grid(True)
plt.show()
```

Trang trình bày 14: Tài nguyên bổ sung

Để biết thêm thông tin chuyên sâu về InceptionNet và các biến thể của nó, hãy tham khảo các tài liệu nghiên cứu sau:

1. Szegedy, C., và cộng sự. (2015). Đi sâu hơn với các cuộn xoắn. ArXiv:1409.4842 \[cs.CV\] URL: [https://arxiv.org/abs/1409.4842](https://arxiv.org/abs/1409.4842)
2. Szegedy, C., và cộng sự. (2016). Xem xét lại Kiến trúc khởi đầu cho thị giác máy tính. ArXiv:1512.00567 \[cs.CV\] URL: [https://arxiv.org/abs/1512.00567](https://arxiv.org/abs/1512.00567)
3. Szegedy, C., và cộng sự. (2017). Inception-v4, Inception-ResNet và tác động của các kết nối còn lại đối với việc học. ArXiv:1602.07261 \[cs.CV\] URL: [https://arxiv.org/abs/1602.07261](https://arxiv.org/abs/1602.07261)

Những bài viết này cung cấp những giải thích chi tiết về kiến ​​trúc, các lựa chọn thiết kế và kết quả thử nghiệm.
