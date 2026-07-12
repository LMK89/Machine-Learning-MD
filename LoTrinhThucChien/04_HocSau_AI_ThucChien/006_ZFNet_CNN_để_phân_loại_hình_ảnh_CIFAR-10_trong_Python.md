## ZFNet CNN để phân loại hình ảnh CIFAR-10 trong Python
Trang trình bày 1:

Giới thiệu ZFNet để phân loại CIFAR-10

ZFNet, được Zeiler và Fergus giới thiệu vào năm 2013, là kiến ​​trúc Mạng thần kinh chuyển đổi (CNN) đã đạt được kết quả tiên tiến trên bộ dữ liệu phân loại hình ảnh CIFAR-10. Trong phần trình bày này, chúng ta sẽ khám phá cách triển khai ZFNet bằng Python và thư viện Pandas để xử lý và thao tác trước dữ liệu.

```python
import pandas as pd
import numpy as np
from keras.datasets import cifar10
```

Slide 2:

Loading the CIFAR-10 Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. We can load the dataset using the Keras library.

```python
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

Trang trình bày 3:

Tiền xử lý dữ liệu

Trước khi cung cấp dữ liệu cho mô hình ZFNet, chúng ta cần xử lý trước dữ liệu. Điều này thường liên quan đến việc chuẩn hóa và định hình lại dữ liệu theo định dạng đầu vào dự kiến.

```python
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

X_train = X_train.reshape(-1, 32, 32, 3)
X_test = X_test.reshape(-1, 32, 32, 3)
```

Trang trình bày 4:

Mã hóa một lần các nhãn

Vì các nhãn trong tập dữ liệu CIFAR-10 là số nguyên nên chúng tôi cần mã hóa chúng một lần trước khi sử dụng chúng làm mục tiêu cho nhiệm vụ phân loại.

```python
from keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
```

Trang trình bày 5:

Xác định kiến ​​trúc ZFNet

ZFNet là một kiến ​​trúc CNN bao gồm một số lớp chập, gộp và được kết nối đầy đủ. Chúng ta có thể xác định kiến ​​trúc bằng thư viện Keras.

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential([
    Conv2D(96, (7, 7), strides=(2, 2), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
    # ... (Add more layers here)
])
```

Trang trình bày 6:

Biên dịch mô hình

Sau khi xác định được kiến ​​trúc mô hình, chúng ta cần biên dịch nó bằng trình tối ưu hóa, hàm mất mát và số liệu đánh giá.

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

Slide 7:

Training the Model

We can train the ZFNet model on the CIFAR-10 dataset using the `fit` method from Keras.

```python
model.fit(X_train, y_train,
          batch_size=64,
          epochs=50,
          validation_data=(X_test, y_test))
```

Trang trình bày 8:

Đánh giá mô hình

Sau khi đào tạo, chúng ta có thể đánh giá hiệu suất của mô hình trên tập kiểm tra bằng phương pháp `evaluate`.

```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc * 100:.2f}%')
```

Slide 9:

Making Predictions

To make predictions on new data, we can use the `predict` method of the trained model.

```python
import matplotlib.pyplot as plt

# Load a new image
new_image = ... # Load image data here

# Preprocess the new image
new_image = new_image.reshape(1, 32, 32, 3)
new_image = new_image.astype('float32') / 255

# Make a prediction
prediction = model.predict(new_image)
class_idx = np.argmax(prediction)
class_name = cifar10.load_data()[1].class_names[class_idx]

# Display the image and prediction
plt.imshow(new_image.reshape(32, 32, 3))
plt.title(f'Prediction: {class_name}')
plt.show()
```

Trang trình bày 10:

Tăng cường dữ liệu

Tăng cường dữ liệu có thể được sử dụng để tăng kích thước của tập dữ liệu huấn luyện một cách giả tạo và cải thiện hiệu suất mô hình. Chúng ta có thể sử dụng lớp ImageDataGenerator từ Keras cho mục đích này.

```python
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow(X_train, y_train, batch_size=64)
```

Trang trình bày 11:

Chuyển giao học tập với ZFNet

Học chuyển là một kỹ thuật trong đó chúng ta có thể sử dụng các trọng số được đào tạo trước từ một mô hình được đào tạo trên một tập dữ liệu lớn và tinh chỉnh nó cho nhiệm vụ cụ thể của chúng ta. Điều này có thể dẫn đến hiệu suất tốt hơn và hội tụ nhanh hơn.

```python
from keras.applications import ZFNet

base_model = ZFNet(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)

# Create the transfer learning model
transfer_model = Model(inputs=base_model.input, outputs=predictions)

# Compile and train the model
transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
transfer_model.fit(train_generator, epochs=10, validation_data=(X_test, y_test))
```

Trang trình bày 12:

Trực quan hóa ZFNet

Chúng ta có thể trực quan hóa các bộ lọc đã học và bản đồ tính năng của mô hình ZFNet để hiểu rõ hơn về các biểu diễn bên trong của nó.

```python
from keras.models import Model

# Create a model that outputs the activations of a specific layer
layer_idx = 2  # Index of the layer to visualize
layer_outputs = [layer.output for layer in model.layers[:layer_idx+1]]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

# Visualize the activations
img = X_test[0].reshape(1, 32, 32, 3)
activations = activation_model.predict(img)

# Plot the activations
for i, activation in enumerate(activations):
    plt.subplot(1, len(activations), i+1)
    plt.imshow(activation[0, :, :, :])
    plt.title(f'Layer {i}')
plt.show()
```

Trang trình bày 13:

Phân tích hiệu suất ZFNet

Chúng tôi có thể phân tích hiệu suất của mô hình ZFNet bằng cách đánh giá độ chính xác, độ chính xác, khả năng thu hồi và điểm F1 của nó trên bộ kiểm tra.

```python
from sklearn.metrics import precision_score, recall_score, f1_score

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = (y_pred == y_true).mean()
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
```

Trang trình bày 14:

Tài nguyên bổ sung

Để đọc và khám phá thêm, đây là một số tài nguyên bổ sung trên ZFNet và các chủ đề liên quan từ ArXiv.org:

1. Zeiler, MD, & Fergus, R. (2013). Trực quan hóa và hiểu các mạng tích chập. arXiv:1311.2901 \[cs.CV\] [https://arxiv.org/abs/1311.2901](https://arxiv.org/abs/1311.2901)
2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Phân loại ImageNet với Mạng thần kinh chuyển đổi sâu. arXiv:1202.2683 \[cs.CV\] [https://arxiv.org/abs/1202.2683](https://arxiv.org/abs/1202.2683)
3. Simonyan, K., & Zisserman, A. (2014). Mạng tích chập rất sâu để nhận dạng hình ảnh quy mô lớn. arXiv:1409.1556 \[cs.CV\] [https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)
4. Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Đi sâu hơn với các kết cấu. arXiv:1409.4842 \[cs.CV\] [https://arxiv.org/abs/1409.4842](https://arxiv.org/abs/1409.4842)
5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Học tập dư thừa sâu để nhận dạng hình ảnh. arXiv:1512.03385 \[cs.CV\] [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

Các bài viết này bao gồm kiến ​​trúc ZFNet ban đầu, kiến ​​trúc AlexNet đã truyền cảm hứng cho ZFNet, các kiến ​​trúc CNN sâu hơn như VGGNet và GoogLeNet, cũng như kiến ​​trúc ResNet đột phá, được xây dựng dựa trên ý tưởng từ các kiến ​​trúc trước đó như ZFNet.
