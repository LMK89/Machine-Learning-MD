## Làm chủ mạng nơ-ron tích chập bằng Python
Trang trình bày 1: Giới thiệu về Mạng thần kinh chuyển đổi (CNN)

Mạng thần kinh chuyển đổi là một lớp mô hình học sâu được thiết kế để xử lý dữ liệu dạng lưới, chẳng hạn như hình ảnh. Chúng đặc biệt hiệu quả đối với các tác vụ như phân loại hình ảnh, phát hiện đối tượng và nhận dạng khuôn mặt. CNN sử dụng các lớp chuyên biệt áp dụng các phép toán tích chập để trích xuất các tính năng từ dữ liệu đầu vào.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

# Load a sample image
china = load_sample_image("china.jpg")
plt.imshow(china)
plt.axis('off')
plt.title("Sample Image for CNN Processing")
plt.show()
```

Slide 2: Tổng quan về kiến ​​trúc CNN

Kiến trúc CNN điển hình bao gồm một số thành phần chính: lớp tích chập, lớp gộp và lớp được kết nối đầy đủ. Các lớp tích chập áp dụng các bộ lọc cho đầu vào, các lớp gộp làm giảm kích thước không gian và các lớp được kết nối đầy đủ sẽ đưa ra dự đoán cuối cùng.

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()
```

Trang trình bày 3: Lớp chập

Các lớp tích chập là các khối xây dựng cốt lõi của CNN. Họ áp dụng một bộ bộ lọc có thể học được cho đầu vào, tạo ra các bản đồ đặc trưng làm nổi bật các đặc điểm quan trọng của dữ liệu. Mỗi bộ lọc trượt qua đầu vào, thực hiện phép nhân theo từng phần tử và tính tổng kết quả.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 5x5 image
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

# Define a 3x3 filter
filter = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

# Perform convolution
output = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        output[i, j] = np.sum(image[i:i+3, j:j+3] * filter)

# Visualize the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Input Image')
ax2.imshow(filter, cmap='gray')
ax2.set_title('Filter')
ax3.imshow(output, cmap='gray')
ax3.set_title('Output Feature Map')
plt.show()
```

Slide 4: Hàm kích hoạt trong CNN

Các chức năng kích hoạt đưa tính phi tuyến tính vào mạng, cho phép mạng tìm hiểu các mẫu phức tạp. Đơn vị tuyến tính chỉnh lưu (ReLU) là một lựa chọn phổ biến cho CNN do tính đơn giản và hiệu quả của nó trong việc giảm thiểu vấn đề độ dốc biến mất.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

x = np.linspace(-10, 10, 100)

plt.figure(figsize=(12, 4))
plt.plot(x, relu(x), label='ReLU')
plt.plot(x, leaky_relu(x), label='Leaky ReLU')
plt.title('ReLU and Leaky ReLU Activation Functions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()
```

Trang trình bày 5: Các lớp gộp

Các lớp gộp làm giảm kích thước không gian của bản đồ đặc điểm, giảm độ phức tạp tính toán và giúp đạt được tính bất biến về không gian. Tổng hợp tối đa thường được sử dụng, lấy giá trị tối đa trong mỗi cửa sổ tổng hợp.

```python
import numpy as np
import matplotlib.pyplot as plt

def max_pooling(input_array, pool_size):
    input_height, input_width = input_array.shape
    pool_height, pool_width = pool_size

    output_height = input_height // pool_height
    output_width = input_width // pool_width

    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.max(input_array[i*pool_height:(i+1)*pool_height,
                                              j*pool_width:(j+1)*pool_width])

    return output

# Create a sample 4x4 input
input_array = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# Apply max pooling with a 2x2 window
pooled = max_pooling(input_array, (2, 2))

# Visualize the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(input_array, cmap='viridis')
ax1.set_title('Input Array')
ax2.imshow(pooled, cmap='viridis')
ax2.set_title('After Max Pooling (2x2)')
plt.show()
```

Slide 6: Các lớp được kết nối đầy đủ

Các lớp được kết nối đầy đủ sẽ xuất hiện sau các lớp chập và lớp gộp. Chúng lấy đầu ra đã được làm phẳng từ các lớp trước đó và thực hiện nhiệm vụ phân loại hoặc hồi quy cuối cùng. Các lớp này tìm hiểu các mẫu chung trong không gian đặc trưng.

```python
import tensorflow as tf

# Create a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Visualize the model architecture
tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)
```

Slide 7: Tiền xử lý dữ liệu cho CNN

Việc xử lý trước dữ liệu phù hợp là rất quan trọng để đào tạo CNN hiệu quả. Điều này bao gồm thay đổi kích thước hình ảnh, chuẩn hóa giá trị pixel và tăng cường dữ liệu để tăng tính đa dạng của tập huấn luyện.

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    rescale=1./255
)

# Load and preprocess a sample image
img = tf.keras.preprocessing.image.load_img('sample_image.jpg', target_size=(150, 150))
x = tf.keras.preprocessing.image.img_to_array(img)
x = x.reshape((1,) + x.shape)

# Generate augmented images
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure()
    imgplot = plt.imshow(tf.keras.preprocessing.image.array_to_img(batch[0]))
    plt.axis('off')
    i += 1
    if i % 5 == 0:
        break

plt.show()
```

Slide 8: Đào tạo CNN

Việc huấn luyện một CNN bao gồm việc truyền dữ liệu về phía trước, tính toán tổn hao, lan truyền ngược và cập nhật tham số. Chúng tôi sử dụng các thuật toán tối ưu hóa như Stochastic gradient Descent (SGD) hoặc Adam để giảm thiểu hàm mất mát.

```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

# Create a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, validation_split=0.2)

# Plot the training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 9: Chuyển giao học tập với CNN

Học chuyển giao cho phép chúng tôi tận dụng các mô hình được đào tạo trước trên các tập dữ liệu lớn để cải thiện hiệu suất đối với các tác vụ nhỏ hơn, có liên quan. Chúng ta có thể sử dụng các kiến ​​trúc phổ biến như VGG, ResNet hoặc Inception làm trình trích xuất tính năng hoặc tinh chỉnh chúng cho các tác vụ cụ thể.

```python
import tensorflow as tf

# Load a pre-trained VGG16 model
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for our specific task
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()
```

Slide 10: Trực quan hóa các tính năng của CNN

Trực quan hóa các tính năng mà CNN đã học giúp chúng tôi hiểu mạng đang tập trung vào điều gì. Chúng ta có thể sử dụng các kỹ thuật như tối đa hóa kích hoạt hoặc các phương pháp dựa trên độ dốc để tạo ra hình ảnh giúp tối đa hóa việc kích hoạt các nơ-ron cụ thể.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def visualize_filters(model, layer_name, filter_index):
    layer = model.get_layer(layer_name)

    # Create a model that maps the input to the activations of the target layer
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)

    # Start from a gray image with some noise
    input_img_data = np.random.random((1, 224, 224, 3)) * 20 + 128.

    # Define the loss as the mean activation of a specific filter
    loss = tf.reduce_mean(feature_extractor(input_img_data)[:, :, :, filter_index])

    # Compute the gradient of the input picture with respect to this loss
    grads = tf.GradientTape().gradient(loss, input_img_data)

    # Normalization trick: we normalize the gradient
    grads /= (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)

    # Perform gradient ascent
    input_img_data += grads * 20

    img = input_img_data[0].astype(np.uint8)
    return img

# Assuming we have a pre-trained VGG16 model
model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

# Visualize the first 4 filters of the first convolutional layer
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    img = visualize_filters(model, 'block1_conv1', i)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'Filter {i}')

plt.tight_layout()
plt.show()
```

Slide 11: Ví dụ thực tế: Phân loại hình ảnh

Phân loại ảnh là một ứng dụng phổ biến của CNN. Hãy sử dụng mô hình MobileNetV2 được đào tạo trước để phân loại hình ảnh từ bộ dữ liệu CIFAR-10 phổ biến, chứa 60.000 hình ảnh màu 32x32 trong 10 lớp.

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Preprocess the data
x_test = tf.keras.applications.mobilenet_v2.preprocess_input(x_test)

# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Make predictions
predictions = model.predict(x_test)

# Display some predictions
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_test[i])
    plt.title(f"Actual: {class_names[y_test[i][0]]}\nPredicted: {class_names[np.argmax(predictions[i])]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
```

Trang trình bày 12: Ví dụ thực tế: Phát hiện đối tượng

Phát hiện đối tượng là một ứng dụng mạnh mẽ khác của CNN. Chúng tôi sẽ sử dụng mô hình YOLO (Bạn chỉ nhìn một lần) được đào tạo trước để phát hiện các đối tượng trong ảnh. YOLO được biết đến với tốc độ và độ chính xác trong việc phát hiện đối tượng theo thời gian thực.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load image
image = cv2.imread("sample_image.jpg")
height, width = image.shape[:2]

# Prepare image for YOLO
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get output layer names
output_layers = net.getUnconnectedOutLayersNames()

# Forward pass
outs = net.forward(output_layers)

# Process detections
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding boxes
for i in indices:
    box = boxes[i]
    x, y, w, h = box
    label = str(classes[class_ids[i]])
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, f"{label} {confidences[i]:.2f}", (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the result
plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Object Detection with YOLO")
plt.show()
```

Slide 13: Xử lý Overfitting trong CNN

Quá khớp xảy ra khi một mô hình hoạt động tốt trên dữ liệu huấn luyện nhưng kém trên dữ liệu không nhìn thấy. Để chống lại điều này, chúng ta có thể sử dụng các kỹ thuật như tăng cường dữ liệu, loại bỏ và chính quy hóa.

```python
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2

model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1),
           kernel_regularizer=l2(0.01)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Train the model with data augmentation
history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                    epochs=10, validation_data=(x_test, y_test))

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.legend()

plt.show()
```

Slide 14: Diễn giải các quyết định của CNN

Hiểu lý do tại sao CNN đưa ra một số quyết định nhất định là rất quan trọng để xây dựng niềm tin vào mô hình và gỡ lỗi. Các kỹ thuật như Grad-CAM (Ánh xạ kích hoạt lớp theo độ dốc) có thể giúp trực quan hóa phần nào của hình ảnh là quan trọng đối với quyết định của mô hình.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Assume we have a pre-trained model and an input image
model = tf.keras.applications.MobileNetV2(weights='imagenet')
img = tf.keras.preprocessing.image.load_img('elephant.jpg', target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

heatmap = make_gradcam_heatmap(img_array, model, 'Conv_1')

plt.matshow(heatmap)
plt.title("Grad-CAM Heatmap")
plt.show()

# Superimpose heatmap on original image
img = tf.keras.preprocessing.image.img_to_array(img)
heatmap = np.uint8(255 * heatmap)
jet = plt.cm.get_cmap("jet")
jet_colors = jet(np.arange(256))[:, :3]
jet_heatmap = jet_colors[heatmap]
jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

superimposed_img = jet_heatmap * 0.4 + img
superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

plt.imshow(superimposed_img)
plt.title("Grad-CAM Result")
plt.axis('off')
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Để khám phá thêm về Mạng thần kinh chuyển đổi, hãy xem xét các tài nguyên sau:

1. "Học sâu" của Ian Goodfellow, Yoshua Bengio và Aaron Courville (Có sẵn trực tuyến: [http://www.deeplearningbook.org/](http://www.deeplearningbook.org/))
2. CS231n: Mạng thần kinh tích chập để nhận dạng hình ảnh (Tài liệu khóa học của Đại học Stanford: [http://cs231n.stanford.edu/](http://cs231n.stanford.edu/))
3. Bài báo ArXiv: "Khảo sát về kiến trúc gần đây của mạng lưới thần kinh chuyển đổi sâu" của Khan và cộng sự. (2020) (ArXiv:1901.06032)
4. Bài viết ArXiv: "Trực quan hóa và hiểu về mạng tích chập" của Zeiler và Fergus (2013) (ArXiv:1311.2901)
5. Tài liệu về TensorFlow và Keras để triển khai CNN ([https://www.tensorflow.org/tutorials/images/cnn](https://www.tensorflow.org/tutorials/images/cnn))

Những tài nguyên này cung cấp sự kết hợp giữa nền tảng lý thuyết và cách triển khai thực tế để giúp bạn hiểu sâu hơn về CNN.
