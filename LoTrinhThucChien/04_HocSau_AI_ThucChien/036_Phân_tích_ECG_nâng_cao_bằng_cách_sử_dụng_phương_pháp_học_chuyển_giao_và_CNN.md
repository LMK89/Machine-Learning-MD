## Phân tích ECG nâng cao bằng cách sử dụng Học chuyển và CNN
Trang trình bày 1: Tổng quan về dự án: Phân tích ECG với Deep Learning

Dự án này tập trung vào việc phân loại hình ảnh ECG bằng kỹ thuật học sâu tiên tiến. Chúng ta sẽ khám phá việc chuẩn bị dữ liệu, phát triển mô hình và ứng dụng học chuyển giao để cải thiện độ chính xác trong phân loại cho các tình trạng tim khác nhau.

```python
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load and prepare the data
train_generator = train_datagen.flow_from_directory(
    'path/to/train_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'path/to/test_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

Slide 2: Khám phá và trực quan hóa dữ liệu

Hiểu được tập dữ liệu là rất quan trọng. Chúng tôi sẽ trực quan hóa các mẫu từ từng danh mục ECG để hiểu rõ hơn về đặc điểm và phân bổ dữ liệu.

```python
import matplotlib.pyplot as plt
import numpy as np

# Function to plot sample images
def plot_samples(generator, n=4):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    for i in range(n):
        images, labels = next(generator)
        ax = axes[i]
        ax.imshow(images[0])
        ax.set_title(f"Class: {np.argmax(labels[0])}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Plot sample images
plot_samples(train_generator)
```

Trang trình bày 3: Mô hình cơ sở: CNN tùy chỉnh

Chúng tôi sẽ bắt đầu với Mạng thần kinh chuyển đổi (CNN) tùy chỉnh làm mô hình cơ sở để thiết lập các chỉ số hiệu suất ban đầu.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the baseline CNN model
def create_baseline_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(4, activation='softmax')  # 4 classes
    ])
    return model

baseline_model = create_baseline_model()
baseline_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the baseline model
history = baseline_model.fit(train_generator, epochs=10, validation_data=test_generator)
```

Trang trình bày 4: Hiệu suất của mô hình cơ sở

Hãy đánh giá hiệu suất của mô hình CNN cơ bản của chúng tôi và trực quan hóa tiến trình đào tạo.

```python
# Evaluate the baseline model
test_loss, test_acc = baseline_model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.2f}")

# Plot training history
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

Slide 5: Chuyển giao học tập với VGG16

Để cải thiện hiệu suất của mô hình, chúng tôi sẽ tận dụng phương pháp học chuyển giao bằng mô hình VGG16 được đào tạo trước.

```python
# Load VGG16 model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Create new model on top
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=20, validation_data=test_generator)
```

Trang trình bày 6: Hiệu suất của mô hình học tập chuyển giao

Chúng tôi sẽ đánh giá hiệu suất của mô hình học chuyển giao của chúng tôi và so sánh nó với mô hình cơ sở.

```python
# Evaluate the transfer learning model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc:.2f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Transfer Learning Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Transfer Learning Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

Trang trình bày 7: Tinh chỉnh mô hình học tập chuyển giao

Để cải thiện hiệu suất hơn nữa, chúng tôi sẽ tinh chỉnh một số lớp cuối cùng của mô hình VGG16.

```python
# Unfreeze the last 4 layers of the base model
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tune the model
history_fine = model.fit(train_generator,
                         epochs=10,
                         validation_data=test_generator)
```

Slide 8: Tinh chỉnh hiệu suất mô hình

Hãy đánh giá hiệu suất của mô hình học chuyển giao đã được tinh chỉnh của chúng tôi.

```python
# Evaluate the fine-tuned model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy after fine-tuning: {test_acc:.2f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_fine.history['accuracy'], label='Training Accuracy')
plt.plot(history_fine.history['val_accuracy'], label='Validation Accuracy')
plt.title('Fine-tuned Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_fine.history['loss'], label='Training Loss')
plt.plot(history_fine.history['val_loss'], label='Validation Loss')
plt.title('Fine-tuned Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

Slide 9: Dự đoán và trực quan hóa mô hình

Chúng tôi sẽ sử dụng mô hình đã đào tạo của mình để đưa ra dự đoán về dữ liệu thử nghiệm và trực quan hóa kết quả.

```python
import numpy as np

# Get a batch of test images
test_images, test_labels = next(test_generator)

# Make predictions
predictions = model.predict(test_images)

# Function to plot images with predictions
def plot_predictions(images, true_labels, predictions, n=4):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()

    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i])
        true_class = np.argmax(true_labels[i])
        pred_class = np.argmax(predictions[i])
        ax.set_title(f"True: {true_class}, Pred: {pred_class}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Plot predictions
plot_predictions(test_images, test_labels, predictions)
```

Slide 10: Khả năng diễn giải mô hình với Grad-CAM

Để hiểu những tính năng mà mô hình của chúng tôi tập trung vào, chúng tôi sẽ sử dụng Ánh xạ kích hoạt lớp theo trọng số gradient (Grad-CAM).

```python
from tensorflow.keras.models import Model

# Create a Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
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

# Generate and plot Grad-CAM for a sample image
img = test_images[0]
heatmap = make_gradcam_heatmap(img[np.newaxis, ...], model, 'block5_conv3')

plt.matshow(heatmap)
plt.title("Grad-CAM Heatmap")
plt.show()
```

Slide 11: Ví dụ thực tế: Sàng lọc ECG tự động

Trong môi trường bệnh viện, mô hình của chúng tôi có thể được sử dụng để sàng lọc nhanh chóng các bất thường tiềm ẩn trên ECG, cho phép các chuyên gia y tế ưu tiên các trường hợp cần được chăm sóc ngay lập tức.

```python
def ecg_screening(ecg_image_path, model):
    # Load and preprocess the ECG image
    img = tf.keras.preprocessing.image.load_img(ecg_image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    confidence = prediction[0][class_index]

    classes = ['Normal', 'Abnormal Beat', 'Myocardial Infarction', 'Other']
    result = f"ECG Classification: {classes[class_index]}"
    result += f"\nConfidence: {confidence:.2f}"

    return result

# Example usage
ecg_path = 'path/to/patient_ecg.jpg'
screening_result = ecg_screening(ecg_path, model)
print(screening_result)
```

Slide 12: Ví dụ thực tế: Hệ thống giám sát ECG

Mô hình của chúng tôi có thể được tích hợp vào hệ thống theo dõi ECG liên tục cho bệnh nhân trong các đơn vị chăm sóc đặc biệt, cảnh báo nhân viên y tế về các biến cố tim mạch tiềm ẩn trong thời gian thực.

```python
import time

def continuous_ecg_monitoring(model, interval=60):
    while True:
        # Simulate getting a new ECG reading every 'interval' seconds
        ecg_data = simulate_ecg_reading()  # This function would capture real ECG data

        # Preprocess the ECG data
        processed_ecg = preprocess_ecg(ecg_data)

        # Make prediction
        prediction = model.predict(processed_ecg)
        class_index = np.argmax(prediction[0])
        confidence = prediction[0][class_index]

        classes = ['Normal', 'Abnormal Beat', 'Myocardial Infarction', 'Other']

        if class_index != 0:  # If not normal
            alert_medical_staff(classes[class_index], confidence)

        time.sleep(interval)

def simulate_ecg_reading():
    # This function would be replaced with actual ECG data acquisition
    return np.random.rand(224, 224, 3)

def preprocess_ecg(ecg_data):
    # Preprocess the ECG data for the model
    return np.expand_dims(ecg_data, axis=0) / 255.0

def alert_medical_staff(condition, confidence):
    print(f"ALERT: Possible {condition} detected. Confidence: {confidence:.2f}")
    # In a real system, this would send an alert to the medical staff

# Example usage
continuous_ecg_monitoring(model, interval=10)  # Check every 10 seconds for demonstration
```

Trang trình bày 13: Những cải tiến và cân nhắc trong tương lai

Mặc dù mô hình của chúng tôi cho thấy kết quả đầy hứa hẹn nhưng vẫn luôn có cơ hội để cải thiện. Hãy xem xét các bước sau để cải tiến trong tương lai:

1. Thu thập dữ liệu ECG đa dạng hơn để cải thiện khả năng khái quát hóa mô hình.
2. Thử nghiệm với các mô hình được đào tạo trước khác như ResNet hoặc EfficiencyNet.
3. Triển khai các kỹ thuật AI có thể giải thích để có khả năng diễn giải mô hình tốt hơn.
4. Tiến hành các thử nghiệm lâm sàng để xác nhận hiệu suất của mô hình trong các tình huống thực tế.

```python
# Example of using a different pre-trained model (ResNet50)
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
output = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=20, validation_data=test_generator)
```

Trang trình bày 14: Tài nguyên bổ sung

Để khám phá thêm về phân tích ECG bằng cách sử dụng học sâu, hãy xem xét các tài nguyên sau:

1. "Tự động phát hiện đoạn ST trên điện tâm đồ: Ứng dụng trong chẩn đoán bệnh thiếu máu cục bộ" (ArXiv:1809.03452)
2. "Phân loại rối loạn nhịp tim ECG bằng mạng thần kinh chuyển đổi 2-D" (ArXiv:1804.06812)
3. "Phát hiện và phân loại rối loạn nhịp tim ở cấp độ bác sĩ tim mạch trong điện tâm đồ lưu động sử dụng mạng lưới thần kinh sâu" (Nature Medicine, 2019)

Những bài viết này cung cấp những hiểu biết sâu sắc có giá trị về các kỹ thuật và phương pháp tiên tiến trong phân tích ECG bằng cách sử dụng máy học.
