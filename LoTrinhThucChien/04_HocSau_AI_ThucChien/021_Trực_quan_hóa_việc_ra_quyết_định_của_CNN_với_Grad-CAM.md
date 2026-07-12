## Trực quan hóa việc ra quyết định của CNN với Grad-CAM
Slide 1: Giới thiệu về Grad-CAM

Grad-CAM (Bản đồ kích hoạt lớp có trọng số theo độ dốc) là một kỹ thuật mạnh mẽ để trực quan hóa và hiểu quá trình ra quyết định của mạng thần kinh tích chập (CNN). Nó giúp xác định vùng nào của hình ảnh đầu vào là quan trọng nhất đối với dự đoán của mô hình.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def grad_cam(model, img_array, layer_name, class_index):
    # Create a model that maps the input image to the activations
    # of the last convolutional layer and the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    # Rest of the Grad-CAM implementation will follow in subsequent slides
```

Slide 2: Chuẩn bị đầu vào

Trước khi áp dụng Grad-CAM, chúng ta cần chuẩn bị hình ảnh và mô hình đầu vào. Điều này liên quan đến việc tải và xử lý trước hình ảnh cũng như đảm bảo mô hình của chúng tôi sẵn sàng để suy luận.

```python
# Load and preprocess the image
img_path = 'path/to/your/image.jpg'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

# Load a pre-trained model (e.g., ResNet50)
model = tf.keras.applications.ResNet50(weights='imagenet')

# Choose the last convolutional layer
layer_name = 'conv5_block3_out'
```

Slide 3: Tính toán độ dốc

Cốt lõi của Grad-CAM liên quan đến việc tính toán độ dốc của đầu ra đối với các bản đồ đặc trưng của một lớp chập cụ thể. Điều này giúp chúng tôi hiểu những tính năng nào là quan trọng nhất để dự đoán.

```python
def compute_gradients(grad_model, img_array, class_index):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    return conv_outputs, grads

# Assume we're interested in the top predicted class
class_index = tf.argmax(model.predict(img_array)[0])
conv_outputs, grads = compute_gradients(grad_model, img_array, class_index)
```

Slide 4: Tính toán sơ đồ kích hoạt lớp

Sau khi có gradient, chúng ta có thể tính toán bản đồ kích hoạt lớp. Điều này liên quan đến việc lấy nhóm gradient trung bình toàn cầu và sử dụng nó để tính trọng số cho các bản đồ đặc trưng.

```python
def calculate_cam(conv_outputs, grads):
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    cam = tf.nn.relu(cam)  # ReLU to only show positive influences
    return cam

cam = calculate_cam(conv_outputs[0], grads[0])
```

Slide 5: Trực quan hóa Heatmap

Để làm cho bản đồ kích hoạt lớp có thể hiểu được, chúng ta cần thay đổi kích thước của nó để phù hợp với kích thước hình ảnh đầu vào và phủ nó lên hình ảnh gốc.

```python
def create_heatmap(cam, img):
    cam = cv2.resize(cam.numpy(), (img.shape[1], img.shape[0]))
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed = heatmap * 0.4 + img
    return superimposed / 255.0

heatmap = create_heatmap(cam, img)
plt.imshow(heatmap)
plt.axis('off')
plt.show()
```

Slide 6: Diễn giải kết quả Grad-CAM

Grad-CAM tạo ra bản đồ nhiệt làm nổi bật các vùng của hình ảnh đầu vào có ảnh hưởng mạnh nhất đến dự đoán của mô hình cho một lớp cụ thể. Vùng màu đỏ biểu thị tầm quan trọng cao, trong khi vùng màu xanh ít quan trọng hơn.

```python
def interpret_prediction(model, img_array, class_index):
    predictions = model.predict(img_array)
    predicted_class = tf.keras.applications.resnet50.decode_predictions(predictions, top=1)[0][0]
    class_name = predicted_class[1]
    confidence = predicted_class[2]

    print(f"Predicted class: {class_name}")
    print(f"Confidence: {confidence:.2f}")

interpret_prediction(model, img_array, class_index)
```

Trang trình bày 7: Ví dụ thực tế: Phát hiện đối tượng

Hãy áp dụng Grad-CAM cho một tình huống thực tế về phát hiện đối tượng trong hình ảnh đường phố trong thành phố.

```python
img_path = 'path/to/street_image.jpg'
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

class_index = tf.argmax(model.predict(img_array)[0])
conv_outputs, grads = compute_gradients(grad_model, img_array, class_index)
cam = calculate_cam(conv_outputs[0], grads[0])
heatmap = create_heatmap(cam, img)

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(heatmap)
plt.title('Grad-CAM Heatmap')
plt.axis('off')

plt.show()

interpret_prediction(model, img_array, class_index)
```

Trang trình bày 8: Tìm hiểu trọng tâm của mô hình

Bằng cách kiểm tra bản đồ nhiệt Grad-CAM, chúng ta có thể biết mô hình đang tập trung vào phần nào của hình ảnh để đưa ra dự đoán. Điều này có thể giúp chúng tôi biết liệu mô hình có đang sử dụng các tính năng liên quan hay không hoặc liệu nó có bị ảnh hưởng bởi các yếu tố nền không liên quan hay không.

```python
def analyze_focus(heatmap, threshold=0.5):
    high_focus = np.mean(heatmap > threshold)
    print(f"Percentage of image with high focus: {high_focus:.2%}")

    if high_focus > 0.7:
        print("The model is focusing on a large portion of the image.")
    elif high_focus > 0.3:
        print("The model is focusing on specific regions of the image.")
    else:
        print("The model is highly focused on small, specific areas.")

analyze_focus(heatmap)
```

Slide 9: So sánh nhiều lớp

Grad-CAM có thể được sử dụng để so sánh cách mô hình tập trung vào các lớp khác nhau trong cùng một hình ảnh. Điều này đặc biệt hữu ích để hiểu các vấn đề phân loại nhiều lớp.

```python
def compare_classes(model, img_array, class_indices):
    fig, axes = plt.subplots(1, len(class_indices), figsize=(15, 5))

    for i, class_index in enumerate(class_indices):
        conv_outputs, grads = compute_gradients(grad_model, img_array, class_index)
        cam = calculate_cam(conv_outputs[0], grads[0])
        heatmap = create_heatmap(cam, img)

        axes[i].imshow(heatmap)
        axes[i].set_title(f"Class {class_index}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

top_3_classes = tf.argsort(model.predict(img_array)[0])[-3:]
compare_classes(model, img_array, top_3_classes)
```

Trang trình bày 10: Grad-CAM để gỡ lỗi mô hình

Grad-CAM có thể là một công cụ mạnh mẽ để gỡ lỗi và cải thiện mạng lưới thần kinh. Bằng cách hình dung những gì mô hình đang tập trung vào, chúng ta có thể xác định những thành kiến ​​hoặc sai lầm tiềm ẩn trong quá trình ra quyết định của mô hình.

```python
def debug_model(model, img_array, expected_class):
    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions[0])

    if predicted_class != expected_class:
        print("Model prediction doesn't match expected class.")
        print("Analyzing model focus...")

        conv_outputs, grads = compute_gradients(grad_model, img_array, predicted_class)
        cam = calculate_cam(conv_outputs[0], grads[0])
        heatmap = create_heatmap(cam, img)

        plt.imshow(heatmap)
        plt.title(f"Focus for predicted class {predicted_class}")
        plt.axis('off')
        plt.show()

        print("Check if the model is focusing on relevant features.")
    else:
        print("Model prediction matches expected class.")

debug_model(model, img_array, expected_class=242)  # 242 is the class index for 'bull mastiff' in ImageNet
```

Trang trình bày 11: Grad-CAM cho các kiến ​​trúc mạng khác nhau

Grad-CAM có thể được áp dụng cho nhiều kiến ​​trúc CNN khác nhau. Đây là ví dụ về cách sử dụng nó với kiểu máy khác, chẳng hạn như VGG16.

```python
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

vgg_model = VGG16(weights='imagenet')
vgg_layer_name = 'block5_conv3'

vgg_grad_model = tf.keras.models.Model(
    [vgg_model.inputs], [vgg_model.get_layer(vgg_layer_name).output, vgg_model.output]
)

img_array = preprocess_input(img_array)
class_index = tf.argmax(vgg_model.predict(img_array)[0])

conv_outputs, grads = compute_gradients(vgg_grad_model, img_array, class_index)
cam = calculate_cam(conv_outputs[0], grads[0])
heatmap = create_heatmap(cam, img)

plt.imshow(heatmap)
plt.title("Grad-CAM on VGG16")
plt.axis('off')
plt.show()

print(decode_predictions(vgg_model.predict(img_array), top=1)[0])
```

Trang trình bày 12: Ví dụ thực tế: Hình ảnh y tế

Hãy áp dụng Grad-CAM cho một tình huống chụp ảnh y tế, chẳng hạn như xác định bệnh viêm phổi trên phim X-quang ngực. Ví dụ này minh họa cách Grad-CAM có thể được sử dụng để nâng cao khả năng diễn giải trong các ứng dụng quan trọng như chăm sóc sức khỏe.

```python
# Assume we have a pre-trained model for pneumonia detection
pneumonia_model = tf.keras.models.load_model('path/to/pneumonia_model.h5')

# Load and preprocess a chest X-ray image
xray_path = 'path/to/chest_xray.jpg'
xray = tf.keras.preprocessing.image.load_img(xray_path, target_size=(224, 224))
xray_array = tf.keras.preprocessing.image.img_to_array(xray)
xray_array = np.expand_dims(xray_array, axis=0)

# Apply Grad-CAM
pneumonia_grad_model = tf.keras.models.Model(
    [pneumonia_model.inputs],
    [pneumonia_model.get_layer('conv5_block3_out').output, pneumonia_model.output]
)

class_index = 1  # Assume 1 represents pneumonia
conv_outputs, grads = compute_gradients(pneumonia_grad_model, xray_array, class_index)
cam = calculate_cam(conv_outputs[0], grads[0])
heatmap = create_heatmap(cam, xray)

plt.imshow(heatmap, cmap='gray')
plt.title("Pneumonia Detection Heatmap")
plt.axis('off')
plt.show()

prediction = pneumonia_model.predict(xray_array)[0][0]
print(f"Probability of pneumonia: {prediction:.2%}")
```

Trang trình bày 13: Hạn chế và cân nhắc

Mặc dù Grad-CAM là một công cụ mạnh mẽ nhưng điều quan trọng là phải nhận thức được những hạn chế của nó:

1. Nó chỉ hoạt động với CNN và có thể không phù hợp với các kiến ​​trúc khác.
2. Độ phân giải của bản đồ nhiệt bị giới hạn bởi kích thước của bản đồ đặc trưng trong lớp đã chọn.
3. Nó có thể không nắm bắt được các chi tiết chi tiết hoặc mối quan hệ phức tạp giữa các tính năng.
4. Việc lựa chọn lớp chập có thể ảnh hưởng đáng kể đến kết quả.

```python
def grad_cam_resolution_demo(model, img_array, layer_names):
    fig, axes = plt.subplots(1, len(layer_names), figsize=(15, 5))

    for i, layer_name in enumerate(layer_names):
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )

        conv_outputs, grads = compute_gradients(grad_model, img_array, class_index)
        cam = calculate_cam(conv_outputs[0], grads[0])
        heatmap = create_heatmap(cam, img)

        axes[i].imshow(heatmap)
        axes[i].set_title(f"Layer: {layer_name}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

layer_names = ['conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
grad_cam_resolution_demo(model, img_array, layer_names)
```

Trang trình bày 14: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về Grad-CAM và các kỹ thuật liên quan, đây là một số tài nguyên có giá trị:

1. Bài viết gốc về Grad-CAM: "Grad-CAM: Giải thích trực quan từ Mạng sâu thông qua bản địa hóa dựa trên gradient" của Selvaraju et al. (2017) Liên kết ArXiv: [https://arxiv.org/abs/1610.02391](https://arxiv.org/abs/1610.02391)
2. "Phân bổ tiên đề cho mạng sâu" của Sundararajan và cộng sự. (2017) Liên kết ArXiv: [https://arxiv.org/abs/1703.01365](https://arxiv.org/abs/1703.01365)
3. "Kiểm tra độ chính xác cho Bản đồ vị trí nổi bật" của Adebayo và cộng sự. (2018) Liên kết ArXiv: [https://arxiv.org/abs/1810.03292](https://arxiv.org/abs/1810.03292)
4. "Grad-CAM++: Giải thích trực quan được cải thiện cho Mạng kết hợp sâu" của Chattopadhyay et al. (2018) Liên kết ArXiv: [https://arxiv.org/abs/1710.11063](https://arxiv.org/abs/1710.11063)

Các bài viết này cung cấp các cuộc thảo luận chuyên sâu về nền tảng lý thuyết, cải tiến và đánh giá về Grad-CAM cũng như các kỹ thuật trực quan hóa liên quan cho các mô hình học sâu.