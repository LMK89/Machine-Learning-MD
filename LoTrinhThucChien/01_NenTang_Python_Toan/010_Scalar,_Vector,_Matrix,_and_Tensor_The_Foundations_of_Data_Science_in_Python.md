## Vô hướng, Vector, Ma trận và Tensor Nền tảng của Khoa học Dữ liệu trong Python
Slide 1: Bộ tứ khoa học dữ liệu: Vô hướng, Vector, Ma trận và Tensor

Các cấu trúc toán học cơ bản này tạo thành xương sống của khoa học dữ liệu hiện đại, cho phép thực hiện các phép tính và biểu diễn phức tạp bằng Python. Chúng ta sẽ khám phá từng khái niệm, mối quan hệ của chúng và các ứng dụng thực tế trong phân tích dữ liệu và học máy.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple visualization of the quartet
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].text(0.5, 0.5, 'Scalar', ha='center', va='center', fontsize=20)
axs[0, 1].plot([1, 2, 3], [4, 5, 6])
axs[0, 1].set_title('Vector')
axs[1, 0].imshow(np.random.rand(5, 5), cmap='viridis')
axs[1, 0].set_title('Matrix')
axs[1, 1].text(0.5, 0.5, 'Tensor', ha='center', va='center', fontsize=20)
plt.tight_layout()
plt.show()
```

Trang trình bày 2: Vô hướng: Khối xây dựng

Đại lượng vô hướng là một giá trị số duy nhất, biểu thị độ lớn không có hướng. Trong Python, vô hướng thường được biểu diễn bằng các kiểu số đơn giản như số nguyên hoặc số float. Chúng tạo thành nền tảng cho các cấu trúc dữ liệu phức tạp hơn.

```python
# Scalar examples
temperature = 25.5  # Temperature in Celsius
count = 100  # Number of items

# Basic operations with scalars
fahrenheit = (temperature * 9/5) + 32
double_count = count * 2

print(f"Temperature: {temperature}°C = {fahrenheit}°F")
print(f"Count: {count}, Doubled: {double_count}")
```

Slide 3: Vector: Mảng một chiều

Vectơ là mảng một chiều của các đại lượng vô hướng, biểu diễn các đại lượng có cả độ lớn và hướng. Trong Python, chúng ta thường sử dụng mảng NumPy để làm việc với vectơ một cách hiệu quả.

```python
import numpy as np

# Create a vector
v = np.array([1, 2, 3, 4, 5])

# Basic vector operations
magnitude = np.linalg.norm(v)
normalized = v / magnitude

print(f"Vector: {v}")
print(f"Magnitude: {magnitude:.2f}")
print(f"Normalized: {normalized}")

# Dot product of two vectors
u = np.array([2, 3, 4, 5, 6])
dot_product = np.dot(v, u)
print(f"Dot product of {v} and {u}: {dot_product}")
```

Slide 4: Ma trận: Mảng hai chiều

Ma trận là mảng hai chiều của các số vô hướng, được sắp xếp theo hàng và cột. Chúng là nền tảng của đại số tuyến tính và là nền tảng cho nhiều thuật toán khoa học dữ liệu.

```python
import numpy as np

# Create a matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Matrix operations
B = np.array([[9, 8, 7],
              [6, 5, 4],
              [3, 2, 1]])

# Matrix addition
C = A + B

# Matrix multiplication
D = np.dot(A, B)

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)
print("\nA + B:")
print(C)
print("\nA · B:")
print(D)
```

Slide 5: Tensor: Mảng đa chiều

Tensors là sự khái quát hóa của vectơ và ma trận lên các chiều cao hơn. Chúng rất quan trọng trong việc học sâu và biểu diễn dữ liệu phức tạp. Trong Python, chúng ta có thể sử dụng NumPy hoặc các thư viện chuyên dụng như TensorFlow hoặc PyTorch để làm việc với tensor.

```python
import numpy as np

# Create a 3D tensor (3x3x3)
T = np.array([[[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]],
              [[10, 11, 12],
               [13, 14, 15],
               [16, 17, 18]],
              [[19, 20, 21],
               [22, 23, 24],
               [25, 26, 27]]])

print("3D Tensor shape:", T.shape)
print("First 2D slice of the tensor:")
print(T[0])

# Tensor operations
sum_along_axis0 = np.sum(T, axis=0)
print("\nSum along axis 0:")
print(sum_along_axis0)
```

Trang trình bày 6: Các phép tính vô hướng: Ngoài số học cơ bản

Vô hướng trong Python có thể được sử dụng trong nhiều phép toán khác nhau, bao gồm lượng giác, lũy thừa và logarit. Những hoạt động này rất cần thiết trong nhiều ứng dụng khoa học và kỹ thuật.

```python
import math

angle_degrees = 45
angle_radians = math.radians(angle_degrees)

# Trigonometric functions
sin_value = math.sin(angle_radians)
cos_value = math.cos(angle_radians)

# Exponentiation and logarithms
base = 2
exponent = 3
power_result = math.pow(base, exponent)
log_result = math.log(power_result, base)

print(f"Sin({angle_degrees}°) = {sin_value:.4f}")
print(f"Cos({angle_degrees}°) = {cos_value:.4f}")
print(f"{base}^{exponent} = {power_result}")
print(f"log_{base}({power_result}) = {log_result}")
```

Slide 7: Các phép toán vectơ: Các phép biến đổi hình học

Vector là công cụ mạnh mẽ để biểu diễn và xử lý dữ liệu hình học. Chúng ta có thể sử dụng chúng để thực hiện các thao tác dịch, xoay và chia tỷ lệ trong không gian 2D và 3D.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a 2D vector
v = np.array([3, 2])

# Translation
translation = np.array([2, 1])
v_translated = v + translation

# Rotation (45 degrees counterclockwise)
theta = np.radians(45)
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
v_rotated = np.dot(rotation_matrix, v)

# Scaling
scale_factor = 2
v_scaled = v * scale_factor

# Plotting
plt.figure(figsize=(10, 10))
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='r', label='Original')
plt.quiver(0, 0, v_translated[0], v_translated[1], angles='xy', scale_units='xy', scale=1, color='g', label='Translated')
plt.quiver(0, 0, v_rotated[0], v_rotated[1], angles='xy', scale_units='xy', scale=1, color='b', label='Rotated')
plt.quiver(0, 0, v_scaled[0], v_scaled[1], angles='xy', scale_units='xy', scale=1, color='m', label='Scaled')
plt.xlim(-1, 8)
plt.ylim(-1, 8)
plt.legend()
plt.grid(True)
plt.show()
```

Slide 8: Ứng dụng ma trận: Xử lý ảnh

Ma trận được sử dụng rộng rãi trong xử lý ảnh. Chúng ta có thể biểu diễn hình ảnh dưới dạng ma trận 2D và áp dụng các phép biến đổi khác nhau để thao tác với chúng.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 5x5 grayscale image
image = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                  [0.2, 0.3, 0.4, 0.5, 0.6],
                  [0.3, 0.4, 0.5, 0.6, 0.7],
                  [0.4, 0.5, 0.6, 0.7, 0.8],
                  [0.5, 0.6, 0.7, 0.8, 0.9]])

# Define a blur kernel
blur_kernel = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]]) / 16

# Apply convolution for blurring
blurred_image = np.zeros_like(image)
for i in range(1, image.shape[0]-1):
    for j in range(1, image.shape[1]-1):
        blurred_image[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * blur_kernel)

# Plot original and blurred images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title('Original Image')
ax2.imshow(blurred_image, cmap='gray')
ax2.set_title('Blurred Image')
plt.show()
```

Slide 9: Các thao tác với tensor: Xử lý ảnh màu

Tensor cho phép chúng ta làm việc với dữ liệu đa chiều, chẳng hạn như hình ảnh màu. Chúng ta có thể sử dụng tensor 3D để biểu diễn và xử lý hình ảnh RGB.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 5x5x3 RGB image
rgb_image = np.zeros((5, 5, 3))
rgb_image[:, :, 0] = np.linspace(0, 1, 25).reshape(5, 5)  # Red channel
rgb_image[:, :, 1] = np.linspace(0, 1, 25).reshape(5, 5)[::-1]  # Green channel
rgb_image[:, :, 2] = np.linspace(0, 1, 25).reshape(5, 5).T  # Blue channel

# Increase brightness
brightened_image = np.clip(rgb_image * 1.5, 0, 1)

# Convert to grayscale
grayscale_image = np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140])

# Plot images
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.imshow(rgb_image)
ax1.set_title('Original RGB Image')
ax2.imshow(brightened_image)
ax2.set_title('Brightened Image')
ax3.imshow(grayscale_image, cmap='gray')
ax3.set_title('Grayscale Image')
plt.show()
```

Trang trình bày 10: Ví dụ thực tế: Phân tích dữ liệu thời tiết

Vô hướng, vectơ và ma trận có thể được sử dụng để phân tích dữ liệu thời tiết. Chúng tôi sẽ trình bày cách làm việc với dữ liệu nhiệt độ cho nhiều thành phố theo thời gian.

```python
import numpy as np
import matplotlib.pyplot as plt

# Weather data: temperatures for 5 cities over 7 days
weather_data = np.array([
    [20, 22, 23, 19, 21, 24, 22],  # City 1
    [18, 20, 19, 21, 23, 24, 22],  # City 2
    [22, 21, 23, 24, 25, 23, 21],  # City 3
    [17, 18, 20, 21, 22, 20, 19],  # City 4
    [19, 20, 22, 23, 21, 20, 18]   # City 5
])

# Calculate average temperature for each city
avg_temp = np.mean(weather_data, axis=1)

# Find the hottest day for each city
hottest_day = np.argmax(weather_data, axis=1)

# Plot the data
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(weather_data[i], label=f'City {i+1}')
plt.xlabel('Day')
plt.ylabel('Temperature (°C)')
plt.title('Weekly Temperature Data for 5 Cities')
plt.legend()
plt.grid(True)
plt.show()

print("Average temperatures:")
for i, temp in enumerate(avg_temp):
    print(f"City {i+1}: {temp:.1f}°C")

print("\nHottest day for each city:")
for i, day in enumerate(hottest_day):
    print(f"City {i+1}: Day {day+1}")
```

Slide 11: Ví dụ thực tế: Nén ảnh bằng SVD

Phân tách giá trị số ít (SVD) là một kỹ thuật phân tích hệ số ma trận có thể được sử dụng để nén hình ảnh. Chúng tôi sẽ trình bày cách sử dụng SVD để nén hình ảnh thang độ xám.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image

# Load a sample image and convert to grayscale
image = load_sample_image("china.jpg")
gray_image = np.mean(image, axis=2).astype(np.float64)

# Perform SVD
U, s, Vt = np.linalg.svd(gray_image, full_matrices=False)

# Function to reconstruct image with k singular values
def reconstruct_image(U, s, Vt, k):
    return np.matrix(U[:, :k]) * np.diag(s[:k]) * np.matrix(Vt[:k, :])

# Reconstruct images with different numbers of singular values
k_values = [5, 20, 50, 200]
reconstructed_images = [reconstruct_image(U, s, Vt, k) for k in k_values]

# Plot original and reconstructed images
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(gray_image, cmap='gray')
axes[0, 0].set_title("Original Image")

for i, (k, img) in enumerate(zip(k_values, reconstructed_images), 1):
    ax = axes[i // 3, i % 3]
    ax.imshow(img, cmap='gray')
    ax.set_title(f"k = {k}")
    compression_ratio = (k * (U.shape[0] + Vt.shape[1] + 1)) / (U.shape[0] * Vt.shape[1])
    ax.set_xlabel(f"Compression Ratio: {compression_ratio:.2%}")

plt.tight_layout()
plt.show()
```

Trang trình bày 12: Tensors trong Machine Learning: Mạng nơ-ron

Tensors là nền tảng trong học sâu, đặc biệt là trong mạng lưới thần kinh. Chúng ta sẽ tạo một mạng lưới thần kinh chuyển tiếp đơn giản để minh họa cách sử dụng tensor trong bối cảnh này.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feedforward(input_data, weights, biases):
    # Input layer to hidden layer
    hidden = sigmoid(np.dot(input_data, weights[0]) + biases[0])
    # Hidden layer to output layer
    output = sigmoid(np.dot(hidden, weights[1]) + biases[1])
    return output

# Define network architecture
input_size = 3
hidden_size = 4
output_size = 2

# Initialize random weights and biases
np.random.seed(42)
weights = [
    np.random.randn(input_size, hidden_size),
    np.random.randn(hidden_size, output_size)
]
biases = [
    np.random.randn(hidden_size),
    np.random.randn(output_size)
]

# Create a batch of input data
batch_size = 5
input_data = np.random.randn(batch_size, input_size)

# Perform feedforward pass
output = feedforward(input_data, weights, biases)

print("Input shape:", input_data.shape)
print("Output shape:", output.shape)
print("\nSample input:")
print(input_data[0])
print("\nCorresponding output:")
print(output[0])
```

Trang trình bày 13: Sức mạnh của bộ tứ trong khoa học dữ liệu

Sự tương tác giữa các đại số vô hướng, vectơ, ma trận và tensor tạo thành nền tảng của nhiều thuật toán khoa học dữ liệu. Sức mạnh tổng hợp này cho phép thực hiện các phép tính và biểu diễn phức tạp rất quan trọng đối với phân tích nâng cao và học máy.

```python
import numpy as np

# Scalar: Simple statistic
data = np.array([1, 2, 3, 4, 5])
mean = np.mean(data)

# Vector: Feature representation
features = np.array([height, weight, age])

# Matrix: Dataset of features
dataset = np.array([
    [170, 70, 30],
    [165, 65, 25],
    [180, 80, 35]
])

# Tensor: Time series of image data
image_series = np.random.rand(10, 64, 64, 3)  # 10 RGB images of 64x64 pixels

print(f"Scalar (mean): {mean}")
print(f"Vector (features): {features}")
print("Matrix (dataset):")
print(dataset)
print(f"Tensor (image series) shape: {image_series.shape}")
```

Slide 14: Ứng dụng thực tế: Phân tích thành phần chính (PCA)

PCA là một kỹ thuật giảm kích thước tận dụng sức mạnh của ma trận và phân tách giá trị riêng. Nó được sử dụng rộng rãi trong tiền xử lý dữ liệu và trích xuất tính năng.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Generate sample data
np.random.seed(42)
n_samples = 300
X = np.dot(np.random.randn(n_samples, 2), [[2, 1], [1, 3]])

# Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot original and transformed data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[:, 0], X[:, 1], alpha=0.7)
ax1.set_title('Original Data')

ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
ax2.set_title('PCA Transformed Data')

plt.tight_layout()
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)
```

Slide 15: Định hướng tương lai và các chủ đề nâng cao

Bộ tứ tính toán vô hướng, vectơ, ma trận và tenxơ tiếp tục phát triển, thúc đẩy những đổi mới trong khoa học dữ liệu và học máy. Một số chủ đề nâng cao bao gồm:

1. Mạng Tensor: Được sử dụng trong điện toán lượng tử và mô hình hóa hệ thống phức tạp
2. Hình học vi phân: Ứng dụng phép tính tensor vào machine learning
3. Tensor lượng tử: Biểu diễn các trạng thái và hoạt động lượng tử
4. Tensor Decompositions: Các kỹ thuật nâng cao để phân tích dữ liệu đa chiều

Các chủ đề này giới thiệu quá trình nghiên cứu và phát triển đang diễn ra trong việc tận dụng các cấu trúc toán học này cho các ứng dụng tiên tiến trong khoa học dữ liệu và hơn thế nữa.

Trang trình bày 16: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về các chủ đề này, đây là một số tài nguyên có giá trị:

1. Bài viết của ArXiv về Mạng Tensor: "Mạng Tensor cho các vấn đề phân tích dữ liệu lớn và tối ưu hóa quy mô lớn" (arXiv:1407.3124)
2. Bài viết của ArXiv về Hình học vi phân trong học máy: "Hình học Riemannian trong học máy" (arXiv:2011.01538)
3. Bài viết của ArXiv về Tensor lượng tử: "Mạng Tensor lượng tử: Con đường dẫn đến học máy" (arXiv:1803.11537)
4. Bài viết của ArXiv về Phân rã Tensor: "Ứng dụng và Phân hủy Tensor" (arXiv:0905.0454)

Các bài viết này cung cấp các cuộc thảo luận chuyên sâu về các ứng dụng tiên tiến của tensor và các khái niệm liên quan trong các lĩnh vực khoa học dữ liệu và điện toán lượng tử khác nhau.
