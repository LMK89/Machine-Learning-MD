## NumPy Broadcasting Đơn giản hóa mã Python
Trang trình bày 1: Phát sóng NumPy là gì?

Phát sóng NumPy là một cơ chế mạnh mẽ cho phép sử dụng các mảng có hình dạng khác nhau trong các phép tính số học. Nó tự động mở rộng các mảng thành các hình dạng tương thích, cho phép tính toán hiệu quả và ngắn gọn mà không cần định hình lại hoặc nhập dữ liệu một cách rõ ràng.

```python
import numpy as np

# Broadcasting example
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])

result = a + b
print(result)
```

Slide 2: Khái niệm cơ bản về phát thanh truyền hình

Việc phát sóng tuân theo một bộ quy tắc để xác định cách kết hợp các mảng có hình dạng khác nhau. Nó bắt đầu với các thứ nguyên ở cuối và tiến dần về phía trước, so sánh kích thước của từng thứ nguyên.

```python
import numpy as np

# 1D array broadcasting with a scalar
arr = np.array([1, 2, 3, 4])
result = arr * 2
print(result)  # Output: [2 4 6 8]

# 2D array broadcasting with a 1D array
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([10, 20, 30])
result = matrix + vector
print(result)
```

Slide 3: Quy tắc phát sóng

1. Mảng có ít kích thước hơn được đệm bằng các mảng ở bên trái.
2. Kích thước Size-1 được kéo dài để phù hợp với hình dạng của mảng khác.
3. Nếu các mảng có hình dạng tương thích, việc phát sóng sẽ tiếp tục.

```python
import numpy as np

# Demonstrating rule 1: Padding with ones
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])
print(a.shape, b.shape)
result = a + b
print(result.shape)

# Demonstrating rule 2: Stretching size-1 dimensions
x = np.ones((3, 1))
y = np.arange(4)
print(x.shape, y.shape)
result = x + y
print(result.shape)
print(result)
```

Trang trình bày 4: Phát sóng trong hành động: Hoạt động theo từng phần tử

Việc phát sóng cho phép thực hiện các hoạt động hiệu quả theo từng phần tử giữa các mảng có hình dạng khác nhau, loại bỏ sự cần thiết của các vòng lặp rõ ràng.

```python
import numpy as np

# Element-wise multiplication with broadcasting
temperatures = np.array([20, 25, 30, 35])  # Celsius
conversion_factor = np.array([1.8])  # For Celsius to Fahrenheit
offset = np.array([32])

fahrenheit = temperatures * conversion_factor + offset
print(f"Celsius: {temperatures}")
print(f"Fahrenheit: {fahrenheit}")
```

Trang trình bày 5: Phát sóng với kích thước cao hơn

Truyền phát có thể hoạt động với các mảng có số lượng kích thước bất kỳ, giúp nó trở nên mạnh mẽ trong việc xử lý dữ liệu đa chiều.

```python
import numpy as np

# 3D array broadcasting
cube = np.arange(24).reshape(2, 3, 4)
plane = np.arange(12).reshape(3, 4)

result = cube + plane
print("Cube shape:", cube.shape)
print("Plane shape:", plane.shape)
print("Result shape:", result.shape)
print(result)
```

Trang trình bày 6: Ví dụ thực tế: Xử lý hình ảnh

Phát sóng đặc biệt hữu ích trong các tác vụ xử lý hình ảnh, chẳng hạn như điều chỉnh độ sáng hoặc áp dụng các bộ lọc.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a sample grayscale image
image = np.random.rand(5, 5)

# Increase brightness using broadcasting
brightness_factor = 1.5
brightened_image = image * brightness_factor

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(122)
plt.imshow(brightened_image, cmap='gray')
plt.title('Brightened Image')
plt.tight_layout()
plt.show()
```

Trang trình bày 7: Ví dụ thực tế: Phân tích dữ liệu thời tiết

Việc phát sóng giúp đơn giản hóa các thao tác trên dữ liệu thời tiết đa chiều, chẳng hạn như tính toán các dị thường về nhiệt độ.

```python
import numpy as np

# Sample temperature data (3D: year, month, city)
temperatures = np.random.rand(5, 12, 3) * 30  # 5 years, 12 months, 3 cities

# Calculate monthly averages across years
monthly_averages = np.mean(temperatures, axis=0)

# Calculate temperature anomalies
anomalies = temperatures - monthly_averages[np.newaxis, :, :]

print("Temperature anomalies shape:", anomalies.shape)
print("Sample anomaly for year 1, month 1, city 1:", anomalies[0, 0, 0])
```

Trang trình bày 8: Đơn giản hóa mã với phát sóng

Việc phát sóng có thể đơn giản hóa đáng kể mã của bạn bằng cách giảm nhu cầu về các vòng lặp rõ ràng và mảng tạm thời.

```python
import numpy as np
import time

# Without broadcasting
def without_broadcasting(arr1, arr2):
    result = np.zeros_like(arr1)
    for i in range(arr1.shape[0]):
        for j in range(arr1.shape[1]):
            result[i, j] = arr1[i, j] + arr2[j]
    return result

# With broadcasting
def with_broadcasting(arr1, arr2):
    return arr1 + arr2

# Compare performance
arr1 = np.random.rand(1000, 1000)
arr2 = np.random.rand(1000)

start = time.time()
result1 = without_broadcasting(arr1, arr2)
end = time.time()
print(f"Without broadcasting: {end - start:.5f} seconds")

start = time.time()
result2 = with_broadcasting(arr1, arr2)
end = time.time()
print(f"With broadcasting: {end - start:.5f} seconds")

print("Results are equal:", np.allclose(result1, result2))
```

Trang trình bày 9: Cạm bẫy khi phát sóng: Hình dạng không khớp

Mặc dù mạnh mẽ nhưng việc phát sóng có thể dẫn đến lỗi nếu hình dạng mảng không tương thích. Hiểu những lỗi này là rất quan trọng để sử dụng hiệu quả việc phát sóng.

```python
import numpy as np

try:
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([1, 2])
    result = a + b
except ValueError as e:
    print("Error:", str(e))

# Correcting the shape mismatch
b_corrected = np.array([[1], [2]])
result = a + b_corrected
print("Corrected result:\n", result)
```

Trang trình bày 10: Phát sóng nâng cao: Trục tùy chỉnh

NumPy cho phép chỉ định các trục tùy chỉnh để phát sóng, cung cấp nhiều quyền kiểm soát hơn về cách kết hợp các mảng.

```python
import numpy as np

# Create sample data
data = np.random.rand(2, 3, 4)
weights = np.random.rand(3)

# Broadcasting with a specified axis
weighted_data = data * weights[:, np.newaxis]

print("Data shape:", data.shape)
print("Weights shape:", weights.shape)
print("Weighted data shape:", weighted_data.shape)
print("Weighted data:\n", weighted_data)
```

Slide 11: Phát sóng trong các phép toán đại số tuyến tính

Truyền phát đặc biệt hữu ích trong các phép toán đại số tuyến tính, đơn giản hóa việc tính toán ma trận-vectơ.

```python
import numpy as np

# Matrix-vector multiplication using broadcasting
matrix = np.array([[1, 2, 3], [4, 5, 6]])
vector = np.array([2, 3, 4])

# Traditional approach
result1 = np.dot(matrix, vector)

# Using broadcasting
result2 = np.sum(matrix * vector, axis=1)

print("Traditional result:", result1)
print("Broadcasting result:", result2)
print("Results are equal:", np.allclose(result1, result2))
```

Trang trình bày 12: Tối ưu hóa việc sử dụng bộ nhớ với tính năng phát sóng

Việc phát sóng có thể giúp tối ưu hóa việc sử dụng bộ nhớ bằng cách tránh việc sao chép và phân bổ mảng không cần thiết.

```python
import numpy as np
import memory_profiler

@memory_profiler.profile
def without_broadcasting():
    x = np.random.rand(1000, 1000)
    y = np.random.rand(1000, 1000)
    return x + y

@memory_profiler.profile
def with_broadcasting():
    x = np.random.rand(1000, 1000)
    y = np.random.rand(1000)  # Only 1D array
    return x + y[:, np.newaxis]

print("Memory usage without broadcasting:")
without_broadcasting()

print("\nMemory usage with broadcasting:")
with_broadcasting()
```

Trang trình bày 13: Gỡ lỗi các vấn đề phát sóng

Khi làm việc với các hình dạng mảng phức tạp, có thể hữu ích khi sử dụng hàm `broadcast_arrays` của NumPy để trực quan hóa cách các mảng sẽ được phát cùng nhau.

```python
import numpy as np

def debug_broadcasting(arr1, arr2):
    try:
        broadcasted = np.broadcast_arrays(arr1, arr2)
        print("Broadcasted shapes:", [b.shape for b in broadcasted])
        return np.add(arr1, arr2)
    except ValueError as e:
        print("Broadcasting error:", str(e))
        return None

# Example 1: Compatible shapes
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])
result = debug_broadcasting(a, b)
print("Result 1:", result)

# Example 2: Incompatible shapes
c = np.array([[1, 2], [3, 4]])
d = np.array([1, 2, 3])
result = debug_broadcasting(c, d)
```

Slide 14: Phát sóng trong trực quan hóa dữ liệu

Việc phát sóng có thể đơn giản hóa việc chuẩn bị dữ liệu cho các tác vụ trực quan hóa, chẳng hạn như tạo dải màu hoặc bản đồ nhiệt.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a color gradient using broadcasting
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)

# Create RGB values using broadcasting
R = X
G = Y
B = 1 - X

# Combine RGB channels
color_gradient = np.dstack((R, G, B))

plt.figure(figsize=(8, 6))
plt.imshow(color_gradient)
plt.title('Color Gradient using Broadcasting')
plt.axis('off')
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Để biết thêm thông tin về phát sóng NumPy và các ứng dụng của nó, hãy xem xét khám phá các tài nguyên sau:

1. Tài liệu chính thức của NumPy về phát sóng: [https://numpy.org/doc/stable/user/basics.broadcasting.html](https://numpy.org/doc/stable/user/basics.broadcasting.html)
2. "Hoạt động được vector hóa và phát sóng trong NumPy" của Jake VanderPlas: arXiv:1411.5038
3. "Giới thiệu nhẹ nhàng về phát sóng trong mảng NumPy" của Jason Brownlee: [https://machinelearningmastery.com/broadcasting-with-numpy-arrays/](https://machinelearningmastery.com/broadcasting-with-numpy-arrays/)

Các tài nguyên này cung cấp những giải thích sâu sắc, các kỹ thuật nâng cao và các ví dụ thực tế để nâng cao hơn nữa sự hiểu biết của bạn về phát sóng NumPy.
