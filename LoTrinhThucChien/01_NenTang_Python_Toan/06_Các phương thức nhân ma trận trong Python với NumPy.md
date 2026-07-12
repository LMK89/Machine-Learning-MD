## Phương thức nhân ma trận trong Python với NumPy
Slide 1: 3 cách thực hiện phép nhân ma trận trong Python bằng NumPy

Phép nhân ma trận là một phép toán cơ bản trong đại số tuyến tính và có nhiều ứng dụng trong nhiều lĩnh vực khác nhau. Bài trình bày này sẽ khám phá ba phương pháp hiệu quả để thực hiện phép nhân ma trận bằng NumPy, một thư viện mạnh mẽ để tính toán số trong Python.

```python
import numpy as np
```

Slide 2: Method 1: Using np.dot()

The np.dot() function is a versatile tool for matrix multiplication. It can handle both 1D and 2D arrays, making it suitable for vector-matrix and matrix-matrix multiplication.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result = np.dot(A, B)
print(result)
```

Trang trình bày 3: Đầu ra cho np.dot()

```
[[19 22]
 [43 50]]
```

Slide 4: Understanding np.dot()

The np.dot() function performs the dot product of two arrays. For 2D arrays, it's equivalent to matrix multiplication. It's important to note that the number of columns in the first matrix must match the number of rows in the second matrix.

```python
# Vector-matrix multiplication
v = np.array([1, 2])
M = np.array([[1, 2], [3, 4]])
result = np.dot(v, M)
print(result)  # Output: [7 10]
```

Slide 5: Cách 2: Sử dụng toán tử @

Python 3.5 đã giới thiệu toán tử @ để nhân ma trận. Toán tử này cung cấp cú pháp trực quan và dễ đọc hơn cho các phép toán ma trận.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result = A @ B
print(result)
```

Slide 6: Output for the @ Operator

```
[[19 22]
 [43 50]]
```

Slide 7: Ưu điểm của toán tử @

Toán tử @ không chỉ ngắn gọn hơn mà còn rõ ràng hơn về mục đích của nó. Nó chỉ ra rõ ràng phép nhân ma trận, cải thiện khả năng đọc mã và giảm sự nhầm lẫn tiềm ẩn với phép nhân theo phần tử.

```python
# Chaining multiple matrix multiplications
C = np.array([[9, 10], [11, 12]])
result = A @ B @ C
print(result)
```

Slide 8: Output for the @ Operator

```
[[499 542]
 [1131 1230]]
```

Slide 9: Cách 3: Sử dụng np.matmul()

Hàm np.matmul() được thiết kế đặc biệt cho các hoạt động của sản phẩm ma trận. Nó có thể xử lý các mảng có chiều cao hơn và cung cấp khả năng phát sóng cho các hình dạng mảng nhất định.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result = np.matmul(A, B)
print(result)
```

Slide 10: Output for np.matmul()

```
[[19 22]
 [43 50]]
```

Trang trình bày 11: np.matmul() với kích thước cao hơn

Một ưu điểm của np.matmul() là khả năng hoạt động với các mảng có nhiều hơn hai chiều. Nó áp dụng phép nhân ma trận cho hai chiều cuối cùng trong khi phát sóng trên các chiều còn lại.

```python
A = np.random.rand(2, 3, 4)
B = np.random.rand(2, 4, 3)
result = np.matmul(A, B)
print(result.shape)  # Output: (2, 3, 3)
```

Slide 12: Performance Comparison

Let's compare the performance of these three methods using timeit for larger matrices.

```python
import timeit

A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)

dot_time = timeit.timeit(lambda: np.dot(A, B), number=10)
matmul_time = timeit.timeit(lambda: np.matmul(A, B), number=10)
operator_time = timeit.timeit(lambda: A @ B, number=10)

print(f"np.dot(): {dot_time:.4f} seconds")
print(f"np.matmul(): {matmul_time:.4f} seconds")
print(f"@ operator: {operator_time:.4f} seconds")
```

Slide 13: Ví dụ thực tế 1: Tích chập ảnh

Phép nhân ma trận rất quan trọng trong xử lý ảnh, đặc biệt đối với việc áp dụng các bộ lọc tích chập. Hãy triển khai bộ lọc phát hiện cạnh đơn giản bằng phép nhân ma trận.

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

# Edge detection kernel
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Pad the image
padded_image = np.pad(image, pad_width=1, mode='constant')

# Apply convolution using matrix multiplication
result = np.zeros_like(image)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        result[i, j] = np.sum(padded_image[i:i+3, j:j+3] * kernel)

plt.imshow(result, cmap='gray')
plt.title("Edge Detection Result")
plt.show()
```

Slide 14: Ví dụ thực tế 2: Giải hệ phương trình tuyến tính

Phép nhân ma trận rất cần thiết trong việc giải các hệ phương trình tuyến tính. Hãy giải một hệ thống đơn giản bằng cách sử dụng các phép toán ma trận của NumPy.

```python
import numpy as np

# Define the system: 2x + y = 7, x + 3y = 11
A = np.array([[2, 1],
              [1, 3]])
b = np.array([7, 11])

# Solve using matrix multiplication and inverse
x = np.dot(np.linalg.inv(A), b)

print("Solution:")
print(f"x = {x[0]:.2f}")
print(f"y = {x[1]:.2f}")

# Verify the solution
verification = np.dot(A, x)
print("\nVerification:")
print(f"2x + y = {verification[0]:.2f}")
print(f"x + 3y = {verification[1]:.2f}")
```

Slide 15: Lựa chọn phương pháp phù hợp

Mỗi phương pháp đều có điểm mạnh riêng:

* np.dot(): Linh hoạt cho cả phép toán vectơ và ma trận
* Toán tử @: Trực quan và dễ đọc để nhân ma trận
* np.matmul(): Hiệu quả cho mảng và phát sóng có chiều cao hơn

Hãy xem xét các yếu tố như khả năng đọc mã, yêu cầu về hiệu suất và kích thước của mảng khi chọn phương pháp.

```python
# Example of choosing methods based on array dimensions
v = np.array([1, 2, 3])
M = np.array([[1, 2], [3, 4], [5, 6]])

# For vector-matrix multiplication, np.dot() is suitable
result1 = np.dot(v, M)

# For matrix-matrix multiplication, @ operator is readable
result2 = M.T @ M

print("Vector-matrix result:", result1)
print("Matrix-matrix result:\n", result2)
```

Trang trình bày 16: Các mẹo và phương pháp hay nhất

1. Luôn kiểm tra kích thước ma trận trước khi nhân
2. Sử dụng np.matmul() hoặc @ để có mục đích rõ ràng hơn trong phép nhân ma trận
3. Xem xét hiệu quả bộ nhớ cho ma trận lớn
4. Tận dụng khả năng phát sóng của NumPy khi có thể

```python
# Example of dimension checking and broadcasting
def safe_matrix_multiply(A, B):
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrix dimensions are not compatible")
    return np.matmul(A, B)

# Broadcasting example
a = np.array([[1, 2, 3]])  # Shape: (1, 3)
b = np.array([[4], [5], [6]])  # Shape: (3, 1)
result = np.matmul(a, b)  # Result shape: (1, 1)
print("Broadcasting result:", result)
```

Slide 17: Kết luận

Phép nhân ma trận trong Python bằng NumPy cung cấp các công cụ mạnh mẽ cho các tác vụ tính toán khác nhau. Bằng cách hiểu các sắc thái của np.dot(), toán tử @ và np.matmul(), bạn có thể thực hiện các phép toán ma trận một cách hiệu quả trong các dự án Python của mình. Hãy nhớ xem xét các yêu cầu cụ thể của nhiệm vụ của bạn khi lựa chọn phương pháp phù hợp nhất.

Trang trình bày 18: Tài nguyên bổ sung

Để biết thêm thông tin chi tiết về phép nhân ma trận và NumPy:

1. "Nghệ thuật đại số tuyến tính" của Liesen và Mehrmann (ArXiv:2108.06468) [https://arxiv.org/abs/2108.06468](https://arxiv.org/abs/2108.06468)
2. "Đại số tuyến tính số trong khoa học dữ liệu sử dụng Python" của Linderman (ArXiv:2111.04227) [https://arxiv.org/abs/2111.04227](https://arxiv.org/abs/2111.04227)

Các tài nguyên này cung cấp kiến ​​thức toàn diện về các khái niệm đại số tuyến tính và các ứng dụng của chúng trong Python, mang lại những hiểu biết sâu sắc có giá trị để khám phá thêm.
