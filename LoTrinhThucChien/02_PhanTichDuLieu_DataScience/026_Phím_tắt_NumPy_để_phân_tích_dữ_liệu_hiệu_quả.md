## Phím tắt NumPy để phân tích hiệu quả dữ liệu
Trang trình bày 1: NumPy Essentials: Lối tắt để phân tích dữ liệu hiệu quả

NumPy là một thư viện cơ sở dữ liệu tính toán khoa học bằng Python. Hướng dẫn này sẽ hướng dẫn bạn các lệnh và thao tác NumPy cần thiết, giúp bạn hợp lý hóa dữ liệu phân tích trình phân tích của mình. Hãy cùng đi sâu vào một số ví dụ thực tế và đoạn mã.

```python
import numpy as np

# Create a simple array
arr = np.array([1, 2, 3, 4, 5])
print(arr)
# Output: [1 2 3 4 5]
```

Trang trình bày 2: Tạo mảng: xây dựng khối NumPy

NumPy cung cấp nhiều phương pháp khác nhau để tạo mảng. Chúng tôi sẽ khám phá một số kỹ thuật phổ biến để tạo mảng, bao gồm việc sử dụng danh sách, phạm vi và các hàm đặc biệt.

```python
# Create an array from a list
list_array = np.array([1, 2, 3, 4, 5])

# Create an array with a range of values
range_array = np.arange(0, 10, 2)

# Create an array of ones
ones_array = np.ones((3, 3))

print("List array:", list_array)
print("Range array:", range_array)
print("Ones array:\n", ones_array)
```

Trang trình bày 3: Thuộc tính mảng: Tìm hiểu dữ liệu của bạn

Table NumPy có một số thuộc tính cung cấp thông tin hữu ích về cấu trúc và nội dung của chúng. Hãy khám phá một số thuộc tính chính.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print("Shape:", arr.shape)
print("Dimensions:", arr.ndim)
print("Data type:", arr.dtype)
print("Size:", arr.size)

# Output:
# Shape: (2, 3)
# Dimensions: 2
# Data type: int64
# Size: 6
```

Slide 4: Lập chỉ mục và cắt lát: Truy cập mảng tử phần tử

Tác vụ hiệu quả dữ liệu thường yêu cầu truy cập các phần tử hoặc tập hợp cụ thể của một mảng. NumPy cung cấp khả năng lập chỉ mục và cắt mạnh mẽ.

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Indexing
print("Element at (1, 2):", arr[1, 2])

# Slicing
print("First two rows, last two columns:\n", arr[:2, 2:])

# Boolean indexing
mask = arr > 5
print("Elements greater than 5:\n", arr[mask])
```

Slide 5: Thao tác với mảng: Định hình lại và xếp chồng

NumPy cung cấp nhiều chức năng khác nhau để thao tác với nhiều mảng và kết hợp nhiều mảng. Hoạt động này rất quan trọng đối với quá trình xử lý dữ liệu và kỹ năng.

```python
# Reshape an array
arr = np.arange(12)
reshaped = arr.reshape((3, 4))
print("Reshaped array:\n", reshaped)

# Stack arrays vertically
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
vertical_stack = np.vstack((a, b))
print("Vertical stack:\n", vertical_stack)

# Stack arrays horizontally
horizontal_stack = np.hstack((a, b))
print("Horizontal stack:", horizontal_stack)
```

Slide 6: Các phép toán: Toán và ma trận

NumPy đơn giản hóa các hoạt động của thú và ma trận, cho phép tính toán và phát hiện các bước sóng theo từng phần tử cho các mảng có hình dạng khác nhau.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise operations
print("Addition:", a + b)
print("Multiplication:", a * b)

# Broadcasting
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("Matrix + vector:\n", matrix + a)

# Matrix multiplication
result = np.dot(matrix, a)
print("Matrix-vector product:", result)
```

Slide 7: Các thao tác thống kê: Thống kê mô tả

NumPy cung cấp nhiều thống kê chức năng để phân tích dữ liệu của bạn một cách nhanh chóng và hiệu quả.

```python
data = np.array([14, 23, 32, 41, 50, 59])

print("Mean:", np.mean(data))
print("Median:", np.median(data))
print("Standard deviation:", np.std(data))
print("Variance:", np.var(data))
print("Min and Max:", np.min(data), np.max(data))

# Output:
# Mean: 36.5
# Median: 36.5
# Standard deviation: 16.19
# Variance: 262.25
# Min and Max: 14 59
```

Slide 8: Đại số tuyến tính: Các ma trận toán được phép

Mô-đun đại số tuyến tính của NumPy cung cấp các công cụ mạnh mẽ cho các ma trận toán phép, tính toán giá trị riêng và giải các hệ thống tuyến tính.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = np.dot(A, B)
print("Matrix multiplication:\n", C)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Solve linear system Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)
print("Solution to Ax = b:", x)
```

Slide 9: Broadcasting: Hoạt động mảng hiệu ứng

Broadcasting là một tính năng NumPy mạnh mẽ cho phép hoạt động giữa các mảng có nhiều dạng hình khác nhau. Nó có thể đơn giản hóa đáng kể mã hóa của bạn và cải thiện hiệu suất.

```python
# Broadcasting example
a = np.array([1, 2, 3])
b = np.array([[1], [2], [3]])

result = a + b
print("Broadcasting result:\n", result)

# Without broadcasting, we'd need to do:
# result = np.array([a + b[i] for i in range(3)])

# Output:
# Broadcasting result:
# [[2 3 4]
#  [3 4 5]
#  [4 5 6]]
```

Slide 10: Tạo số ngẫu nhiên: Mô phỏng và lấy mẫu

Mô-đun ngẫu nhiên của NumPy cung cấp nhiều chức năng khác nhau để tạo ra số ngẫu nhiên, rất quan trọng cho mô phỏng, phân tích thống kê và học máy.

```python
# Set a seed for reproducibility
np.random.seed(42)

# Generate random integers
random_ints = np.random.randint(1, 11, size=5)
print("Random integers:", random_ints)

# Generate random floats
random_floats = np.random.random(5)
print("Random floats:", random_floats)

# Generate numbers from a normal distribution
normal_dist = np.random.normal(loc=0, scale=1, size=5)
print("Normal distribution:", normal_dist)
```

Slide 11: Ví dụ thực tế: Xử lý ảnh

NumPy được sử dụng rộng rãi trong quá trình xử lý ảnh. Hãy tạo một ví dụ đơn giản về thao tác hình ảnh bằng NumPy.

```python
# Create a simple 5x5 grayscale image
image = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

# Rotate the image by 90 degrees
rotated = np.rot90(image)

print("Original image:\n", image)
print("\nRotated image:\n", rotated)

# Apply a simple filter (e.g., edge detection)
filter = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

filtered = np.zeros_like(image)
for i in range(1, 4):
    for j in range(1, 4):
        filtered[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * filter)

print("\nFiltered image (edge detection):\n", filtered)
```

Trang trình bày 12: Ví dụ thực tế: Phân tích chuỗi thời gian

NumPy cũng có giá trị cho việc phân tích chuỗi thời gian. Hãy tạo một ví dụ đơn giản về phân tích nhiệt độ dữ liệu.

```python
# Generate synthetic temperature data
days = np.arange(1, 31)
temperatures = 20 + 5 * np.sin(days * 2 * np.pi / 30) + np.random.normal(0, 1, 30)

# Calculate moving average
window_size = 3
moving_avg = np.convolve(temperatures, np.ones(window_size), 'valid') / window_size

print("First 10 days temperatures:", temperatures[:10])
print("Moving average (first 8 days):", moving_avg[:8])

# Find days with temperature above average
above_avg = days[temperatures > np.mean(temperatures)]
print("Days with above-average temperature:", above_avg)

# Calculate temperature range
temp_range = np.ptp(temperatures)
print(f"Temperature range: {temp_range:.2f}")
```

Trang trình bày 13: Hiệu suất NumPy: Vector hóa so với vòng lặp

Một trong những ưu tiên chính của NumPy là khả năng thực hiện các phép toán được phép hóa vector nhanh hơn nhiều so với hệ thống truyền Python vòng lặp. Hãy so sánh hiệu suất.

```python
import time

# Create a large array
arr = np.random.random(1000000)

# Using a loop
start_time = time.time()
result_loop = [x**2 for x in arr]
loop_time = time.time() - start_time

# Using NumPy vectorization
start_time = time.time()
result_numpy = arr**2
numpy_time = time.time() - start_time

print(f"Loop time: {loop_time:.6f} seconds")
print(f"NumPy time: {numpy_time:.6f} seconds")
print(f"NumPy is {loop_time/numpy_time:.2f}x faster")
```

Trang trình bày 14: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về NumPy và các ứng dụng của nó trong dữ liệu khoa học, thì đây là một số tài nguyên có giá trị:

1. Tài liệu NumPy: [https://numpy.org/doc/](https://numpy.org/doc/)
2. "Từ Python đến Numpy" của Nicolas P. Rougier: [https://www.labri.fr/perso/nrougier/from-python-to-numpy/](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)
3. "NumPy: Hướng dẫn về NumPy" của Travis E. Oliphant: [https://web.mit.edu/dvp/Public/numpybook.pdf](https://web.mit.edu/dvp/Public/numpybook.pdf)
4. Bài viết ArXiv: "Lập trình mảng với NumPy" (2020): [https://arxiv.org/abs/2006.10256](https://arxiv.org/abs/2006.10256)

Những tài nguyên này cung cấp những giải thích sâu sắc, các kỹ thuật tiên tiến và các ứng dụng thực tế của NumPy trong tính toán khoa học và phân tích dữ liệu.
