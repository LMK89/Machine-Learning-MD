## Bảng cheat NumPy toàn diện để tạo mảng trong Python
Slide 1: Giới thiệu về mảng NumPy

NumPy là một thư viện mạnh mẽ để tính toán số bằng Python. Cốt lõi của nó là mảng NumPy, là những vùng chứa đa chiều, hiệu quả cho dữ liệu tối đa. Mảng này tạo ra nền tảng cho nhiều khoa học và học toán được phép trong Python.

```python
import numpy as np

# Create a 1D array
arr_1d = np.array([1, 2, 3, 4, 5])
print("1D array:", arr_1d)

# Create a 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D array:\n", arr_2d)

# Output:
# 1D array: [1 2 3 4 5]
# 2D array:
# [[1 2 3]
#  [4 5 6]]
```

Slide 2: Hàm tạo mảng

NumPy cung cấp nhiều chức năng khác nhau để tạo mảng cho các thuộc tính cụ thể. Hàm này rất cần thiết để khởi tạo hiệu ứng dữ liệu cấu trúc.

```python
import numpy as np

# Create an array of zeros
zeros_arr = np.zeros((3, 4))
print("Zeros array:\n", zeros_arr)

# Create an array of ones
ones_arr = np.ones((2, 3))
print("Ones array:\n", ones_arr)

# Create an identity matrix
identity_matrix = np.eye(3)
print("Identity matrix:\n", identity_matrix)

# Output:
# Zeros array:
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]
# Ones array:
# [[1. 1. 1.]
#  [1. 1. 1.]]
# Identity matrix:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]
```

Slide 3: Chuỗi và dãy

NumPy cung cấp các hàm để tạo mảng với các giá trị đều nhau, rất hữu ích cho việc tạo chuỗi và phạm vi.

```python
import numpy as np

# Create an array with a range of values
range_arr = np.arange(0, 10, 2)
print("Range array:", range_arr)

# Create an array with evenly spaced values
linspace_arr = np.linspace(0, 1, 5)
print("Linspace array:", linspace_arr)

# Create a logarithmically spaced array
logspace_arr = np.logspace(0, 2, 5)
print("Logspace array:", logspace_arr)

# Output:
# Range array: [0 2 4 6 8]
# Linspace array: [0.   0.25 0.5  0.75 1.  ]
# Logspace array: [  1.           3.16227766  10.          31.6227766  100.        ]
```

Slide 4: Định hình lại và một số mảng

NumPy cho phép dễ dàng thao tác với các mảng hình dạng và kích thước, cho phép tái sử dụng hiệu quả cơ sở dữ liệu.

```python
import numpy as np

# Create a 1D array
arr = np.arange(12)
print("Original array:", arr)

# Reshape the array to 2D
reshaped_arr = arr.reshape(3, 4)
print("Reshaped array:\n", reshaped_arr)

# Transpose the 2D array
transposed_arr = reshaped_arr.T
print("Transposed array:\n", transposed_arr)

# Output:
# Original array: [ 0  1  2  3  4  5  6  7  8  9 10 11]
# Reshaped array:
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]
# Transposed array:
# [[ 0  4  8]
#  [ 1  5  9]
#  [ 2  6 10]
#  [ 3  7 11]]
```

Slide 5: Lập chỉ mục và cắt mảng

Truy cập và vận hành hiệu quả dữ liệu trong mảng NumPy đạt được thông tin qua chỉ mục và cắt hoạt động.

```python
import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("Original array:\n", arr)

# Indexing: Get a single element
print("Element at (1, 2):", arr[1, 2])

# Slicing: Get a sub-array
print("Slice (first 2 rows, last 2 columns):\n", arr[:2, 2:])

# Boolean indexing
mask = arr > 5
print("Elements greater than 5:\n", arr[mask])

# Output:
# Original array:
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
# Element at (1, 2): 7
# Slice (first 2 rows, last 2 columns):
# [[3 4]
#  [7 8]]
# Elements greater than 5:
# [ 6  7  8  9 10 11 12]
```

Slide 6: Các cơ sở dữ liệu mảng hoạt động

NumPy cung cấp các kết quả tính toán được phép theo từng phần tử trên mảng, đơn giản hóa các phép tính toán học.

```python
import numpy as np

# Create two arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Addition
print("Addition:", a + b)

# Multiplication
print("Multiplication:", a * b)

# Exponentiation
print("Exponentiation:", a ** 2)

# Dot product
print("Dot product:", np.dot(a, b))

# Output:
# Addition: [5 7 9]
# Multiplication: [ 4 10 18]
# Exponentiation: [1 4 9]
# Dot product: 32
```

Slide 7: Phát sóng mảng

Việc phát hiện sóng được phép NumPy thực hiện các thao tác trên các mảng có hình dạng khác nhau, mở rộng các mảng nhỏ hơn để phù hợp với các mảng lớn hơn.

```python
import numpy as np

# Create a 2D array and a 1D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_1d = np.array([10, 20, 30])

# Broadcasting: Add 1D array to each row of 2D array
result = arr_2d + arr_1d
print("Result of broadcasting:")
print(result)

# Broadcasting with scalar
scalar_result = arr_2d * 2
print("Result of scalar broadcasting:")
print(scalar_result)

# Output:
# Result of broadcasting:
# [[11 22 33]
#  [14 25 36]]
# Result of scalar broadcasting:
# [[ 2  4  6]
#  [ 8 10 12]]
```

Slide 8: Hàm tổng hợp

NumPy cung cấp nhiều chức năng khác nhau để thực hiện hợp nhất các phép tính trên mảng, đưa ra giới hạn như tính tổng, phương tiện và cực trị.

```python
import numpy as np

# Create a 2D array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original array:\n", arr)

# Sum of all elements
print("Sum of all elements:", np.sum(arr))

# Mean of all elements
print("Mean of all elements:", np.mean(arr))

# Maximum and minimum values
print("Maximum value:", np.max(arr))
print("Minimum value:", np.min(arr))

# Sum along axis 0 (columns)
print("Sum along columns:", np.sum(arr, axis=0))

# Mean along axis 1 (rows)
print("Mean along rows:", np.mean(arr, axis=1))

# Output:
# Original array:
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]
# Sum of all elements: 45
# Mean of all elements: 5.0
# Maximum value: 9
# Minimum value: 1
# Sum along columns: [12 15 18]
# Mean along rows: [2. 5. 8.]
```

Slide 9: Sắp xếp và tìm kiếm mảng

NumPy cung cấp các kết quả chức năng để sắp xếp mảng và tìm kiếm các phần tử hoặc điều kiện cụ thể.

```python
import numpy as np

# Create an unsorted array
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
print("Original array:", arr)

# Sort the array
sorted_arr = np.sort(arr)
print("Sorted array:", sorted_arr)

# Find indices that would sort the array
sort_indices = np.argsort(arr)
print("Indices that would sort the array:", sort_indices)

# Find unique elements
unique_elements = np.unique(arr)
print("Unique elements:", unique_elements)

# Search for a value
value_to_search = 5
indices = np.where(arr == value_to_search)
print(f"Indices where {value_to_search} is found:", indices[0])

# Output:
# Original array: [3 1 4 1 5 9 2 6 5 3 5]
# Sorted array: [1 1 2 3 3 4 5 5 5 6 9]
# Indices that would sort the array: [1 3 6 0 9 2 4 8 10 7 5]
# Unique elements: [1 2 3 4 5 6 9]
# Indices where 5 is found: [4 8 10]
```

Slide 10: Nối và chia mảng

NumPy cho phép dễ dàng kết hợp và phân chia mảng dọc theo các trục được xác định chỉ.

```python
import numpy as np

# Create two arrays
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

# Concatenate arrays vertically
vertical_concat = np.concatenate((arr1, arr2), axis=0)
print("Vertical concatenation:\n", vertical_concat)

# Concatenate arrays horizontally
horizontal_concat = np.concatenate((arr1, arr2), axis=1)
print("Horizontal concatenation:\n", horizontal_concat)

# Split an array vertically
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])
vertical_split = np.split(arr, 3, axis=0)
print("Vertical split:")
for i, sub_arr in enumerate(vertical_split):
    print(f"Sub-array {i}:\n", sub_arr)

# Output:
# Vertical concatenation:
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]
# Horizontal concatenation:
# [[1 2 5 6]
#  [3 4 7 8]]
# Vertical split:
# Sub-array 0:
#  [[1 2 3 4]]
# Sub-array 1:
#  [[5 6 7 8]]
# Sub-array 2:
#  [[ 9 10 11 12]]
```

Trang trình bày 11: Lấy biến ngẫu nhiên ngẫu nhiên

Mô-đun ngẫu nhiên của NumPy cung cấp các chức năng tạo ra số ngẫu nhiên và lấy mẫu từ các phân bố cụ thể khác nhau.

```python
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Generate random integers
random_ints = np.random.randint(0, 10, size=5)
print("Random integers:", random_ints)

# Generate random floats
random_floats = np.random.random(5)
print("Random floats:", random_floats)

# Sample from a normal distribution
normal_samples = np.random.normal(loc=0, scale=1, size=5)
print("Samples from normal distribution:", normal_samples)

# Shuffle an array
arr = np.arange(10)
np.random.shuffle(arr)
print("Shuffled array:", arr)

# Output:
# Random integers: [6 3 7 4 6]
# Random floats: [0.37454012 0.95071431 0.73199394 0.59865848 0.15601864]
# Samples from normal distribution: [ 0.60276338 -0.54488318  0.43788141  0.88053932  1.46564877]
# Shuffled array: [2 8 4 9 1 6 7 3 0 5]
```

Slide 12: Ví dụ thực tế: Xử lý ảnh

NumPy được sử dụng rộng rãi trong các ảnh xử lý tác vụ. Đây là ví dụ về cách tải hình ảnh, chuyển đổi nó sang thang độ xám và áp dụng bộ lọc đơn giản.

```python
import numpy as np
from PIL import Image

# Load an image (assuming 'image.jpg' exists in the current directory)
img = np.array(Image.open('image.jpg'))
print("Original image shape:", img.shape)

# Convert to grayscale
gray_img = np.mean(img, axis=2).astype(np.uint8)
print("Grayscale image shape:", gray_img.shape)

# Apply a simple blur filter
kernel = np.ones((5, 5)) / 25  # 5x5 averaging filter
blurred = np.zeros_like(gray_img)
for i in range(2, gray_img.shape[0] - 2):
    for j in range(2, gray_img.shape[1] - 2):
        blurred[i, j] = np.sum(gray_img[i-2:i+3, j-2:j+3] * kernel)

print("Blurred image shape:", blurred.shape)

# Save the processed images
Image.fromarray(gray_img).save('grayscale.jpg')
Image.fromarray(blurred).save('blurred.jpg')

# Note: This code assumes you have an 'image.jpg' file in your working directory
# and the necessary permissions to read and write files.
```

Trang trình bày 13: Ví dụ thực tế: Phân tích dữ liệu

NumPy rất quan trọng đối với các nhiệm vụ phân tích dữ liệu. Đây là ví dụ về phân tích nhiệt độ dữ liệu của một thành phố trong hơn một năm.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic temperature data for a year (365 days)
temperatures = np.random.normal(loc=15, scale=10, size=365)  # Mean 15°C, std dev 10°C

# Calculate basic statistics
avg_temp = np.mean(temperatures)
max_temp = np.max(temperatures)
min_temp = np.min(temperatures)

print(f"Average temperature: {avg_temp:.2f}°C")
print(f"Maximum temperature: {max_temp:.2f}°C")
print(f"Minimum temperature: {min_temp:.2f}°C")

# Find days above 25°C (hot days)
hot_days = np.sum(temperatures > 25)
print(f"Number of hot days (>25°C): {hot_days}")

# Calculate moving average (7-day window)
moving_avg = np.convolve(temperatures, np.ones(7), 'valid') / 7

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(temperatures, label='Daily Temperature')
plt.plot(np.arange(3, 362), moving_avg, label='7-day Moving Average', color='red')
plt.xlabel('Day of the Year')
plt.ylabel('Temperature (°C)')
plt.title('Yearly Temperature Analysis')
plt.legend()
plt.grid(True)
plt.savefig('temperature_analysis.png')
plt.close()

print("Temperature analysis plot saved as 'temperature_analysis.png'")

# Note: This code generates a plot file. Ensure you have write permissions
# in your working directory.
```

Trang trình bày 14: Tài nguyên bổ sung

Để khám phá thêm về NumPy và các ứng dụng của nó trong điện toán học khoa học, hãy xem xét các tài nguyên sau:

1. Tài liệu chính thức của NumPy: [https://numpy.org/doc/](https://numpy.org/doc/) Hướng dẫn toàn diện này bao gồm tất cả các cạnh của NumPy, từ các chủ đề cơ bản nâng cao.
2. "Từ Python đến NumPy" của Nicolas P. Rougier Có tại: [https://www.labri.fr/perso/nrougier/from-python-to-numpy/](https://www.labri.fr/perso/nrougier/from-python-to-numpy/) Cuốn sách trực tuyến miễn phí này cung cấp cái nhìn sâu sắc về khả năng và mức độ tối ưu hóa của NumPy.
3. "Cẩm nang khoa học dữ liệu Python" của Jake VanderPlas Cuốn sách này bao gồm nội dung thú vị về NumPy và sự hợp lý của các công cụ khoa học dữ liệu khác.
4. Ghi chú bài giải SciPy Có sẵn tại: [https://scipy-lectures.org/](https://scipy-lectures.org/) Ghi chú bài giải này bao gồm NumPy cùng với các thư viện Python khoa học khác.
5. Hướng dẫn NumPy về Python thực tế: [https://realpython.com/tutorials/numpy/](https://realpython.com/tutorials/numpy/) Tập hợp các hướng dẫn thực tế bao gồm các cạnh viền khác nhau của NumPy.
6. Bài viết ArXiv: "Lập trình mảng với NumPy" của Harris et al. (2020) URL ArXiv: [https://arxiv.org/abs/2006.10256](https://arxiv.org/abs/2006.10256) Bài viết này cung cấp thông tin chi tiết về thiết kế của NumPy và hoạt động của nó đối với điện toán học.

Tài nguyên này cung cấp sự kết hợp giữa tài liệu chính thức, sách, hướng dẫn và tài liệu học thuật để giúp bạn hiểu sâu hơn về NumPy và các ứng dụng của nó trong tính toán khoa học và dữ liệu phân tích.
