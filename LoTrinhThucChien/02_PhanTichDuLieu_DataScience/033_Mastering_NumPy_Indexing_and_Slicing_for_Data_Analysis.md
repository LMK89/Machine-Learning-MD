## Nắm vững cách lập chỉ mục và cắt NumPy để phân tích dữ liệu
Trang trình bày 1: Giới thiệu về lập chỉ mục và cắt NumPy

NumPy, một thư viện cơ bản cho tính toán khoa học bằng Python, cung cấp các công cụ mạnh mẽ để thao tác dữ liệu. Lập chỉ mục và cắt lát là các kỹ thuật chính cho phép truy cập và sửa đổi hiệu quả các phần tử mảng. Các hoạt động này tạo thành nền tảng cho việc phân tích và xử lý dữ liệu nâng cao trong NumPy.

```python
import numpy as np

# Create a sample 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print("Original array:")
print(arr)
```

Slide 2: Lập chỉ mục cơ bản trong NumPy

Mảng NumPy hỗ trợ lập chỉ mục số nguyên tương tự như danh sách Python. Tuy nhiên, NumPy mở rộng khái niệm này sang nhiều chiều, cho phép lựa chọn phần tử chính xác trong mảng đa chiều.

```python
# Accessing elements using integer indexing
print("Element at index [1, 2]:", arr[1, 2])
print("First row:", arr[0])
print("Last column:", arr[:, -1])
```

Slide 3: Slicing in NumPy

Slicing in NumPy allows extracting subarrays by specifying start, stop, and step values for each dimension. This powerful feature enables efficient data subset selection and manipulation.

```python
# Slicing examples
print("First two rows, all columns:")
print(arr[:2, :])

print("All rows, last two columns:")
print(arr[:, 1:])

print("Every other element in the first row:")
print(arr[0, ::2])
```

Trang trình bày 4: Lập chỉ mục nâng cao: Lập chỉ mục Boolean

Lập chỉ mục Boolean sử dụng mảng boolean để chọn các phần tử thỏa mãn các điều kiện cụ thể. Kỹ thuật này đặc biệt hữu ích để lọc dữ liệu dựa trên các tiêu chí phức tạp.

```python
# Boolean indexing
mask = arr > 5
print("Elements greater than 5:")
print(arr[mask])

# Combining conditions
complex_mask = (arr > 3) & (arr < 8)
print("Elements between 3 and 8:")
print(arr[complex_mask])
```

Trang trình bày 5: Lập chỉ mục nâng cao: Lập chỉ mục mảng số nguyên

Lập chỉ mục mảng số nguyên cho phép chọn các phần tử bằng cách sử dụng mảng chỉ mục. Kỹ thuật này cho phép thực hiện các hoạt động lựa chọn và sắp xếp lại phần tử phức tạp.

```python
# Integer array indexing
row_indices = np.array([0, 1, 2])
col_indices = np.array([2, 1, 0])
print("Selected elements:")
print(arr[row_indices, col_indices])

# Selecting specific elements
print("Elements at (0,0), (1,1), and (2,2):")
print(arr[np.arange(3), np.arange(3)])
```

Slide 6: Sửa đổi các phần tử mảng

Khả năng lập chỉ mục và cắt của NumPy cũng cho phép sửa đổi mảng hiệu quả. Các phần tử có thể được cập nhật riêng lẻ hoặc theo nhóm bằng nhiều kỹ thuật lập chỉ mục khác nhau.

```python
# Modifying elements
arr[1, 1] = 10
arr[:, 2] = [30, 60, 90]
print("Modified array:")
print(arr)

# Broadcasting with boolean indexing
arr[arr < 30] *= 2
print("Array after doubling elements < 30:")
print(arr)
```

Slide 7: Ví dụ thực tế: Xử lý hình ảnh

Việc lập chỉ mục và cắt lát của NumPy được sử dụng rộng rãi trong xử lý ảnh. Hãy trình bày một thao tác cắt ảnh đơn giản.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a sample 8x8 grayscale image
image = np.random.randint(0, 256, (8, 8))

# Crop the image
cropped = image[2:6, 2:6]

# Display original and cropped images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(image, cmap='gray')
ax1.set_title("Original Image")
ax2.imshow(cropped, cmap='gray')
ax2.set_title("Cropped Image")
plt.show()
```

Slide 8: Lập chỉ mục ưa thích

Lập chỉ mục ưa thích cho phép chọn hoặc sửa đổi các tập hợp con của một mảng một cách linh hoạt bằng cách sử dụng mảng số nguyên hoặc mặt nạ boolean. Kỹ thuật này đặc biệt hữu ích cho các tác vụ thao tác dữ liệu phức tạp.

```python
# Create a sample array
arr = np.arange(16).reshape(4, 4)

# Select specific rows and columns
rows = np.array([0, 2, 3])
cols = np.array([1, 2])
selected = arr[rows[:, np.newaxis], cols]

print("Original array:")
print(arr)
print("\nSelected sub-array:")
print(selected)
```

Trang trình bày 9: Tạo mặt nạ và lọc

Mặt nạ cho phép lựa chọn có điều kiện các phần tử mảng dựa trên giá trị của chúng hoặc các tiêu chí khác. Kỹ thuật này rất quan trọng để làm sạch và tiền xử lý dữ liệu.

```python
# Create a sample array
data = np.random.randn(5, 5)

# Create a mask for positive values
mask = data > 0

# Apply the mask
filtered_data = data[mask]

print("Original data:")
print(data)
print("\nFiltered data (positive values only):")
print(filtered_data)
```

Slide 10: Cắt lát với kích thước bước

NumPy cho phép chỉ định kích thước bước khi cắt, cho phép chọn mọi phần tử thứ n. Điều này rất hữu ích cho việc lấy mẫu xuống hoặc chọn các mẫu cụ thể trong dữ liệu.

```python
# Create a sample array
arr = np.arange(20)

# Select every third element
every_third = arr[::3]

# Reverse the array
reversed_arr = arr[::-1]

print("Original array:", arr)
print("Every third element:", every_third)
print("Reversed array:", reversed_arr)
```

Slide 11: Cắt lát đa chiều

Khả năng cắt của NumPy mở rộng liền mạch sang các mảng đa chiều, cho phép trích xuất và thao tác dữ liệu phức tạp ở các chiều cao hơn.

```python
# Create a 3D array
arr_3d = np.arange(27).reshape(3, 3, 3)

# Extract a 2D slice
slice_2d = arr_3d[1, :, :]

# Extract a 1D slice
slice_1d = arr_3d[1, 1, :]

print("3D array:")
print(arr_3d)
print("\n2D slice:")
print(slice_2d)
print("\n1D slice:")
print(slice_1d)
```

Trang trình chiếu 12: Ví dụ thực tế: Phân tích chuỗi thời gian

Việc lập chỉ mục và cắt của NumPy là vô giá trong phân tích chuỗi thời gian. Hãy trình bày cách chọn khoảng thời gian cụ thể từ tập dữ liệu.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample time series data
dates = np.arange('2023-01-01', '2024-01-01', dtype='datetime64[D]')
values = np.cumsum(np.random.randn(365))

# Select data for Q2 2023
q2_mask = (dates >= '2023-04-01') & (dates < '2023-07-01')
q2_dates = dates[q2_mask]
q2_values = values[q2_mask]

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(dates, values, label='Full Year')
plt.plot(q2_dates, q2_values, label='Q2 2023')
plt.title('Time Series Data Analysis')
plt.legend()
plt.show()
```

Trang trình bày 13: Cân nhắc về hiệu suất

Các hoạt động lập chỉ mục và cắt của NumPy được tối ưu hóa cao về hiệu suất. Tuy nhiên, một số phương pháp nhất định có thể tác động đáng kể đến hiệu quả, đặc biệt khi xử lý các tập dữ liệu lớn.

```python
import numpy as np
import timeit

# Create a large array
large_arr = np.random.rand(1000000)

# Compare performance of different indexing methods
def method1():
    return large_arr[large_arr > 0.5]

def method2():
    mask = large_arr > 0.5
    return large_arr[mask]

time1 = timeit.timeit(method1, number=100)
time2 = timeit.timeit(method2, number=100)

print(f"Method 1 time: {time1:.6f} seconds")
print(f"Method 2 time: {time2:.6f} seconds")
```

Slide 14: Lập chỉ mục nâng cao: Kỹ thuật kết hợp

NumPy cho phép kết hợp các kỹ thuật lập chỉ mục và cắt khác nhau để thao tác dữ liệu phức tạp. Tính linh hoạt này rất quan trọng đối với các nhiệm vụ phân tích dữ liệu nâng cao.

```python
# Create a sample 3D array
arr_3d = np.arange(27).reshape(3, 3, 3)

# Combine boolean and integer indexing
mask = arr_3d > 10
selected = arr_3d[mask][:5]

# Combine slicing and fancy indexing
complex_slice = arr_3d[1:, [0, 2], ::2]

print("Selected elements:", selected)
print("\nComplex slice:")
print(complex_slice)
```

Trang trình bày 15: Tài nguyên bổ sung

Để khám phá thêm về lập chỉ mục và cắt NumPy:

1. Tài liệu chính thức của NumPy: [https://numpy.org/doc/stable/user/basics.indexing.html](https://numpy.org/doc/stable/user/basics.indexing.html)
2. "Giới thiệu trực quan về NumPy và biểu diễn dữ liệu" của Jay Alammar: [https://jalammar.github.io/visual-numpy/](https://jalammar.github.io/visual-numpy/)
3. Chương "NumPy: Tạo và thao tác dữ liệu số" trong "Sổ tay khoa học dữ liệu Python" của Jake VanderPlas: [https://arxiv.org/abs/1607.01719](https://arxiv.org/abs/1607.01719)

Các tài nguyên này cung cấp những giải thích sâu sắc và các ví dụ bổ sung để nâng cao hiểu biết của bạn về khả năng lập chỉ mục và cắt lát mạnh mẽ của NumPy.
