## Thuộc tính khung dữ liệu Pandas và ví dụ về mã Python
Trang trình bày 1: Thuộc tính Pandas DataFrame

DataFrames là cấu trúc dữ liệu được sử dụng phổ biến nhất trong gấu trúc. Chúng là các cấu trúc dữ liệu được dán nhãn hai chiều với các cột có thể có các kiểu khác nhau. Hiểu các thuộc tính DataFrame là rất quan trọng để thao tác và phân tích dữ liệu hiệu quả.

```python
import pandas as pd

# Create a sample DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Paris']
})

print(df)
```

Trang trình bày 2: DataFrame.shape

Thuộc tính hình dạng trả về một bộ dữ liệu biểu thị chiều của DataFrame. Nó cung cấp số lượng hàng và cột trong DataFrame.

```python
# Get the shape of the DataFrame
shape = df.shape

print(f"Number of rows: {shape[0]}")
print(f"Number of columns: {shape[1]}")
```

Trang trình bày 3: DataFrame.dtypes

Thuộc tính dtypes trả về kiểu dữ liệu của từng cột trong DataFrame. Điều này rất cần thiết để hiểu bản chất dữ liệu của bạn và thực hiện các hoạt động thích hợp.

```python
# Display the data types of each column
print(df.dtypes)

# Change the data type of a column
df['Age'] = df['Age'].astype(float)
print(df.dtypes)
```

Trang trình bày 4: DataFrame.index

Thuộc tính chỉ mục đại diện cho nhãn hàng của DataFrame. Nó có thể được tùy chỉnh để sử dụng các mã định danh có ý nghĩa thay vì các chỉ số nguyên mặc định.

```python
# Display the current index
print(df.index)

# Set a custom index
df.set_index('Name', inplace=True)
print(df)
print(df.index)
```

Trang trình bày 5: DataFrame.columns

Thuộc tính cột trả về nhãn cột của DataFrame. Nó có thể được sử dụng để truy cập, sửa đổi hoặc đổi tên các cột.

```python
# Display column names
print(df.columns)

# Rename columns
df.columns = ['Years', 'Location']
print(df)
```

Trang trình bày 6: DataFrame.values

Thuộc tính giá trị trả về mảng NumPy chứa dữ liệu trong DataFrame. Điều này hữu ích khi bạn cần thực hiện các thao tác yêu cầu mảng NumPy thuần túy.

```python
# Get the values as a NumPy array
array_data = df.values
print(array_data)
print(type(array_data))
```

Slide 7: DataFrame.empty

The empty attribute returns a boolean indicating whether the DataFrame is empty (contains no data). This is useful for error checking and flow control in data processing pipelines.

```python
# Check if the DataFrame is empty
print(f"Is the DataFrame empty? {df.empty}")

# Create an empty DataFrame
empty_df = pd.DataFrame()
print(f"Is the new DataFrame empty? {empty_df.empty}")
```

Trang trình bày 8: DataFrame.size

Thuộc tính size trả về tổng số phần tử trong DataFrame. Nó bằng số hàng nhân với số cột.

```python
# Get the size of the DataFrame
print(f"Total number of elements: {df.size}")

# Verify the calculation
total_elements = df.shape[0] * df.shape[1]
print(f"Calculated total elements: {total_elements}")
```

Trang trình bày 9: DataFrame.ndim

Thuộc tính ndim trả về số thứ nguyên của DataFrame. Đối với DataFrame tiêu chuẩn, giá trị này sẽ luôn là 2 (hàng và cột).

```python
# Get the number of dimensions
print(f"Number of dimensions: {df.ndim}")

# Create a Series (1-dimensional) for comparison
series = pd.Series([1, 2, 3])
print(f"Number of dimensions in a Series: {series.ndim}")
```

Trang trình bày 10: DataFrame.axes

Thuộc tính trục trả về danh sách nhãn trục hàng và nhãn trục cột. Điều này có thể hữu ích để hiểu cấu trúc DataFrame của bạn.

```python
# Get the axes of the DataFrame
axes = df.axes
print(f"Row labels: {axes[0]}")
print(f"Column labels: {axes[1]}")
```

Slide 11: DataFrame.info()

While not strictly an attribute, the info() method provides a concise summary of the DataFrame, including the index dtype and column dtypes, non-null values, and memory usage.

```python
# Display DataFrame info
df.info()

# Display DataFrame info with memory usage
df.info(memory_usage="deep")
```

Trang trình bày 12: Ví dụ thực tế: Phân tích dữ liệu thời tiết

Hãy sử dụng thuộc tính DataFrame để phân tích dữ liệu thời tiết cho các thành phố khác nhau.

```python
import pandas as pd
import numpy as np

# Create a DataFrame with weather data
weather_data = pd.DataFrame({
    'City': ['Tokyo', 'New York', 'London', 'Paris'],
    'Temperature': [25.5, 22.1, 18.7, 20.3],
    'Humidity': [60, 55, 70, 65],
    'Wind_Speed': [10.2, 8.5, 12.1, 9.8]
})

print(weather_data)
print(f"\nShape: {weather_data.shape}")
print(f"\nData Types:\n{weather_data.dtypes}")
print(f"\nColumn Names: {weather_data.columns}")
```

Slide 13: Ví dụ thực tế: Phân tích kết quả học tập của sinh viên

Hãy sử dụng thuộc tính DataFrame để phân tích dữ liệu hiệu suất của học sinh.

```python
# Create a DataFrame with student performance data
student_data = pd.DataFrame({
    'Student_ID': ['S001', 'S002', 'S003', 'S004', 'S005'],
    'Math_Score': [85, 92, 78, 95, 88],
    'Science_Score': [90, 88, 82, 96, 85],
    'Literature_Score': [75, 85, 92, 88, 91]
})

student_data.set_index('Student_ID', inplace=True)
print(student_data)
print(f"\nIndex: {student_data.index}")
print(f"\nSize: {student_data.size}")
print(f"\nMean Scores:\n{student_data.mean()}")
```

Trang trình bày 14: Tài nguyên bổ sung

Để biết thêm các chủ đề nâng cao và giải thích chuyên sâu về thuộc tính DataFrame của gấu trúc, hãy cân nhắc khám phá các tài nguyên sau:

1. Tài liệu chính thức về gấu trúc: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
2. "Những chú gấu trúc hiệu quả" của Matt Harrison: [https://github.com/mattharrison/effect\_pandas](https://github.com/mattharrison/effect_pandas)
3. "Python để phân tích dữ liệu" của Wes McKinney (người tạo ra gấu trúc): [https://wesmckinney.com/book/](https://wesmckinney.com/book/)

Các tài nguyên này cung cấp thông tin toàn diện về gấu trúc và các khả năng của nó, giúp bạn nắm vững thao tác và phân tích DataFrame.
