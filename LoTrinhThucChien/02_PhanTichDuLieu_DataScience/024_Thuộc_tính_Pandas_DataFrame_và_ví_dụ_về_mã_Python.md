## Thuộc tính Pandas dữ liệu khung và ví dụ về Python mã hóa
Trang trình bày 1: Thuộc tính Pandas DataFrame

DataFrames là cấu trúc dữ liệu được sử dụng phổ biến nhất trong gấu trúc. Chúng là các cấu trúc dữ liệu được dán hai chiều nhãn với các cột có thể có các kiểu khác nhau. Hiểu các thuộc tính DataFrame là rất quan trọng để thao tác và phân tích hiệu quả dữ liệu.

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

Định dạng thuộc tính trả về một bộ dữ liệu biểu thị chiều của DataFrame. Nó cung cấp số lượng hàng và cột trong DataFrame.

```python
# Get the shape of the DataFrame
shape = df.shape

print(f"Number of rows: {shape[0]}")
print(f"Number of columns: {shape[1]}")
```

Trang trình bày 3: DataFrame.dtypes

Các dtype thuộc tính trả về kiểu dữ liệu của từng cột trong DataFrame. Điều này rất cần thiết để hiểu bản chất dữ liệu của bạn và thực hiện hợp lý các hoạt động thích hợp.

```python
# Display the data types of each column
print(df.dtypes)

# Change the data type of a column
df['Age'] = df['Age'].astype(float)
print(df.dtypes)
```

Trang trình bày 4: DataFrame.index

Thuộc tính đại diện được xác định chỉ định cho hàng nhãn của DataFrame. Nó có thể được tùy chỉnh để sử dụng các định nghĩa mã hóa thay vì mặc định các số nguyên duy nhất.

```python
# Display the current index
print(df.index)

# Set a custom index
df.set_index('Name', inplace=True)
print(df)
print(df.index)
```

Trang trình bày 5: DataFrame.columns

Cột thuộc tính trả về cột của DataFrame. Nó có thể được sử dụng để truy cập, sửa đổi hoặc đổi tên các cột.

```python
# Display column names
print(df.columns)

# Rename columns
df.columns = ['Years', 'Location']
print(df)
```

Trang trình bày 6: DataFrame.values

Trả về thuộc tính giá trị về mảng NumPy chứa dữ liệu trong DataFrame. Điều này hữu ích khi bạn cần thực hiện các mảng yêu cầu hoạt động NumPy trí tuệ.

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

Trả thuộc tính kích thước về tổng số phần tử trong DataFrame. Nó có số nhân với số cột.

```python
# Get the size of the DataFrame
print(f"Total number of elements: {df.size}")

# Verify the calculation
total_elements = df.shape[0] * df.shape[1]
print(f"Calculated total elements: {total_elements}")
```

Trang trình bày 9: DataFrame.ndim

Thuộc tính ndim trả về số nguyên của DataFrame. Đối với DataFrame tiêu chuẩn, giá trị này sẽ luôn là 2 (hàng và cột).

```python
# Get the number of dimensions
print(f"Number of dimensions: {df.ndim}")

# Create a Series (1-dimensional) for comparison
series = pd.Series([1, 2, 3])
print(f"Number of dimensions in a Series: {series.ndim}")
```

Trang trình bày 10: DataFrame.axes

Thuộc tính trả lời sai lệch về danh sách nhãn trục và cột trục. Điều này có thể hữu ích để hiểu cấu trúc DataFrame của bạn.

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

Trang trình bày 12: Ví dụ thực tế: Phân tích dữ liệu thời gian

Vui lòng sử dụng thuộc tính DataFrame để phân tích chi tiết dữ liệu cho các thành phố khác nhau.

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

Vui lòng sử dụng thuộc tính DataFrame để phân tích hiệu suất của dữ liệu sinh viên.

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

Để biết thêm các chủ đề nâng cao và giải thích chuyên sâu về DataFrame thuộc tính của gấu trúc, hãy cân nhắc khám phá các tài nguyên sau:

1. Tài liệu chính thức về gấu trúc: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
2. "Những chú gấu trúc hiệu quả" của Matt Harrison: [https://github.com/mattharrison/effect\_pandas](https://github.com/mattharrison/effect_pandas)
3. "Python để phân tích dữ liệu" của Wes McKinney (người tạo ra gấu trúc): [https://wesmckinney.com/book/](https://wesmckinney.com/book/)

Tài nguyên này cung cấp thông tin toàn diện về gấu trúc và các khả năng của nó, giúp bạn nắm chắc các hoạt động và phân tích DataFrame.
