## Pandas để sắp xếp và phân tích dữ liệu
Slide 1: Giới thiệu về Pandas

Pandas là một thư viện Python mạnh mẽ để thao tác và phân tích dữ liệu. Nó cung cấp dữ liệu cấu trúc như DataFrame và Series, cho phép dữ liệu cấu hình xử lý hiệu quả. Pandas đơn giản hóa các tác vụ như làm sạch, chuyển đổi, hợp nhất và phân tích dữ liệu, khiến nó trở thành một công cụ thiết yếu cho các nhà khoa học và phân tích dữ liệu.

```python
import pandas as pd

# Create a simple DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'London', 'Paris']}
df = pd.DataFrame(data)

print(df)
```

Trang trình bày 2: Tạo DataFrames

DataFrames là dữ liệu chính cấu trúc trong Pandas. Chúng biểu diễn bảng dạng dữ liệu với các hàng và cột được gắn nhãn. Bạn có thể tạo DataFrames từ nhiều nguồn dữ liệu khác nhau, bao gồm từ điển, danh sách hoặc tệp bên ngoài.

```python
# Create a DataFrame from a dictionary
data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]}
df = pd.DataFrame(data)

# Create a DataFrame from a list of lists
data = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
df = pd.DataFrame(data, columns=['A', 'B', 'C'])

print(df)
```

Trang trình bày 3: Tải dữ liệu từ nguồn bên ngoài

Pandas cung cấp các chức năng đọc dữ liệu từ nhiều định dạng tệp khác nhau, bao gồm CSV, Excel và JSON. Điều này cho phép bạn dễ dàng nhập dữ liệu từ các nguồn bên ngoài vào môi trường Python để phân tích.

```python
# Read data from a CSV file
df_csv = pd.read_csv('data.csv')

# Read data from an Excel file
df_excel = pd.read_excel('data.xlsx')

# Read data from a JSON file
df_json = pd.read_json('data.json')

print(df_csv.head())
```

Trang trình bày 4: Kiểm tra dữ liệu

Sau khi tải dữ liệu, điều quan trọng phải kiểm tra nó để hiểu cấu trúc và nội dung của nó. Pandas cung cấp một số phương pháp để bạn nhanh chóng kiểm tra DataFrame.

```python
# Display the first few rows
print(df.head())

# Get basic information about the DataFrame
print(df.info())

# Display summary statistics
print(df.describe())

# Check the shape of the DataFrame
print(df.shape)
```

Slide 5: Thiếu dữ liệu xử lý

Thiếu dữ liệu là một phổ biến vấn đề trong bộ dữ liệu trong thế giới thực. Pandas cung cấp các phương pháp để xác định và xử lý các giá trị nhưng thiếu một kết quả hiệu quả.

```python
# Create a DataFrame with missing values
df = pd.DataFrame({'A': [1, 2, None, 4], 'B': [5, None, 7, 8]})

# Fill missing values with a specific value
df_filled = df.fillna(0)

# Drop rows with missing values
df_dropped = df.dropna()

print("Original DataFrame:")
print(df)
print("\nFilled DataFrame:")
print(df_filled)
print("\nDropped DataFrame:")
print(df_dropped)
```

Trang trình bày 6: Lựa chọn và cài đặt dữ liệu chỉ mục

Pandas cung cấp sức mạnh để lựa chọn và cài đặt dữ liệu chỉ mục trong DataFrame. Bạn có thể truy cập dữ liệu theo nhãn, vị trí hoặc cài đặt boolean mục mục.

```python
# Create a sample DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]},
                  index=['x', 'y', 'z'])

# Select a column
print(df['A'])

# Select multiple columns
print(df[['A', 'B']])

# Select rows by label
print(df.loc['x'])

# Select rows and columns by position
print(df.iloc[0, 1])

# Boolean indexing
print(df[df['A'] > 1])
```

Trang trình bày 7: Chuyển đổi dữ liệu

Chuyển đổi dữ liệu là một bước quan trọng trong phân tích dữ liệu. Pandas cung cấp nhiều phương pháp khác nhau để sửa đổi và định cấu hình lại dữ liệu của bạn.

```python
# Create a sample DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Apply a function to a column
df['C'] = df['A'].apply(lambda x: x * 2)

# Rename columns
df = df.rename(columns={'A': 'X', 'B': 'Y'})

# Add a new column based on existing ones
df['Z'] = df['X'] + df['Y']

print(df)
```

Slide 8: Phân nhóm và tổng hợp

Nhóm và tổng hợp là những kỹ năng mạnh mẽ để tắt dữ liệu. Pandas giúp dễ dàng nhóm dữ liệu theo một hoặc nhiều cột và áp dụng các hàm tổng hợp.

```python
# Create a sample DataFrame
df = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B', 'A'],
    'Value': [10, 20, 30, 40, 50]
})

# Group by Category and calculate mean
grouped = df.groupby('Category')['Value'].mean()

# Group by Category and apply multiple aggregations
agg_funcs = {'Value': ['mean', 'sum', 'count']}
result = df.groupby('Category').agg(agg_funcs)

print("Grouped mean:")
print(grouped)
print("\nMultiple aggregations:")
print(result)
```

Trang trình bày 9: Hợp nhất và tham gia các DataFrames

Kết quả tổng hợp từ nhiều nguồn là một nhiệm vụ phổ biến trong phân tích dữ liệu. Pandas cung cấp nhiều phương pháp khác nhau để hợp nhất và kết nối các DataFrames dựa trên chung các cột hoặc chỉ mục.

```python
# Create two sample DataFrames
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value': [4, 5, 6]})

# Merge DataFrames on the 'key' column
merged = pd.merge(df1, df2, on='key', how='outer')

# Join DataFrames based on index
df3 = pd.DataFrame({'value': [7, 8, 9]}, index=['A', 'B', 'E'])
joined = df1.set_index('key').join(df3, how='outer')

print("Merged DataFrame:")
print(merged)
print("\nJoined DataFrame:")
print(joined)
```

Trang trình bày 10: Bảng tổng hợp và định lại dữ liệu

Bảng tổng hợp rất hữu ích cho việc tóm tắt và phân tích dữ liệu. Pandas cung cấp các chức năng để tạo bảng tổng hợp và định cấu hình lại dữ liệu giữa định dạng rộng và dài.

```python
# Create a sample DataFrame
df = pd.DataFrame({
    'Date': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
    'Product': ['A', 'B', 'A', 'B'],
    'Sales': [100, 150, 120, 180]
})

# Create a pivot table
pivot = df.pivot_table(values='Sales', index='Date', columns='Product', aggfunc='sum')

# Melt the DataFrame from wide to long format
melted = pd.melt(df, id_vars=['Date'], value_vars=['Sales'], var_name='Metric', value_name='Value')

print("Pivot Table:")
print(pivot)
print("\nMelted DataFrame:")
print(melted)
```

Trang trình bày 11: Phân tích chuỗi thời gian

Pandas vượt trội trong việc xử lý thời gian chuỗi dữ liệu. Nó cung cấp các công cụ mạnh mẽ để làm việc theo ngày, giờ và các hoạt động dựa trên thời gian.

```python
# Create a time series DataFrame
dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
ts = pd.DataFrame({'Value': range(len(dates))}, index=dates)

# Resample to weekly frequency
weekly = ts.resample('W').sum()

# Shift the time series
shifted = ts.shift(periods=2)

# Calculate rolling mean
rolling_mean = ts.rolling(window=3).mean()

print("Original Time Series:")
print(ts)
print("\nWeekly Resampled:")
print(weekly)
print("\nShifted Time Series:")
print(shifted)
print("\nRolling Mean:")
print(rolling_mean)
```

Slide 12: Dữ liệu trực quan với Pandas

Pandas tích hợp tốt với matplotlib, cho phép bạn tạo trực tuyến hóa nhanh chóng trực tiếp từ DataFrames của mình.

```python
import matplotlib.pyplot as plt

# Create a sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [2, 4, 6, 8, 10],
    'C': [3, 6, 9, 12, 15]
})

# Create a line plot
df.plot(kind='line')
plt.title('Line Plot')
plt.show()

# Create a bar plot
df.plot(kind='bar')
plt.title('Bar Plot')
plt.show()

# Create a scatter plot
df.plot(kind='scatter', x='A', y='B')
plt.title('Scatter Plot')
plt.show()
```

Trang trình bày 13: Ví dụ thực tế: Phân tích dữ liệu thời gian

Vui lòng phân tích dữ liệu chi tiết để chứng minh khả năng của Pandas trong tình huống thực tế.

```python
# Load weather data
weather_data = pd.read_csv('weather_data.csv')

# Display basic information about the dataset
print(weather_data.info())

# Calculate average temperature by month
weather_data['Date'] = pd.to_datetime(weather_data['Date'])
monthly_temp = weather_data.groupby(weather_data['Date'].dt.to_period('M'))['Temperature'].mean()

# Find the hottest and coldest days
hottest_day = weather_data.loc[weather_data['Temperature'].idxmax()]
coldest_day = weather_data.loc[weather_data['Temperature'].idxmin()]

print("\nAverage Monthly Temperature:")
print(monthly_temp)
print("\nHottest Day:")
print(hottest_day)
print("\nColdest Day:")
print(coldest_day)
```

Trang trình bày 14: Ví dụ thực tế: Quản lý tồn tại kho sản phẩm

Ví dụ này minh họa cách sử dụng Pandas để quản lý và phân tích dữ liệu kiểm tra sản phẩm.

```python
# Create a sample inventory DataFrame
inventory = pd.DataFrame({
    'Product': ['Widget A', 'Widget B', 'Widget C', 'Widget D'],
    'Quantity': [100, 150, 200, 75],
    'Price': [10.99, 15.99, 8.99, 12.99],
    'Category': ['Electronics', 'Tools', 'Electronics', 'Tools']
})

# Calculate total value of inventory
inventory['Total Value'] = inventory['Quantity'] * inventory['Price']

# Find products with low stock (less than 100)
low_stock = inventory[inventory['Quantity'] < 100]

# Calculate average price by category
avg_price_by_category = inventory.groupby('Category')['Price'].mean()

print("Inventory Summary:")
print(inventory)
print("\nLow Stock Products:")
print(low_stock)
print("\nAverage Price by Category:")
print(avg_price_by_category)
```

Trang trình bày 15: Tài nguyên bổ sung

Để mở rộng hơn nữa các kiến ​​thức của bạn về Pandas và phân tích dữ liệu bằng Python, hãy xem xét khám phá các tài nguyên sau:

1. Tài liệu chính thức về Pandas: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
2. "Python to parsing data" của Wes McKinney (người tạo ra Pandas)
3. Hướng dẫn về Pandas của DataCamp: [https://www.datacamp.com/courses/data-manipulation-with-pandas](https://www.datacamp.com/courses/data-manipulation-with-pandas)
4. "Những chú gấu trúc biểu tượng" của Matt Harrison (có trên GitHub)
5. Khóa học vi mô về gấu trúc của Kaggle: [https://www.kaggle.com/learn/pandas](https://www.kaggle.com/learn/pandas)

Hãy nhớ thực hành thường xuyên với các bộ dữ liệu trong thế giới thực để củng cố kỹ năng Pandas của bạn.
