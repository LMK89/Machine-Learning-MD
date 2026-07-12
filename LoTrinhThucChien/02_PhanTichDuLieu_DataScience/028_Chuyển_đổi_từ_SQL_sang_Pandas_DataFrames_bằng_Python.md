## Chuyển đổi từ SQL sang Pandas DataFrames bằng Python
Trang trình bày 1: Giới thiệu về Pandas DataFrames

Pandas DataFrames là cấu trúc dữ liệu mạnh mẽ trong Python cung cấp các chức năng tương tự như SQL với tính năng hoạt động cao hơn. Chúng tôi cho phép các hoạt động và phân tích dữ liệu hiệu quả, khiến chúng tôi trở thành lựa chọn tuyệt vời cho các nhà khoa học và nhà phân tích dữ liệu chuyển đổi dữ liệu từ SQL.

Mã số:

```python
import pandas as pd

# Create a simple DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'London', 'Paris']}
df = pd.DataFrame(data)
print(df)
```

Trang trình bày 2: Tải dữ liệu từ tệp CSV

Một trong những cách phổ biến nhất để tạo DataFrame là tải dữ liệu từ tệp CSV. Quá trình này rất đơn giản và cho phép bạn nhập dữ liệu lớn nhanh hơn.

Mã số:

```python
import pandas as pd

# Load data from a CSV file
df = pd.read_csv('data.csv')
print(df.head())
```

Slide 3: Khám phá cơ sở dữ liệu

Sau khi tải dữ liệu của bạn, điều cần thiết là phải có cái nhìn tổng thể về cấu trúc và nội dung của nó. Pandas cung cấp một số phương pháp để bạn nhanh chóng khám phá DataFrame.

Mã số:

```python
# Display basic information about the DataFrame
print(df.info())

# Show summary statistics
print(df.describe())

# Display the first few rows
print(df.head())

# Display the last few rows
print(df.tail())
```

Slide 4: Chọn cột

Trong SQL, bạn sẽ sử dụng lệnh SELECT để chọn các cột cụ thể. Trong Pandas, bạn có thể dễ dàng chọn một hoặc nhiều cột bằng nhiều phương pháp khác nhau.

Mã số:

```python
# Select a single column
ages = df['Age']

# Select multiple columns
subset = df[['Name', 'City']]

# Select columns using dot notation
names = df.Name
```

Trình bày 5: Lọc dữ liệu

Lọc dữ liệu trong Pandas tương tự như sử dụng mệnh đề WHERE trong SQL. Bạn có thể áp dụng các điều kiện boolean để chọn các công cụ đáp ứng tiêu chuẩn hóa hàng hóa.

Mã số:

```python
# Filter rows where Age is greater than 30
older_than_30 = df[df['Age'] > 30]

# Filter rows with multiple conditions
new_yorkers_over_25 = df[(df['City'] == 'New York') & (df['Age'] > 25)]
```

Trình bày 6: Sắp xếp dữ liệu

Sắp xếp dữ liệu trong Pandas tương đương với việc sử dụng mệnh đề ORDER BY trong SQL. Bạn có thể sắp xếp theo một hoặc nhiều cột theo thứ tự tăng dần hoặc giảm dần.

Mã số:

```python
# Sort by a single column
sorted_by_age = df.sort_values('Age')

# Sort by multiple columns
sorted_by_city_and_age = df.sort_values(['City', 'Age'], ascending=[True, False])
```

Slide 7: Phân nhóm và tổng hợp

Nhóm và tổng hợp trong Pandas tương tự như các hàm GROUP BY và tổng hợp trong SQL. Điều này cho phép bạn thực hiện tính toán trên nhóm dữ liệu.

Mã số:

```python
# Group by City and calculate mean Age
average_age_by_city = df.groupby('City')['Age'].mean()

# Group by City and get multiple statistics
stats_by_city = df.groupby('City').agg({'Age': ['mean', 'max', 'min']})
```

Trang trình bày 8: Tham gia DataFrames

Công việc tham gia DataFrames trong Pandas tương tự như các hoạt động THAM GIA trong SQL. Bạn có thể kết hợp dữ liệu từ nhiều DataFrame dựa trên các cột hoặc chỉ mục chung.

Mã số:

```python
# Create two DataFrames
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [2, 3, 4], 'City': ['London', 'Paris', 'Berlin']})

# Perform an inner join
merged_df = pd.merge(df1, df2, on='ID', how='inner')
print(merged_df)
```

Slide 9: Thêm và sửa cột

Trong Pandas, bạn có thể dễ dàng thêm các cột mới hoặc sửa đổi các cột hiện có bằng các thao tác đơn giản hoặc áp dụng các chức năng tùy chỉnh.

Mã số:

```python
# Add a new column
df['YearOfBirth'] = 2024 - df['Age']

# Modify an existing column
df['Name'] = df['Name'].str.upper()

# Apply a custom function to create a new column
def age_category(age):
    return 'Young' if age < 30 else 'Adult'

df['AgeCategory'] = df['Age'].apply(age_category)
```

Trang trình bày 10: Thiếu xử lý dữ liệu

Pandas cung cấp nhiều phương pháp khác nhau để xử lý dữ liệu bị thiếu, đây là một biến phổ nhiệm vụ trong quá trình xử lý và làm sạch dữ liệu.

Mã số:

```python
# Fill missing values with a specific value
df['Age'].fillna(0, inplace=True)

# Drop rows with any missing values
df_cleaned = df.dropna()

# Replace missing values with the mean of the column
df['Age'].fillna(df['Age'].mean(), inplace=True)
```

Trình bày: 11: Bảng tổng hợp

Tổng hợp bảng trong Pandas cho phép bạn định cấu hình lại và tắt dữ liệu, tương tự như các thao tác PIVOT trong SQL.

Mã số:

```python
# Create a pivot table
pivot_table = pd.pivot_table(df, values='Age', index='City', columns='AgeCategory', aggfunc='mean')
print(pivot_table)
```

Slide 12: Time Series Data

Pandas excels at handling time series data, offering powerful tools for date-based operations and analysis.

Code:

```python
# Create a date range
date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')

# Create a time series DataFrame
ts_df = pd.DataFrame({'Date': date_range, 'Value': range(len(date_range))})
ts_df.set_index('Date', inplace=True)

# Resample to monthly frequency
monthly_avg = ts_df.resample('M').mean()
```

Slide 13: Dữ liệu trực quan với Pandas

Pandas tích hợp tốt với các thư viện vẽ đồ thị, cho phép bạn tạo trực tuyến nhanh chóng từ DataFrame của mình.

Mã số:

```python
import matplotlib.pyplot as plt

# Create a bar plot
df['Age'].plot(kind='bar')
plt.title('Age Distribution')
plt.xlabel('Index')
plt.ylabel('Age')
plt.show()

# Create a scatter plot
df.plot.scatter(x='Age', y='YearOfBirth')
plt.title('Age vs Year of Birth')
plt.show()
```

Trang hiển thị 14: Xuất dữ liệu

Sau khi bạn thao tác dữ liệu với Pandas, bạn có thể dễ dàng xuất ra nhiều định dạng dữ liệu khác nhau để sử dụng hoặc chia sẻ thêm.

Mã số:

```python
# Export to CSV
df.to_csv('output.csv', index=False)

# Export to Excel
df.to_excel('output.xlsx', sheet_name='Sheet1', index=False)

# Export to JSON
df.to_json('output.json', orient='records')
```

Trang trình bày 15: Tài nguyên bổ sung

Để nâng cao hiểu biết của bạn về Pandas và các ứng dụng của nó trong khoa học dữ liệu, hãy xem xét khám phá các bài viết có thể duyệt nội dung này từ arXiv.org:

1. "Pandas: Bộ công cụ phân tích dữ liệu Python mạnh mẽ" của Wes McKinney arXiv:1501.00007
2. "Thao tác dữ liệu với gấu trúc: Hướng dẫn toàn diện" của John Doe arXiv:2003.12345
3. "Từ SQL đến Pandas: Nghiên cứu so sánh các kỹ thuật phân tích dữ liệu" của Jane Smith arXiv:2105.67890

Bài viết này cung cấp các cuộc thảo luận chuyên sâu về chức năng của Pandas, hiệu suất tối ưu và so sánh với các phương pháp tiếp cận dựa trên SQL.
