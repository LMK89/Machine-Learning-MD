## Sắp xếp dữ liệu với Pandas trong Python
Trang trình bày 1: Giới thiệu về sắp xếp dữ liệu với Pandas

Sắp xếp dữ liệu được làm sạch, cấu trúc và tạo phong phú dữ liệu luồng thành định dạng mong muốn để đưa ra quyết định tốt hơn trong thời gian ngắn hơn. Pandas là một thư viện Python mạnh mẽ cung cấp các công cụ phân tích dữ liệu và hiệu suất cao dữ liệu cấu trúc, dễ dàng sử dụng để xử lý dữ liệu có cấu trúc.

```python
import pandas as pd
import numpy as np

# Create a sample dataset
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']
}

# Create a DataFrame
df = pd.DataFrame(data)

print(df)
```

Slide 2: Load dữ liệu bằng Pandas

Gấu trúc có thể đọc dữ liệu từ nhiều định dạng tệp khác nhau, bao gồm cơ sở dữ liệu CSV, Excel, JSON và SQL. Vui lòng khám phá cách tải dữ liệu từ tệp CSV.

```python
# Load data from a CSV file
df = pd.read_csv('sample_data.csv')

# Display the first few rows
print(df.head())

# Get basic information about the dataset
print(df.info())
```

Trang trình bày 3: Khám phá dữ liệu

Sau khi tải dữ liệu, điều cần thiết là khám phá cấu trúc và nội dung của nó. Pandas cung cấp một số phương pháp để bạn hiểu dữ liệu nhanh hơn.

```python
# Display basic statistics of numerical columns
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Display unique values in a column
print(df['Category'].unique())

# Get the shape of the DataFrame
print(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")
```

Trang trình bày 4: Làm sạch dữ liệu - Xử lý thiếu giá trị

Việc thiếu các giá trị có thể gây ra những ảnh hưởng đáng kể đến phân tích của bạn. Pandas cung cấp nhiều phương pháp khác nhau để xử lý chúng một cách hiệu quả.

```python
# Fill missing values with a specific value
df['Column_A'].fillna(0, inplace=True)

# Fill missing values with the mean of the column
df['Column_B'].fillna(df['Column_B'].mean(), inplace=True)

# Drop rows with any missing values
df_cleaned = df.dropna()

# Drop columns with more than 50% missing values
df_cleaned = df.dropna(thresh=len(df) * 0.5, axis=1)

print(df_cleaned.isnull().sum())
```

Slide 5: Chuyển đổi dữ liệu - Đổi tên và sắp xếp lại các cột

Việc sắp xếp dữ liệu của bạn có thể cải thiện khả năng đọc và phân tích hiệu quả. Hãy khám phá cách đổi tên và sắp xếp lại các cột.

```python
# Rename columns
df = df.rename(columns={'old_name1': 'new_name1', 'old_name2': 'new_name2'})

# Reorder columns
desired_order = ['column3', 'column1', 'column2']
df = df[desired_order]

print(df.head())
```

Trang trình bày 6: Lọc và chọn lọc dữ liệu

Pandas cung cấp sức mạnh để lọc và lựa chọn cơ sở dữ liệu trên nhiều điều kiện khác nhau.

```python
# Select specific columns
selected_columns = df[['Name', 'Age', 'City']]

# Filter rows based on a condition
adults = df[df['Age'] >= 18]

# Filter using multiple conditions
target_group = df[(df['Age'] >= 25) & (df['City'] == 'New York')]

# Select data using .loc and .iloc
specific_data = df.loc[df['Name'] == 'Alice', 'Age']
first_two_rows = df.iloc[:2, :3]

print(target_group)
print(specific_data)
print(first_two_rows)
```

Slide 7: Phân nhóm và tổng hợp

Bạn được phép thực hiện các thao tác trên dữ liệu tập hợp. Hãy cùng khám phá các kỹ thuật và tổng hợp các nhóm.

```python
# Group by a column and calculate mean
mean_age_by_city = df.groupby('City')['Age'].mean()

# Multiple aggregations
agg_results = df.groupby('City').agg({
    'Age': ['mean', 'max', 'min'],
    'Salary': ['mean', 'median']
})

# Reset index to make grouped column a regular column
agg_results = agg_results.reset_index()

print(mean_age_by_city)
print(agg_results)
```

Trang trình bày 8: Hợp nhất và tham gia các DataFrames

Kết quả tổng hợp từ nhiều nguồn là một nhiệm vụ phổ biến trong việc sắp xếp dữ liệu. Pandas cung cấp nhiều phương pháp khác nhau để hợp nhất và tham gia DataFrames.

```python
# Create two sample DataFrames
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [2, 3, 4], 'Age': [25, 30, 35]})

# Inner join
inner_join = pd.merge(df1, df2, on='ID', how='inner')

# Left join
left_join = pd.merge(df1, df2, on='ID', how='left')

# Concatenate DataFrames vertically
df3 = pd.DataFrame({'ID': [5, 6], 'Name': ['David', 'Eve']})
concatenated = pd.concat([df1, df3], ignore_index=True)

print("Inner Join:")
print(inner_join)
print("\nLeft Join:")
print(left_join)
print("\nConcatenated:")
print(concatenated)
```

Slide 9: Xử lý dữ liệu lặp vòng lặp

Dữ liệu vòng lặp có thể làm sai lệch phân tích của bạn. Hãy khám phá cách xác định và loại bỏ các bản sao.

```python
# Create a sample DataFrame with duplicates
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Alice', 'David'],
    'Age': [25, 30, 35, 25, 40]
}
df = pd.DataFrame(data)

# Identify duplicate rows
print("Duplicate rows:")
print(df[df.duplicated()])

# Remove duplicate rows
df_unique = df.drop_duplicates()

# Remove duplicates based on specific columns
df_unique_name = df.drop_duplicates(subset=['Name'])

print("\nDataFrame after removing all duplicates:")
print(df_unique)
print("\nDataFrame after removing duplicates based on 'Name':")
print(df_unique_name)
```

Trang trình bày 10: Chuyển đổi loại dữ liệu

Đảm bảo các loại dữ liệu chính xác là rất quan trọng để phân tích chính xác và sử dụng bộ nhớ hiệu quả.

```python
# Create a sample DataFrame
df = pd.DataFrame({
    'ID': ['1', '2', '3'],
    'Value': ['10.5', '20.0', '30.7'],
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03']
})

print("Original DataFrame:")
print(df.dtypes)

# Convert 'ID' to integer
df['ID'] = df['ID'].astype(int)

# Convert 'Value' to float
df['Value'] = df['Value'].astype(float)

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

print("\nDataFrame after type conversion:")
print(df.dtypes)
print(df)
```

Slide 11: Xử lý dữ liệu phân loại

Đặc biệt yêu cầu xử lý loại dữ liệu. Pandas cung cấp các công cụ để mã hóa và vận hành các biến phân loại loại.

```python
# Create a sample DataFrame with categorical data
df = pd.DataFrame({
    'ID': [1, 2, 3, 4, 5],
    'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red']
})

# Convert 'Color' to categorical type
df['Color'] = df['Color'].astype('category')

# Get category codes
df['Color_Code'] = df['Color'].cat.codes

# One-hot encoding
color_dummies = pd.get_dummies(df['Color'], prefix='Color')

# Combine with original DataFrame
df_encoded = pd.concat([df, color_dummies], axis=1)

print("DataFrame with categorical data:")
print(df)
print("\nDataFrame with one-hot encoding:")
print(df_encoded)
```

Slide 12: Ví dụ thực tế: Phân tích đánh giá sản phẩm

Vui lòng phân tích tập dữ liệu đánh giá sản phẩm để hiểu rõ hơn về sự hài lòng của khách hàng.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (assuming we have a CSV file with product reviews)
df = pd.read_csv('product_reviews.csv')

# Clean the data
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df = df.dropna(subset=['Rating', 'Review'])

# Calculate average rating per product
avg_ratings = df.groupby('ProductID')['Rating'].mean().sort_values(ascending=False)

# Get the top 10 products by rating
top_10_products = avg_ratings.head(10)

# Plot the results
plt.figure(figsize=(12, 6))
top_10_products.plot(kind='bar')
plt.title('Top 10 Products by Average Rating')
plt.xlabel('Product ID')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Top 10 Products by Average Rating:")
print(top_10_products)
```

Trang trình bày 13: Ví dụ thực tế: Phân tích dữ liệu thời gian

Vui lòng phân tích lịch sử dữ liệu để xác định hướng và mô hình.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the weather dataset (assuming we have a CSV file with daily weather data)
df = pd.read_csv('weather_data.csv')

# Convert date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Set date as index
df.set_index('Date', inplace=True)

# Resample data to monthly average temperature
monthly_temp = df['Temperature'].resample('M').mean()

# Plot the monthly average temperature
plt.figure(figsize=(12, 6))
monthly_temp.plot()
plt.title('Monthly Average Temperature')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate year-over-year temperature change
yearly_temp = df['Temperature'].resample('Y').mean()
temp_change = yearly_temp.pct_change() * 100

print("Year-over-Year Temperature Change (%):")
print(temp_change)
```

Trang trình bày 14: Tài nguyên bổ sung

Để tìm hiểu và khám phá thêm về cách sắp xếp dữ liệu với Pandas, hãy xem xét các tài nguyên sau:

1. Tài liệu chính thức của Pandas: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
2. "Python để phân tích dữ liệu" của Wes McKinney (người tạo ra Pandas)
3. Hướng dẫn về Pandas của DataCamp: [https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python](https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python)
4. Hướng dẫn về Pandas của Python thực sự: [https://realpython.com/learning-paths/pandas-data-science/](https://realpython.com/learning-paths/pandas-data-science/)
5. Bài viết ArXiv: "Pandas: Bộ công cụ phân tích dữ liệu Python mạnh mẽ" của Wes McKinney (2018) - [https://arxiv.org/abs/1801.01323](https://arxiv.org/abs/1801.01323)

Tài nguyên này cung cấp các giải pháp chuyên sâu, ví dụ và các phương pháp hay nhất để sử dụng Pandas một kết quả hiệu quả trong quá trình xử lý dự án dữ liệu dữ liệu của bạn.
