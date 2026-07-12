## Giới thiệu về Làm sạch dữ liệu bằng Pandas và Python

Slide 1: Giới thiệu về làm sạch dữ liệu với Pandas

Làm sạch dữ liệu là một bước quan trọng trong phân tích dữ liệu, đảm bảo dữ liệu chính xác, nhất quán và sẵn sàng để phân tích. Pandas, một thư viện Python mạnh mẽ, cung cấp nhiều công cụ và chức năng khác nhau để xử lý các tác vụ dọn dẹp dữ liệu một cách hiệu quả.

Mã số:

```python
import pandas as pd
```

Slide 2: Handling Missing Data

Missing data is a common issue in datasets. Pandas provides several methods to handle missing values, such as dropping rows or columns, filling with a specific value, or using interpolation techniques.

Code:

```python
# Drop rows with missing values
df.dropna(inplace=True)

# Fill missing values with a specific value
df.fillna(0, inplace=True)

# Fill missing values with the mean of the column
df['column_name'] = df['column_name'].fillna(df['column_name'].mean())
```

Slide 3: Loại bỏ trùng lặp

Dữ liệu trùng lặp có thể dẫn đến phân tích không chính xác và kết quả sai lệch. Pandas cung cấp các phương pháp để xác định và xóa các hàng hoặc cột trùng lặp khỏi DataFrame.

Mã số:

```python
# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Remove duplicate rows based on specific columns
df.drop_duplicates(subset=['column1', 'column2'], inplace=True)
```

Slide 4: Chuyển đổi dữ liệu

Chuyển đổi dữ liệu liên quan đến việc chuyển đổi dữ liệu sang định dạng phù hợp hơn để phân tích. Pandas cung cấp các chức năng để thực hiện các hoạt động như chuyển đổi kiểu dữ liệu, thao tác chuỗi và xử lý ngày/giờ.

Mã số:

```python
# Convert data types
df['column_name'] = df['column_name'].astype('int')

# String manipulation
df['column_name'] = df['column_name'].str.lower()

# Date/time handling
df['date_column'] = pd.to_datetime(df['date_column'])
```

Slide 5: Xử lý các ngoại lệ

Các ngoại lệ có thể tác động đáng kể đến kết quả phân tích. Pandas cung cấp nhiều kỹ thuật khác nhau để xác định và xử lý các ngoại lệ, chẳng hạn như sử dụng các phương pháp thống kê hoặc áp dụng các quy tắc dành riêng cho từng miền.

Mã số:

```python
# Identify outliers using z-scores
z_scores = np.abs(df['column_name'] - df['column_name'].mean()) / df['column_name'].std()
outliers = df[z_scores > 3]

# Replace outliers with a specific value
df.loc[z_scores > 3, 'column_name'] = df['column_name'].median()
```

Trang trình bày 6: Lọc dữ liệu

Lọc dữ liệu là quá trình chọn một tập hợp con dữ liệu dựa trên các tiêu chí cụ thể. Pandas cung cấp khả năng lọc mạnh mẽ bằng cách sử dụng lập chỉ mục boolean và các câu lệnh có điều kiện.

Mã số:

```python
# Filter rows based on a condition
filtered_df = df[df['column_name'] > 10]

# Filter rows based on multiple conditions
filtered_df = df[(df['column1'] > 5) & (df['column2'] == 'value')]
```

Slide 7: Xử lý dữ liệu phân loại

Dữ liệu phân loại đại diện cho các danh mục hoặc nhóm riêng biệt. Pandas cung cấp các công cụ để làm việc với dữ liệu phân loại, chẳng hạn như mã hóa các biến phân loại và thực hiện các hoạt động như nhóm và tổng hợp.

Mã số:

```python
# Convert a column to categorical data type
df['column_name'] = df['column_name'].astype('category')

# Encode categorical data
encoded_df = pd.get_dummies(df, columns=['column_name'])
```

Slide 8: Hợp nhất và nối dữ liệu

Hợp nhất và nối dữ liệu từ nhiều nguồn là một nhiệm vụ phổ biến trong phân tích dữ liệu. Pandas cung cấp các phương pháp để kết hợp các tập dữ liệu dựa trên các cột hoặc chỉ mục chung.

Mã số:

```python
# Merge two DataFrames based on a common column
merged_df = pd.merge(df1, df2, on='common_column')

# Join two DataFrames based on indexes
joined_df = df1.join(df2, how='inner')
```

Slide 9: Định hình lại dữ liệu

Định hình lại dữ liệu liên quan đến việc chuyển đổi cấu trúc của DataFrame, chẳng hạn như dữ liệu xoay vòng hoặc không xoay vòng. Pandas cung cấp các chức năng như `melt` và `pivot` để định hình lại dữ liệu nhằm phân tích tốt hơn.

Mã số:

```python
# Unpivot (melt) data
melted_df = pd.melt(df, id_vars=['column1', 'column2'], var_name='variable', value_name='value')

# Pivot data
pivoted_df = df.pivot(index='column1', columns='column2', values='column3')
```

Trang trình bày 10: Tính toán dữ liệu

Việc tính toán dữ liệu là quá trình thay thế dữ liệu bị thiếu bằng các giá trị thay thế. Pandas cung cấp nhiều kỹ thuật quy định khác nhau, chẳng hạn như quy định trung bình, trung bình hoặc chế độ, cũng như các phương pháp nâng cao hơn như quy mô hồi quy.

Mã số:

```python
# Mean imputation
df['column_name'] = df['column_name'].fillna(df['column_name'].mean())

# Regression imputation
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
df['column_name'] = df['column_name'].fillna(regressor.predict(X_test))
```

Slide 11: Chuẩn hóa dữ liệu

Chuẩn hóa dữ liệu là một kỹ thuật được sử dụng để thay đổi tỷ lệ dữ liệu về một phạm vi chung, thường là từ 0 đến 1 hoặc -1 và 1. Điều này có thể hữu ích cho một số thuật toán học máy nhất định hoặc khi xử lý các quy mô dữ liệu khác nhau.

Mã số:

```python
# Min-max normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Standardization (z-score normalization)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
standardized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
```

Trang trình bày 12: Xác thực dữ liệu

Xác thực dữ liệu là quá trình đảm bảo rằng dữ liệu tuân thủ các quy tắc, ràng buộc hoặc định dạng cụ thể. Pandas cung cấp các phương pháp để xác thực dữ liệu và xử lý các vi phạm, chẳng hạn như phát sinh lỗi hoặc áp dụng các chức năng tùy chỉnh.

Mã số:

```python
# Validate data types
df = df.astype({'column1': 'int', 'column2': 'float'})

# Apply custom validation function
def validate_age(age):
    if age < 0 or age > 120:
        raise ValueError('Invalid age')
    return age

df['age'] = df['age'].apply(validate_age)
```

Slide 13: Hồ sơ dữ liệu

Hồ sơ dữ liệu liên quan đến việc tóm tắt và hiểu các đặc điểm của tập dữ liệu. Pandas cung cấp nhiều phương pháp khác nhau để tạo số liệu thống kê mô tả, xác định loại dữ liệu và phát hiện các giá trị hoặc giá trị ngoại lệ bị thiếu.

Mã số:

```python
# Generate descriptive statistics
df.describe()

# Identify data types
df.dtypes

# Detect missing values
df.isnull().sum()

# Detect duplicates
df.duplicated().sum()
```

Slide 14: Kết luận

Làm sạch dữ liệu là một bước thiết yếu trong quá trình phân tích dữ liệu. Pandas cung cấp bộ công cụ mạnh mẽ và linh hoạt để xử lý các tác vụ làm sạch dữ liệu khác nhau, từ xử lý dữ liệu bị thiếu và trùng lặp đến chuyển đổi, lọc và định hình lại dữ liệu. Bằng cách nắm vững các kỹ thuật này, bạn có thể đảm bảo dữ liệu của mình chính xác, nhất quán và sẵn sàng để phân tích có ý nghĩa.