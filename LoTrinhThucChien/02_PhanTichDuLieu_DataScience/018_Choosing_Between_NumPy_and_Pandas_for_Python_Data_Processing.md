## Lựa chọn giữa NumPy và Pandas để xử lý dữ liệu Python
Trang trình bày 1: Nguyên tắc cơ bản về NumPy - Hoạt động mảng

Mảng NumPy cung cấp khả năng lưu trữ và vận hành hiệu quả cho dữ liệu số thông qua việc phân bổ bộ nhớ liền kề. Không giống như danh sách Python, mảng NumPy thực thi các kiểu dữ liệu đồng nhất, cho phép các hoạt động được vector hóa giúp tăng đáng kể hiệu suất tính toán cho các phép tính toán học.

```python
import numpy as np

# Creating arrays and basic operations
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([6, 7, 8, 9, 10])

# Vectorized operations - no explicit loops needed
addition = arr1 + arr2
multiplication = arr1 * arr2
power = arr1 ** 2

print(f"Addition: {addition}")
print(f"Multiplication: {multiplication}")
print(f"Power: {power}")

# Output:
# Addition: [ 7  9 11 13 15]
# Multiplication: [ 6 14 24 36 50]
# Power: [ 1  4  9 16 25]
```

Trang trình bày 2: Thông tin cơ bản về chuỗi Pandas và DataFrame

Pandas giới thiệu hai cấu trúc dữ liệu chính: Chuỗi (1 chiều) và DataFrame (2 chiều), cả hai đều được xây dựng dựa trên mảng NumPy. Các cấu trúc này bổ sung thêm tính năng lập chỉ mục, căn chỉnh dữ liệu và xử lý các khả năng giá trị bị thiếu cần thiết cho việc phân tích dữ liệu.

```python
import pandas as pd

# Creating Series and DataFrame
series = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
df = pd.DataFrame({
    'numbers': [1, 2, 3, 4],
    'letters': ['a', 'b', 'c', 'd'],
    'values': [1.1, 2.2, 3.3, 4.4]
})

print("Series:\n", series)
print("\nDataFrame:\n", df)

# Accessing data
print("\nAccessing column:", df['numbers'])
print("\nFiltering:", df[df['values'] > 2.5])
```

Trang trình bày 3: Phân tích hiệu suất NumPy

Hiểu được sự khác biệt về hiệu suất giữa các hoạt động NumPy và Python thuần túy là rất quan trọng để tối ưu hóa. Các hoạt động được vector hóa của NumPy thực thi ở cấp độ C, tránh chi phí vòng lặp của Python và tăng tốc đáng kể cho các phép tính số quy mô lớn.

```python
import numpy as np
import time

# Performance comparison: NumPy vs Python lists
size = 1000000

# Python list operation
python_list = list(range(size))
start_time = time.time()
python_result = [x**2 for x in python_list]
python_time = time.time() - start_time

# NumPy operation
numpy_array = np.arange(size)
start_time = time.time()
numpy_result = numpy_array**2
numpy_time = time.time() - start_time

print(f"Python time: {python_time:.4f} seconds")
print(f"NumPy time: {numpy_time:.4f} seconds")
print(f"Speed improvement: {python_time/numpy_time:.2f}x")
```

Trang trình bày 4: Làm sạch và tiền xử lý dữ liệu của Pandas

Làm sạch dữ liệu là một bước quan trọng trong bất kỳ quy trình phân tích dữ liệu nào. Pandas cung cấp các công cụ toàn diện để xử lý các giá trị bị thiếu, loại bỏ trùng lặp và chuyển đổi định dạng dữ liệu, khiến nó không thể thiếu trong việc chuẩn bị các bộ dữ liệu trong thế giới thực.

```python
import pandas as pd
import numpy as np

# Create sample dataset with issues
df = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-02', None, '2023-01-03'],
    'value': [1.0, np.nan, 3.0, 3.0],
    'category': ['A', 'B', 'B', 'A']
})

# Clean the data
cleaned_df = df.copy()
cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])  # Convert to datetime
cleaned_df['value'].fillna(cleaned_df['value'].mean(), inplace=True)  # Fill NaN
cleaned_df.dropna(subset=['date'], inplace=True)  # Remove rows with missing dates
cleaned_df.drop_duplicates(subset=['value', 'category'], inplace=True)  # Remove duplicates

print("Original DataFrame:\n", df)
print("\nCleaned DataFrame:\n", cleaned_df)
```

Trang trình bày 5: Hoạt động ma trận NumPy

Các phép toán ma trận tạo thành xương sống của các thuật toán máy tính và máy học khoa học. NumPy cung cấp các triển khai ma trận được tối ưu hóa cao, tận dụng các thư viện BLAS và LAPACK hiệu quả để tính toán đại số tuyến tính.

```python
import numpy as np

# Create matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix operations
matrix_product = np.dot(A, B)  # Matrix multiplication
eigenvalues, eigenvectors = np.linalg.eig(A)  # Eigendecomposition
inverse = np.linalg.inv(A)  # Matrix inverse
determinant = np.linalg.det(A)  # Determinant

print("Matrix Product:\n", matrix_product)
print("\nEigenvalues:", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)
print("\nInverse:\n", inverse)
print("\nDeterminant:", determinant)
```

Trang trình bày 6: Tổng hợp dữ liệu nâng cao của Pandas

Pandas cung cấp khả năng nhóm và tổng hợp mạnh mẽ thông qua hoạt động GroupBy. Chức năng này cho phép phân tích dữ liệu phức tạp bằng cách chia dữ liệu thành các nhóm, áp dụng các hàm và kết hợp kết quả một cách hiệu quả để phân tích sâu sắc.

```python
import pandas as pd
import numpy as np

# Create sample sales data
sales_data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', '2023-12-31', freq='D'),
    'product': np.random.choice(['A', 'B', 'C'], size=365),
    'region': np.random.choice(['North', 'South', 'East', 'West'], size=365),
    'sales': np.random.normal(1000, 200, size=365),
    'units': np.random.randint(10, 100, size=365)
})

# Complex aggregation
agg_results = sales_data.groupby(['product', 'region']).agg({
    'sales': ['mean', 'sum', 'std'],
    'units': ['count', 'max']
}).round(2)

# Calculate monthly trends
monthly_trends = sales_data.set_index('date').resample('M').agg({
    'sales': 'sum',
    'units': 'mean'
}).round(2)

print("Aggregated Results:\n", agg_results)
print("\nMonthly Trends:\n", monthly_trends)
```

Trang trình bày 7: Phát sóng và Vector hóa NumPy

Broadcasting là một cơ chế mạnh mẽ cho phép NumPy thực hiện các thao tác trên các mảng có hình dạng khác nhau một cách hiệu quả. Hiểu các quy tắc phát sóng là rất quan trọng để viết các phép tính số được tối ưu hóa mà không có vòng lặp rõ ràng.

```python
import numpy as np

# Broadcasting examples
array_2d = np.array([[1, 2, 3],
                     [4, 5, 6]])  # Shape: (2, 3)
vector = np.array([10, 20, 30])   # Shape: (3,)

# Broadcasting in action
broadcast_add = array_2d + vector
broadcast_multiply = array_2d * vector

# Complex broadcasting example
coords = np.array([[0, 0, 0],
                  [1, 1, 1],
                  [2, 2, 2]])  # Shape: (3, 3)
weights = np.array([1, 2, 3]).reshape(3, 1)  # Shape: (3, 1)
weighted_coords = coords * weights  # Shape: (3, 3)

print("Original array:\n", array_2d)
print("\nBroadcast addition:\n", broadcast_add)
print("\nBroadcast multiplication:\n", broadcast_multiply)
print("\nWeighted coordinates:\n", weighted_coords)
```

Trang trình bày 8: Phân tích chuỗi thời gian của Pandas

Phân tích chuỗi thời gian là nền tảng của khoa học dữ liệu và Pandas vượt trội trong việc xử lý dữ liệu tạm thời với chức năng ngày giờ phức tạp, hoạt động lấy mẫu lại và tính toán cửa sổ cuộn.

```python
import pandas as pd
import numpy as np

# Generate time series data
dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
ts_data = pd.Series(np.random.normal(0, 1, len(dates)), index=dates)

# Time series operations
rolling_mean = ts_data.rolling(window=7).mean()  # 7-day moving average
monthly_data = ts_data.resample('M').agg(['mean', 'std'])
year_to_date = ts_data.cumsum()

# Calculate seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_data, period=30, model='additive')

print("Original Time Series Head:\n", ts_data.head())
print("\nRolling Mean Head:\n", rolling_mean.head())
print("\nMonthly Statistics:\n", monthly_data)

# Plot components (commented out as per requirements)
# decomposition.plot()
```

Slide 9: Ứng dụng thực tế - Phân tích danh mục đầu tư

Việc triển khai này thể hiện một ứng dụng thực tế kết hợp NumPy và Pandas để phân tích danh mục đầu tư tài chính, cho thấy cả hai thư viện bổ sung cho nhau như thế nào trong các tình huống thực tế.

```python
import numpy as np
import pandas as pd

# Generate sample stock data
np.random.seed(42)
dates = pd.date_range('2022-01-01', '2023-12-31', freq='B')
stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
prices = pd.DataFrame(
    np.random.randn(len(dates), len(stocks)).cumsum(axis=0) + 100,
    index=dates,
    columns=stocks
)

# Calculate daily returns
returns = prices.pct_change()

# Portfolio analysis
weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal weights
portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized return
portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
sharpe_ratio = portfolio_return / portfolio_vol

print("Portfolio Metrics:")
print(f"Annual Return: {portfolio_return:.4f}")
print(f"Annual Volatility: {portfolio_vol:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
```

Slide 10: Ứng dụng thực tế - Phân tích giỏ thị trường

Việc triển khai phân tích giỏ thị trường bằng Pandas thể hiện sức mạnh của thư viện trong việc xử lý dữ liệu phân loại và tính toán các mối quan hệ phức tạp giữa các mục trong bộ dữ liệu giao dịch.

```python
import pandas as pd
import numpy as np
from itertools import combinations

# Generate sample transaction data
transactions = pd.DataFrame({
    'transaction_id': np.repeat(range(1000), 3),
    'item': np.random.choice(['bread', 'milk', 'eggs', 'cheese', 'butter'], 3000)
})

# Create item pairs and calculate support
def calculate_support(transactions):
    # Convert to binary purchase matrix
    purchase_matrix = pd.crosstab(transactions['transaction_id'], transactions['item'])

    # Calculate item pair frequencies
    n_transactions = len(purchase_matrix)
    item_pairs = []
    support_values = []

    for item1, item2 in combinations(purchase_matrix.columns, 2):
        both_purchased = purchase_matrix[purchase_matrix[item1] & purchase_matrix[item2]].shape[0]
        support = both_purchased / n_transactions
        item_pairs.append(f"{item1} -> {item2}")
        support_values.append(support)

    return pd.DataFrame({
        'item_pair': item_pairs,
        'support': support_values
    }).sort_values('support', ascending=False)

results = calculate_support(transactions)
print("Top 5 Item Pairs by Support:\n", results.head())
```

Trang trình bày 11: Kỹ thuật tối ưu hóa hiệu suất NumPy

Các kỹ thuật tối ưu hóa nâng cao trong NumPy có thể cải thiện đáng kể hiệu quả tính toán thông qua quản lý bộ nhớ, vector hóa và các hoạt động mảng thích hợp nhằm giảm thiểu việc tạo mảng tạm thời.

```python
import numpy as np
import time

# Optimization example: comparing different approaches
size = 1000000

# Inefficient approach with temporary arrays
def inefficient_calculation(arr):
    temp1 = arr * 2
    temp2 = temp1 + 3
    return temp2 ** 2

# Optimized approach without temporary arrays
def efficient_calculation(arr):
    return np.square(np.add(np.multiply(arr, 2), 3))

# Memory pre-allocation example
def optimized_growth():
    result = np.zeros(size, dtype=np.float64)
    for i in range(size):
        result[i] = i * 2
    return result

# Benchmark
arr = np.random.rand(size)

start = time.time()
result1 = inefficient_calculation(arr)
time1 = time.time() - start

start = time.time()
result2 = efficient_calculation(arr)
time2 = time.time() - start

print(f"Inefficient approach time: {time1:.4f} seconds")
print(f"Efficient approach time: {time2:.4f} seconds")
print(f"Speed improvement: {time1/time2:.2f}x")
```

Trang trình bày 12: Lập chỉ mục và lựa chọn nâng cao của Pandas

Các kỹ thuật lập chỉ mục nâng cao trong Pandas cho phép thực hiện các hoạt động lọc và lựa chọn dữ liệu phức tạp, rất quan trọng đối với các nhiệm vụ phân tích dữ liệu phức tạp và kỹ thuật tính năng trong quy trình học máy.

```python
import pandas as pd
import numpy as np

# Create complex dataset
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'value': np.random.randn(1000),
    'flag': np.random.choice([True, False], 1000),
    'group': np.random.randint(1, 5, 1000)
})

# Advanced indexing examples
mask = (df['value'] > 0) & (df['flag']) & (df['category'].isin(['A', 'B']))
filtered = df.loc[mask]

# Multi-level indexing
df.set_index(['date', 'category'], inplace=True)
df.sort_index(inplace=True)

# Complex selection
slice_selection = df.loc['2023-01':'2023-02', 'A':'B']
value_selection = df.xs('A', level='category')

print("Filtered Data:\n", filtered.head())
print("\nMulti-index Selection:\n", slice_selection.head())
print("\nCross-section Selection:\n", value_selection.head())
```

Slide 13: Quản lý bộ nhớ và tối ưu hóa hiệu suất

Kỹ thuật quản lý bộ nhớ nâng cao rất quan trọng khi làm việc với các tập dữ liệu lớn. Hiểu cách NumPy và Pandas xử lý bộ nhớ nội bộ cho phép tối ưu hóa quy trình xử lý dữ liệu để có hiệu suất tốt hơn.

```python
import numpy as np
import pandas as pd
import sys

# Memory usage analysis
def analyze_memory_usage(obj, name="Object"):
    size_bytes = sys.getsizeof(obj)
    if isinstance(obj, (np.ndarray, pd.DataFrame)):
        size_bytes = obj.memory_usage(deep=True).sum()

    # Convert to readable format
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{name} size: {size_bytes:.2f} {unit}"
        size_bytes /= 1024

# Compare different data types
df_float64 = pd.DataFrame(np.random.randn(100000, 4), columns=['A', 'B', 'C', 'D'])
df_float32 = df_float64.astype(np.float32)
df_sparse = pd.DataFrame(np.random.choice([0, 1], size=(100000, 4), p=[0.99, 0.01]))
df_sparse = df_sparse.astype(pd.SparseDtype("int", fill_value=0))

print(analyze_memory_usage(df_float64, "Float64 DataFrame"))
print(analyze_memory_usage(df_float32, "Float32 DataFrame"))
print(analyze_memory_usage(df_sparse, "Sparse DataFrame"))

# Memory-efficient operations
def efficient_operation(df):
    return df.groupby('A')['B'].transform('mean')

def inefficient_operation(df):
    return df.apply(lambda x: x['B'] - x['B'].mean())

# Example of memory-efficient chunking
def process_large_csv(filename, chunksize=10000):
    chunks = []
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        processed = chunk.value.mean()  # Example operation
        chunks.append(processed)
    return pd.concat(chunks)
```

Trang trình bày 14: Đường dẫn NumPy và Pandas tích hợp

Một ví dụ toàn diện trình bày cách kết hợp hiệu quả NumPy và Pandas trong quy trình xử lý dữ liệu trong thế giới thực, tận dụng điểm mạnh của cả hai thư viện để có hiệu suất tối ưu.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create sample dataset
np.random.seed(42)
n_samples = 100000

# Generate data using NumPy (efficient for numerical computations)
numeric_features = np.random.randn(n_samples, 3)
categorical_features = np.random.choice(['A', 'B', 'C'], size=(n_samples, 2))
timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1min')

# Convert to Pandas for data manipulation
df = pd.DataFrame(
    np.hstack([numeric_features, categorical_features]),
    columns=['value1', 'value2', 'value3', 'cat1', 'cat2']
)
df['timestamp'] = timestamps

# Preprocessing pipeline
def preprocess_pipeline(df):
    # Use NumPy for numerical calculations
    numeric_cols = ['value1', 'value2', 'value3']
    numeric_data = df[numeric_cols].values

    # Standardize using NumPy operations
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Back to Pandas for feature engineering
    df[numeric_cols] = scaled_data

    # Add time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek

    # One-hot encoding using Pandas
    categorical_dummies = pd.get_dummies(df[['cat1', 'cat2']], prefix=['cat1', 'cat2'])

    # Combine features
    final_df = pd.concat([df[numeric_cols],
                         df[['hour', 'dayofweek']],
                         categorical_dummies], axis=1)

    return final_df

# Process data
processed_df = preprocess_pipeline(df)
print("Processed DataFrame Shape:", processed_df.shape)
print("\nFeature Names:", processed_df.columns.tolist())
print("\nMemory Usage:", processed_df.memory_usage().sum() / 1024 / 1024, "MB")
```

Trang trình bày 15: Tài nguyên bổ sung

* Học máy với NumPy và Pandas:
    * [https://arxiv.org/abs/2306.15561](https://arxiv.org/abs/2306.15561)
    * [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0267642](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0267642)
* Tối ưu hóa hiệu suất:
    * [https://www.nature.com/articles/s41598-020-76767-0](https://www.nature.com/articles/s41598-020-76767-0)
    * [https://academic.oup.com/gigascience/article/9/10/giaa102/5918883](https://academic.oup.com/gigascience/article/9/10/giaa102/5918883)
* Các phương pháp thực hành và hướng dẫn tốt nhất:
    * [https://scipy.org/](https://scipy.org/)
    * [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
    * [https://numpy.org/doc/stable/user/](https://numpy.org/doc/stable/user/)
