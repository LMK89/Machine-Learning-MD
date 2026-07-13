## Lọc khung dữ liệu hiệu quả với truy vấn Pandas
Trang trình bày 1: Giới thiệu về Phương thức truy vấn DataFrame

Phương thức truy vấn trong pandas cung cấp một cách mạnh mẽ và hiệu quả để lọc DataFrames bằng cách sử dụng biểu thức chuỗi. Không giống như lập chỉ mục boolean truyền thống, truy vấn tận dụng tối ưu hóa tính toán và cung cấp cú pháp dễ đọc hơn cho các hoạt động lọc phức tạp.

```python
import pandas as pd
import numpy as np

# Create sample DataFrame
df = pd.DataFrame({
    'A': np.random.randint(1, 100, 1000),
    'B': np.random.choice(['X', 'Y', 'Z'], 1000),
    'C': np.random.uniform(0, 1, 1000)
})

# Traditional boolean indexing
result1 = df[(df['A'] > 50) & (df['B'] == 'X')]

# Using query method
result2 = df.query('A > 50 and B == "X"')
```

Slide 2: Cú pháp truy vấn và biểu thức chuỗi

Truy vấn chấp nhận các biểu thức chuỗi có thể tham chiếu trực tiếp đến tên cột mà không cần tiền tố DataFrame. Phương thức này hỗ trợ các phép toán logic phức tạp, toán tử so sánh và thậm chí cả các tham chiếu biến nội tuyến bằng ký hiệu '@'.

```python
# Sample DataFrame
df = pd.DataFrame({
    'age': range(20, 30),
    'salary': range(30000, 40000, 1000),
    'department': ['IT', 'HR', 'Sales', 'IT', 'Sales', 'HR', 'IT', 'Sales', 'HR', 'IT']
})

# Multiple conditions
filtered_df = df.query('age >= 25 and salary < 35000')

# Using in/not in operations
filtered_df2 = df.query('department in ["IT", "Sales"]')

# Variable reference
min_age = 22
filtered_df3 = df.query('age > @min_age')
```

Slide 3: Tối ưu hóa hiệu suất trong truy vấn

Phương thức truy vấn biên dịch nội bộ biểu thức chuỗi thành mã byte, giúp nó nhanh hơn đáng kể so với tính năng lọc truyền thống dành cho các tập dữ liệu lớn. Nó cũng làm giảm mức sử dụng bộ nhớ bằng cách tránh tạo mặt nạ boolean trung gian.

```python
import time
import pandas as pd
import numpy as np

# Create large DataFrame
large_df = pd.DataFrame({
    'value': np.random.randn(1000000),
    'category': np.random.choice(['A', 'B', 'C'], 1000000),
    'id': range(1000000)
})

# Measure traditional filtering
start = time.time()
result1 = large_df[(large_df['value'] > 0) & (large_df['category'] == 'A')]
traditional_time = time.time() - start

# Measure query method
start = time.time()
result2 = large_df.query('value > 0 and category == "A"')
query_time = time.time() - start

print(f"Traditional: {traditional_time:.4f}s")
print(f"Query: {query_time:.4f}s")
```

Slide 4: Lọc phức tạp với các phép toán

Phương thức truy vấn hỗ trợ các phép toán và hàm toán học phức tạp trong biểu thức chuỗi. Điều này cho phép các điều kiện lọc phức tạp mà không cần các phép toán boolean lồng nhau.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'x': np.random.uniform(-10, 10, 1000),
    'y': np.random.uniform(-10, 10, 1000),
    'z': np.random.uniform(-10, 10, 1000)
})

# Complex mathematical filtering
result = df.query('(x**2 + y**2) <= 100 and abs(z) < 5')

# Combined with string operations
df['category'] = np.random.choice(['type_A', 'type_B', 'type_C'], 1000)
filtered = df.query('category.str.contains("type_") and z >= 0')
```

Trang trình bày 5: Ví dụ thực tế - Phân tích dữ liệu tài chính

Trong ví dụ thực tế này, chúng tôi sẽ phân tích dữ liệu giao dịch tài chính bằng phương pháp truy vấn để lọc và phân tích hiệu quả các hồ sơ tài chính quy mô lớn.

```python
import pandas as pd
import numpy as np

# Create sample financial dataset
np.random.seed(42)
n_records = 100000

transactions_df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=n_records, freq='5min'),
    'amount': np.random.normal(1000, 500, n_records),
    'transaction_type': np.random.choice(['purchase', 'refund', 'transfer'], n_records),
    'account_type': np.random.choice(['savings', 'checking', 'investment'], n_records),
    'risk_score': np.random.uniform(0, 100, n_records)
})

# Complex financial analysis using query
high_risk_transfers = transactions_df.query(
    'risk_score > 80 and '
    'transaction_type == "transfer" and '
    'amount > 1500 and '
    'account_type != "investment"'
)

print(f"Suspicious transactions found: {len(high_risk_transfers)}")
```

Trang trình bày 6: Làm việc với DateTime trong Truy vấn

Phương thức truy vấn tích hợp liền mạch với chức năng DateTime của gấu trúc, cho phép thực hiện các hoạt động lọc theo thời gian phức tạp. Điều này đặc biệt hữu ích khi phân tích dữ liệu chuỗi thời gian hoặc bộ dữ liệu dựa trên sự kiện.

```python
import pandas as pd
import numpy as np

# Create time-series dataset
dates = pd.date_range('2023-01-01', '2023-12-31', freq='H')
df = pd.DataFrame({
    'timestamp': dates,
    'value': np.random.normal(100, 15, len(dates)),
    'event_type': np.random.choice(['A', 'B', 'C'], len(dates))
})

# Convert timestamp to datetime index
df.set_index('timestamp', inplace=True)

# Query with datetime operations
morning_events = df.query('index.hour >= 9 and index.hour <= 17')
summer_data = df.query('index.month >= 6 and index.month <= 8')

# Complex datetime filtering
busy_periods = df.query(
    'index.hour.between(9, 17) and '
    'index.dayofweek < 5 and '  # Monday = 0, Friday = 4
    'value > 110'
)
```

Trang trình bày 7: Truy vấn với biểu thức chính quy

Phương thức truy vấn hỗ trợ các thao tác chuỗi và biểu thức chính quy thông qua trình truy cập str, cho phép khả năng lọc dựa trên văn bản mạnh mẽ trong khi vẫn duy trì các lợi ích về hiệu suất.

```python
import pandas as pd

# Create dataset with text data
df = pd.DataFrame({
    'email': ['john.doe@company.com', 'jane@gmail.com',
              'support@company.com', 'info@website.org'],
    'message': ['Hello World', 'Query test',
                'Important notice', 'Regular expression'],
    'priority': [1, 2, 1, 3]
})

# String pattern matching
company_emails = df.query('email.str.contains("company.com")')

# Combined regex and numerical filtering
important_company = df.query(
    'email.str.contains("company.com") and '
    'priority < 2 and '
    'message.str.contains("Important|Urgent", case=False)'
)

print("Filtered results:")
print(important_company)
```

Trang trình bày 8: Hoạt động truy vấn lồng nhau

Phân tích dữ liệu phức tạp thường yêu cầu nhiều bước lọc. Phương thức truy vấn có thể được xâu chuỗi một cách hiệu quả, cho phép thực hiện các hoạt động lọc tuần tự trong khi vẫn duy trì khả năng đọc mã.

```python
import pandas as pd
import numpy as np

# Create hierarchical dataset
df = pd.DataFrame({
    'department': np.random.choice(['Sales', 'IT', 'HR'], 1000),
    'team': np.random.choice(['Alpha', 'Beta', 'Gamma'], 1000),
    'performance': np.random.uniform(0, 100, 1000),
    'years_exp': np.random.randint(1, 15, 1000),
    'salary': np.random.uniform(40000, 100000, 1000)
})

# Multiple sequential queries
result = (df.query('department == "Sales"')
           .query('performance > 80')
           .query('years_exp >= 5')
           .query('salary < 80000'))

# Alternative single complex query
result_alternative = df.query(
    'department == "Sales" and '
    'performance > 80 and '
    'years_exp >= 5 and '
    'salary < 80000'
)

print(f"Found {len(result)} matching records")
```

Trang trình bày 9: Kỹ thuật tối ưu hóa hiệu suất truy vấn

Hiểu các kỹ thuật tối ưu hóa truy vấn là rất quan trọng để xử lý các tập dữ liệu lớn một cách hiệu quả. Ví dụ này thể hiện các cách tiếp cận khác nhau để tối ưu hóa hiệu suất truy vấn.

```python
import pandas as pd
import numpy as np
import time

# Create large dataset
large_df = pd.DataFrame({
    'id': range(1000000),
    'category': np.random.choice(['A', 'B', 'C', 'D'], 1000000),
    'value': np.random.uniform(0, 1000, 1000000),
    'text': np.random.choice(['abc', 'def', 'ghi', 'jkl'], 1000000)
})

# Optimization technique 1: Index-based filtering
large_df.set_index('id', inplace=True)

# Optimization technique 2: Pre-computed conditions
threshold = 500
categories = ['A', 'B']

def measure_query_time(query_func):
    start = time.time()
    result = query_func()
    return time.time() - start, len(result)

# Standard query
t1, n1 = measure_query_time(
    lambda: large_df.query('value > @threshold and category in @categories')
)

print(f"Query execution time: {t1:.4f}s, Records found: {n1}")
```

Slide 10: Truy vấn với các thao tác nhóm

Phương pháp truy vấn có thể được kết hợp hiệu quả với các hoạt động theo nhóm để phân tích dữ liệu phức tạp. Cách tiếp cận này cho phép tập hợp được lọc trong khi vẫn duy trì hiệu quả tính toán.

```python
import pandas as pd
import numpy as np

# Create sales dataset
df = pd.DataFrame({
    'product_id': np.random.randint(1000, 2000, 10000),
    'store_id': np.random.randint(1, 50, 10000),
    'sales': np.random.uniform(10, 1000, 10000),
    'date': pd.date_range('2023-01-01', periods=10000, freq='H'),
    'promotion': np.random.choice([True, False], 10000)
})

# Complex query with grouping
result = (df.query('sales > 500 and promotion == True')
          .groupby('store_id')
          .agg({
              'sales': ['count', 'mean', 'sum'],
              'product_id': 'nunique'
          }))

# Time-based grouped analysis
monthly_analysis = (df.query('sales > @df.sales.mean()')
                   .set_index('date')
                   .groupby(pd.Grouper(freq='M'))
                   .agg({'sales': 'sum', 'promotion': 'sum'}))

print("High-value sales analysis:")
print(result.head())
```

Trang trình bày 11: Ví dụ thực tế - Phân khúc khách hàng

Triển khai phân khúc khách hàng bằng phương pháp truy vấn để lọc và phân tích hiệu quả các mẫu hành vi của khách hàng trong bộ dữ liệu thương mại điện tử quy mô lớn.

```python
import pandas as pd
import numpy as np

# Generate customer dataset
n_customers = 100000
customer_data = pd.DataFrame({
    'customer_id': range(n_customers),
    'total_purchases': np.random.normal(500, 200, n_customers),
    'avg_order_value': np.random.normal(100, 30, n_customers),
    'days_since_last_purchase': np.random.randint(1, 365, n_customers),
    'loyalty_score': np.random.uniform(0, 100, n_customers),
    'age': np.random.normal(35, 12, n_customers).astype(int)
})

# Define segment criteria
vip_customers = customer_data.query(
    'total_purchases > @customer_data.total_purchases.quantile(0.9) and '
    'loyalty_score > 80 and '
    'days_since_last_purchase < 30'
)

# At-risk customers
at_risk = customer_data.query(
    'loyalty_score < 40 and '
    'days_since_last_purchase > 60 and '
    'total_purchases > @customer_data.total_purchases.mean()'
)

print(f"VIP Customers: {len(vip_customers)}")
print(f"At-risk Customers: {len(at_risk)}")
```

Trang trình bày 12: Các mẫu tối ưu hóa truy vấn nâng cao

Hiểu các mẫu truy vấn nâng cao giúp tối ưu hóa các hoạt động lọc phức tạp trong khi vẫn duy trì hiệu suất và khả năng đọc mã.

```python
import pandas as pd
import numpy as np

# Create complex dataset
df = pd.DataFrame({
    'metric_a': np.random.normal(100, 15, 100000),
    'metric_b': np.random.normal(50, 10, 100000),
    'category': np.random.choice(['X', 'Y', 'Z'], 100000),
    'subcategory': np.random.choice(['A1', 'A2', 'B1', 'B2'], 100000),
    'value': np.random.uniform(0, 1000, 100000)
})

# Advanced filtering pattern with statistical thresholds
thresholds = {
    'metric_a_mean': df['metric_a'].mean(),
    'metric_b_std': df['metric_b'].std(),
    'value_quantile': df['value'].quantile(0.75)
}

# Optimized complex query
filtered_data = df.query(
    'metric_a > @thresholds["metric_a_mean"] and '
    'abs(metric_b - metric_b.mean()) < @thresholds["metric_b_std"] and '
    'value >= @thresholds["value_quantile"]'
)

print("Statistical filtering results:")
print(f"Original records: {len(df)}")
print(f"Filtered records: {len(filtered_data)}")
```

Trang trình bày 13: Xử lý lỗi truy vấn và các phương pháp hay nhất

Khi làm việc với phương thức truy vấn, việc xử lý lỗi thích hợp và tuân theo các phương pháp hay nhất sẽ đảm bảo mã mạnh mẽ và có thể bảo trì. Ví dụ này cho thấy những cạm bẫy phổ biến và giải pháp của chúng.

```python
import pandas as pd
import numpy as np

# Create sample dataset with potential problematic data
df = pd.DataFrame({
    'numeric_col': [1, 2, np.nan, 4, 5],
    'text_col': ['A', None, 'C', 'D', 'E'],
    'mixed_col': [1, 'text', 3, 4.5, np.nan],
    'date_col': pd.date_range('2023-01-01', periods=5)
})

# Safe query pattern with error handling
def safe_query(dataframe, query_string):
    try:
        result = dataframe.query(query_string)
        return result
    except pd.computation.ops.UndefinedVariableError:
        print("Error: Referenced variable not found")
        return dataframe
    except SyntaxError:
        print("Error: Invalid query syntax")
        return dataframe
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return dataframe

# Example usage with different scenarios
valid_query = safe_query(df, 'numeric_col > 2')
invalid_query = safe_query(df, 'invalid_col > 2')

# Handling missing values
clean_query = df.query('numeric_col.notna()', engine='python')
```

Slide 14: So sánh hiệu suất với các phương pháp lọc khác nhau

Sự so sánh toàn diện này thể hiện sự khác biệt về hiệu suất giữa phương pháp truy vấn và các phương pháp lọc khác trên các kích thước tập dữ liệu khác nhau.

```python
import pandas as pd
import numpy as np
import time

def benchmark_filtering_methods(n_rows):
    # Create test dataset
    df = pd.DataFrame({
        'id': range(n_rows),
        'value': np.random.normal(0, 1, n_rows),
        'category': np.random.choice(['A', 'B', 'C'], n_rows),
        'subcategory': np.random.choice(['X', 'Y', 'Z'], n_rows)
    })

    times = {}

    # Test query method
    start = time.time()
    result1 = df.query('value > 0 and category == "A"')
    times['query'] = time.time() - start

    # Test boolean indexing
    start = time.time()
    result2 = df[(df['value'] > 0) & (df['category'] == 'A')]
    times['boolean'] = time.time() - start

    # Test loc method
    start = time.time()
    result3 = df.loc[(df['value'] > 0) & (df['category'] == 'A')]
    times['loc'] = time.time() - start

    return times

# Test with different dataset sizes
sizes = [1000, 10000, 100000, 1000000]
results = {size: benchmark_filtering_methods(size) for size in sizes}

# Print results
for size, times in results.items():
    print(f"\nDataset size: {size:,} rows")
    for method, time_taken in times.items():
        print(f"{method}: {time_taken:.4f} seconds")
```

Trang trình bày 15: Tài nguyên bổ sung

* "Thao tác dữ liệu hiệu quả trong Python với gấu trúc" - [https://arxiv.org/abs/2001.00789](https://arxiv.org/abs/2001.00789)
* "Kỹ thuật tối ưu hóa hiệu suất để phân tích dữ liệu quy mô lớn" - [https://www.sciencedirect.com/science/article/pii/S0167739X18313189](https://www.sciencedirect.com/science/article/pii/S0167739X18313189)
* "Tối ưu hóa truy vấn trong hoạt động DataFrame" - Xem xét việc tìm kiếm trên Google Scholar các bài viết gần đây về kỹ thuật tối ưu hóa gấu trúc
* "Thao tác khung dữ liệu hiện đại: Đánh giá toàn diện" - [https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0189-0](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0189-0)
* Tài liệu và phương pháp hay nhất: [https://pandas.pydata.org/docs/user\_guide/indexing.html#indexing-query](https://pandas.pydata.org/docs/user_guide/indexing.html#indexing-query)
