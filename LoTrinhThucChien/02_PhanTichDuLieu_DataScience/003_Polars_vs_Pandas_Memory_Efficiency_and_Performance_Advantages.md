## Polars vs Pandas Ưu điểm về hiệu suất và hiệu suất bộ nhớ
Slide 1: Hiệu quả bộ nhớ thông qua định dạng bộ nhớ mũi tên

Polars tận dụng định dạng bộ nhớ cột của Apache Arrow, cho phép thực hiện các thao tác không sao chép và giảm thiểu chi phí bộ nhớ trong quá trình xử lý dữ liệu. Sự khác biệt về kiến ​​trúc cơ bản này so với Pandas giúp giảm đáng kể mức sử dụng bộ nhớ khi xử lý các tập dữ liệu lớn.

```python
import polars as pl
import pandas as pd
import numpy as np
import time

# Create large dataset
n_rows = 1_000_000
data = {
    'id': range(n_rows),
    'values': np.random.randn(n_rows)
}

# Compare memory usage
df_pd = pd.DataFrame(data)
df_pl = pl.DataFrame(data)

print(f"Pandas Memory Usage: {df_pd.memory_usage().sum() / 1024**2:.2f} MB")
print(f"Polars Memory Usage: {df_pl.estimated_size() / 1024**2:.2f} MB")
```

Slide 2: Thực thi truy vấn song song

Polars tự động song song hóa các hoạt động truy vấn trên các lõi CPU có sẵn, tận dụng khả năng phần cứng hiện đại cho các tác vụ xử lý dữ liệu. Trình tối ưu hóa truy vấn tạo ra các kế hoạch thực thi hiệu quả nhằm giảm thiểu việc phân bổ bộ nhớ và tối đa hóa thông lượng.

```python
# Comparing execution speed for groupby operations
start_time = time.time()
result_pd = df_pd.groupby('id').agg({'values': ['mean', 'std']})
pd_time = time.time() - start_time

start_time = time.time()
result_pl = df_pl.groupby('id').agg([
    pl.col('values').mean(),
    pl.col('values').std()
])
pl_time = time.time() - start_time

print(f"Pandas execution time: {pd_time:.2f} seconds")
print(f"Polars execution time: {pl_time:.2f} seconds")
```

Slide 3: Chiến lược đánh giá lười biếng

Polars triển khai một hệ thống đánh giá lười biếng nhằm tối ưu hóa việc thực hiện truy vấn bằng cách xây dựng biểu đồ tính toán trước khi thực hiện thực tế. Điều này cho phép tối ưu hóa truy vấn và sử dụng tài nguyên hiệu quả so với đánh giá háo hức của Pandas.

```python
import polars as pl
import numpy as np

# Create large dataset
data = pl.DataFrame({
    'A': np.random.randn(1_000_000),
    'B': np.random.randn(1_000_000)
})

# Define lazy computation
lazy_query = (
    data.lazy()
    .filter(pl.col('A') > 0)
    .groupby(pl.col('A').round(1))
    .agg([
        pl.col('B').mean().alias('B_mean'),
        pl.col('B').std().alias('B_std')
    ])
    .sort('A')
)

# Execute query
result = lazy_query.collect()
```

Trang trình bày 4: Các thao tác chuỗi được vector hóa

Polars cung cấp các hoạt động chuỗi được tối ưu hóa cao thông qua triển khai vector hóa, mang lại hiệu suất vượt trội cho các tác vụ xử lý văn bản so với các hoạt động chuỗi của Pandas.

```python
import polars as pl
import pandas as pd
import time

# Create dataset with string operations
n_rows = 1_000_000
data = {
    'text': ['hello_world_' + str(i) for i in range(n_rows)]
}

df_pd = pd.DataFrame(data)
df_pl = pl.DataFrame(data)

# Compare string splitting performance
start_time = time.time()
pd_result = df_pd['text'].str.split('_')
pd_time = time.time() - start_time

start_time = time.time()
pl_result = df_pl['text'].str.split('_')
pl_time = time.time() - start_time

print(f"Pandas string split time: {pd_time:.2f} seconds")
print(f"Polars string split time: {pl_time:.2f} seconds")
```

Trang trình bày 5: Thiết kế API dựa trên biểu thức

Polars giới thiệu API dựa trên biểu thức mạnh mẽ cho phép chuyển đổi dữ liệu phức tạp thông qua các hoạt động có thể tổng hợp. Thiết kế này cho phép mã trực quan hơn và dễ bảo trì hơn trong khi vẫn duy trì hiệu suất cao thông qua các đường dẫn thực thi được tối ưu hóa.

```python
import polars as pl
import numpy as np

# Create sample dataset
df = pl.DataFrame({
    'date': pl.date_range(
        start=datetime(2023, 1, 1),
        end=datetime(2023, 12, 31),
        interval='1d'
    ),
    'sales': np.random.normal(1000, 100, 365),
    'costs': np.random.normal(800, 50, 365)
})

# Complex transformations using expressions
result = df.select([
    pl.col('date'),
    pl.col('sales').rolling_mean(window_size=7).alias('sales_ma7'),
    (pl.col('sales') - pl.col('costs')).alias('profit'),
    pl.col('sales').pct_change().alias('sales_growth')
]).filter(
    pl.col('profit') > pl.col('profit').mean()
)
```

Slide 6: Hoạt động chuỗi thời gian nâng cao

Polars vượt trội trong việc thao tác chuỗi thời gian thông qua các hàm ngày giờ chuyên dụng và các thao tác cửa sổ được tối ưu hóa. Khung này cung cấp hỗ trợ riêng cho các phép tổng hợp và chuyển đổi theo thời gian khác nhau với chi phí tối thiểu.

```python
# Time series analytics example
result = df.select([
    pl.col('date'),
    pl.col('sales').rolling_mean(
        window_size='7d',
        by='date',
        closed='right'
    ).alias('weekly_avg'),
    pl.col('sales').rolling_std(
        window_size='30d',
        by='date'
    ).alias('monthly_volatility'),
    pl.col('date').dt.month().alias('month'),
    pl.col('date').dt.year().alias('year')
]).groupby(['year', 'month']).agg([
    pl.col('sales').mean().alias('monthly_avg_sales'),
    pl.col('weekly_avg').last().alias('last_weekly_avg')
])
```

Slide 7: Query Optimization for Large Datasets

Polars implements sophisticated query optimization techniques including predicate pushdown, projection pushdown, and common subexpression elimination. These optimizations significantly reduce memory usage and computation time for complex queries.

```python
# Example of query optimization benefits
df_large = pl.DataFrame({
    'id': range(10_000_000),
    'value': np.random.randn(10_000_000),
    'category': np.random.choice(['A', 'B', 'C'], 10_000_000)
})

# Complex query with optimization
optimized_query = (
    df_large.lazy()
    .filter(pl.col('value') > 0)
    .groupby('category')
    .agg([
        pl.col('value').mean().alias('avg_value'),
        pl.col('value').quantile(0.95).alias('p95_value')
    ])
    .sort('avg_value', descending=True)
).collect(streaming=True)
```

Trang trình bày 8: Ví dụ thực tế - Phân tích dữ liệu tài chính

Ví dụ này thể hiện hiệu quả của Polars trong việc xử lý dữ liệu giao dịch tần số cao, thể hiện hiệu suất vượt trội của nó trong việc xử lý các hoạt động theo chuỗi thời gian và chuyển đổi theo nhóm.

```python
import polars as pl
from datetime import datetime, timedelta

# Generate sample trading data
n_records = 1_000_000
timestamps = [
    datetime(2024, 1, 1) + timedelta(microseconds=i)
    for i in range(n_records)
]

trading_data = pl.DataFrame({
    'timestamp': timestamps,
    'price': np.random.normal(100, 5, n_records),
    'volume': np.random.exponential(1000, n_records),
    'symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT'], n_records)
})

# Complex financial analysis
analysis_result = (
    trading_data.lazy()
    .with_columns([
        pl.col('timestamp').dt.hour().alias('hour'),
        (pl.col('price') * pl.col('volume')).alias('turnover')
    ])
    .groupby(['symbol', 'hour'])
    .agg([
        pl.col('price').mean().alias('vwap'),
        pl.col('volume').sum().alias('total_volume'),
        pl.col('turnover').sum().alias('total_turnover'),
        pl.col('price').std().alias('price_volatility')
    ])
).collect()
```

Trang trình bày 9: Truyền dữ liệu hiệu quả về bộ nhớ

Polars triển khai khả năng phát trực tuyến cho phép xử lý các tập dữ liệu lớn hơn RAM có sẵn. Cách tiếp cận này duy trì mức sử dụng bộ nhớ liên tục bất kể kích thước đầu vào bằng cách xử lý dữ liệu theo khối trong khi vẫn duy trì tối ưu hóa truy vấn.

```python
import polars as pl
import numpy as np

# Simulate large CSV file creation
def generate_large_csv(filename, n_rows=10_000_000):
    chunk_size = 100_000
    with open(filename, 'w') as f:
        f.write('id,value,category\n')
        for i in range(0, n_rows, chunk_size):
            chunk = pl.DataFrame({
                'id': range(i, min(i + chunk_size, n_rows)),
                'value': np.random.randn(min(chunk_size, n_rows - i)),
                'category': np.random.choice(['A', 'B', 'C'], min(chunk_size, n_rows - i))
            })
            chunk.write_csv(f, has_header=False)

# Stream processing example
streaming_query = (
    pl.scan_csv('large_dataset.csv')
    .filter(pl.col('value') > 0)
    .groupby('category')
    .agg([
        pl.col('value').mean(),
        pl.col('value').count()
    ])
).collect(streaming=True)
```

Trang trình bày 10: Trực quan hóa kết quả phân tích bộ nhớ

Trang trình bày này trình bày các số liệu hiệu suất và kiểu sử dụng bộ nhớ khi xử lý các tập dữ liệu lớn bằng Polars so với các phương pháp truyền thống.

```python
import matplotlib.pyplot as plt
import psutil
import time

def measure_memory_usage(func):
    process = psutil.Process()
    start_mem = process.memory_info().rss / 1024 / 1024
    start_time = time.time()

    result = func()

    end_time = time.time()
    end_mem = process.memory_info().rss / 1024 / 1024

    return {
        'execution_time': end_time - start_time,
        'memory_delta': end_mem - start_mem,
        'result': result
    }

# Compare memory usage patterns
def process_with_polars():
    return pl.scan_csv('large_dataset.csv').collect()

def process_with_pandas():
    return pd.read_csv('large_dataset.csv')

polars_metrics = measure_memory_usage(process_with_polars)
pandas_metrics = measure_memory_usage(process_with_pandas)

print(f"Polars Memory Usage: {polars_metrics['memory_delta']:.2f} MB")
print(f"Pandas Memory Usage: {pandas_metrics['memory_delta']:.2f} MB")
print(f"Polars Execution Time: {polars_metrics['execution_time']:.2f} s")
print(f"Pandas Execution Time: {pandas_metrics['execution_time']:.2f} s")
```

Trang trình bày 11: Ví dụ thực tế - Xử lý dữ liệu cảm biến IoT

Ví dụ này thể hiện hiệu quả của Polars trong việc xử lý dữ liệu cảm biến chuỗi thời gian bằng các phép đo tần số cao và các phép tổng hợp phức tạp.

```python
import polars as pl
from datetime import datetime, timedelta

# Generate IoT sensor data
n_sensors = 100
n_measurements = 1_000_000

sensor_data = pl.DataFrame({
    'timestamp': pl.date_range(
        datetime(2024, 1, 1),
        datetime(2024, 1, 31),
        n_measurements
    ),
    'sensor_id': np.random.randint(1, n_sensors + 1, n_measurements),
    'temperature': np.random.normal(25, 5, n_measurements),
    'humidity': np.random.normal(60, 10, n_measurements),
    'pressure': np.random.normal(1013, 10, n_measurements)
})

# Complex sensor data analysis
analysis_result = (
    sensor_data.lazy()
    .with_columns([
        pl.col('timestamp').dt.hour().alias('hour'),
        pl.col('timestamp').dt.date().alias('date')
    ])
    .groupby(['sensor_id', 'date'])
    .agg([
        pl.all().mean().suffix('_avg'),
        pl.all().std().suffix('_std'),
        pl.col(['temperature', 'humidity', 'pressure'])
          .quantile(0.95)
          .suffix('_p95')
    ])
    .sort(['sensor_id', 'date'])
).collect()
```

Trang trình bày 12: Tối ưu hóa vị từ đẩy xuống

Polars triển khai tối ưu hóa đẩy xuống vị từ nâng cao, đẩy các điều kiện lọc càng gần nguồn dữ liệu càng tốt. Sự tối ưu hóa này làm giảm đáng kể lượng dữ liệu cần được tải và xử lý trong bộ nhớ.

```python
import polars as pl
import numpy as np

# Create sample parquet file with partitioned data
df = pl.DataFrame({
    'date': pl.date_range(
        datetime(2023, 1, 1),
        datetime(2023, 12, 31),
        interval='1h'
    ),
    'region': np.random.choice(['NA', 'EU', 'ASIA'], 8760),
    'sales': np.random.normal(1000, 100, 8760)
})

# Example of predicate pushdown
optimized_query = (
    pl.scan_parquet('sales_data.parquet')
    .filter(
        (pl.col('region') == 'NA') &
        (pl.col('date').dt.year() == 2023)
    )
    .groupby(pl.col('date').dt.month())
    .agg([
        pl.col('sales').sum().alias('monthly_sales'),
        pl.col('sales').mean().alias('avg_daily_sales')
    ])
).collect()
```

Trang trình bày 13: Tham gia và tập hợp nâng cao

Polars cung cấp các triển khai kết hợp và tổng hợp được tối ưu hóa cao nhằm tận dụng khả năng xử lý song song và quản lý bộ nhớ hiệu quả để xử lý các hoạt động dữ liệu quy mô lớn với hiệu suất vượt trội.

```python
import polars as pl
import numpy as np

# Create sample datasets
customers = pl.DataFrame({
    'customer_id': range(1000000),
    'region': np.random.choice(['NA', 'EU', 'ASIA'], 1000000),
    'segment': np.random.choice(['A', 'B', 'C'], 1000000)
})

transactions = pl.DataFrame({
    'transaction_id': range(5000000),
    'customer_id': np.random.randint(0, 1000000, 5000000),
    'amount': np.random.normal(100, 25, 5000000),
    'date': np.random.choice(pl.date_range(
        datetime(2023, 1, 1),
        datetime(2023, 12, 31),
        interval='1d'
    ), 5000000)
})

# Complex join and aggregation
result = (
    transactions.lazy()
    .join(
        customers.lazy(),
        on='customer_id',
        how='left'
    )
    .groupby(['region', 'segment'])
    .agg([
        pl.col('amount').sum().alias('total_amount'),
        pl.col('amount').mean().alias('avg_amount'),
        pl.col('customer_id').n_unique().alias('unique_customers'),
        pl.col('transaction_id').count().alias('transaction_count')
    ])
    .sort(['region', 'total_amount'], descending=True)
).collect()
```

Trang trình bày 14: Tài nguyên bổ sung

* "Polars: Thư viện khung dữ liệu nhanh như chớp" - [https://arxiv.org/abs/2111.12077](https://arxiv.org/abs/2111.12077) (Lưu ý: tìm kiếm các bài viết tương tự vì đây là ví dụ điển hình)
* "Tối ưu hóa hiệu suất truy vấn trong phân tích dữ liệu hiện đại" - [https://www.vldb.org/pvldb/vol13/p3502-chen.pdf](https://www.vldb.org/pvldb/vol13/p3502-chen.pdf)
* "Mũi tên Apache: Nền tảng phát triển đa ngôn ngữ cho dữ liệu trong bộ nhớ" - [https://arrow.apache.org/papers/](https://arrow.apache.org/papers/)
* "Xử lý dữ liệu dựa trên rỉ sét: Hiệu suất và an toàn" - Tìm kiếm các bài viết liên quan trên Google Scholar
* "Các phương pháp tiếp cận hiện đại để xử lý dữ liệu quy mô lớn" - Truy cập [https://db.cs.cmu.edu/papers/](https://db.cs.cmu.edu/papers/) để biết các tài nguyên học thuật
