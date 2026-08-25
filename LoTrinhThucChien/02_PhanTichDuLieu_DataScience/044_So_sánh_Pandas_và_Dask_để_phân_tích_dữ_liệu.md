## So sánh Pandas và Dask để phân tích dữ liệu
Slide 1: Giới thiệu về Pandas và Dask

Pandas và Dask là những thư viện mạnh mẽ trong Python để vận hành và phân tích dữ liệu. Trong khi Pandas chiếm ưu thế trong việc xử lý dữ liệu nhỏ hơn trong bộ nhớ thì Dask được thiết kế để xử lý dữ liệu lớn không vừa với bộ nhớ. Trình chiếu này sẽ so sánh hai thư viện này, nêu điểm mạnh và trường hợp sử dụng của chúng.

```python
import pandas as pd
import dask.dataframe as dd

# Creating a small dataset with Pandas
pandas_df = pd.DataFrame({'A': range(5), 'B': range(5, 10)})

# Creating a large dataset with Dask
dask_df = dd.from_pandas(pandas_df, npartitions=2)

print("Pandas DataFrame:")
print(pandas_df)
print("\nDask DataFrame:")
print(dask_df)
```

Slide 2: Cấu hình dữ liệu

Pandas chủ yếu sử dụng các DataFrame và Series tượng trưng để thao tác dữ liệu. Dask mở rộng các khái niệm này để xử lý các dữ liệu lớn hơn bằng cách chia thành các phân vùng có thể xử lý bài hát.

```python
# Pandas DataFrame and Series
pandas_df = pd.DataFrame({'A': range(5), 'B': range(5, 10)})
pandas_series = pd.Series(range(5))

# Dask DataFrame and Series
dask_df = dd.from_pandas(pandas_df, npartitions=2)
dask_series = dd.from_pandas(pandas_series, npartitions=2)

print("Pandas DataFrame shape:", pandas_df.shape)
print("Dask DataFrame shape:", dask_df.shape.compute())
print("\nPandas Series:")
print(pandas_series)
print("\nDask Series:")
print(dask_series.compute())
```

Trang hiển thị 3: Đang tải dữ liệu

Cả Pandas và Dask đều cung cấp các phương thức tải dữ liệu từ nhiều nguồn khác nhau. Pandas tải toàn bộ dữ liệu vào bộ nhớ, trong khi Dask có thể hoạt động với dữ liệu không phù hợp với bộ nhớ bằng cách tải dữ liệu theo từng khối.

```python
import pandas as pd
import dask.dataframe as dd

# Loading CSV with Pandas
pandas_df = pd.read_csv('large_file.csv')

# Loading CSV with Dask
dask_df = dd.read_csv('large_file.csv')

print("Pandas DataFrame info:")
print(pandas_df.info())
print("\nDask DataFrame info:")
print(dask_df.info())
```

Slide 4: Các thao tác cơ bản

Cả hai thư viện đều hỗ trợ các hoạt động tương tự như lọc, sắp xếp và tổng hợp. Tuy nhiên, các hoạt động của Dask rất lười và chỉ được tính toán khi được gọi một cách rõ ràng.

```python
import pandas as pd
import dask.dataframe as dd

# Creating sample data
pandas_df = pd.DataFrame({'A': range(10), 'B': range(10, 20)})
dask_df = dd.from_pandas(pandas_df, npartitions=2)

# Filtering
pandas_filtered = pandas_df[pandas_df['A'] > 5]
dask_filtered = dask_df[dask_df['A'] > 5]

# Sorting
pandas_sorted = pandas_df.sort_values('B')
dask_sorted = dask_df.sort_values('B')

print("Pandas filtered and sorted:")
print(pandas_filtered)
print(pandas_sorted)

print("\nDask filtered and sorted:")
print(dask_filtered.compute())
print(dask_sorted.compute())
```

Trang trình bày 5: Use Memory

Pandas tải tất cả dữ liệu vào bộ nhớ, đây có thể là một chế độ hạn chế đối với các dữ liệu lớn. Mặt khác, Dask có thể hoạt động với các dữ liệu lớn hơn khả năng ghi nhớ bằng cách xử lý dữ liệu theo từng khối.

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np

# Create a large DataFrame
large_df = pd.DataFrame(np.random.randn(1000000, 4), columns=list('ABCD'))

# Convert to Dask DataFrame
dask_df = dd.from_pandas(large_df, npartitions=10)

print("Pandas DataFrame memory usage:")
print(large_df.memory_usage(deep=True).sum() / 1e6, "MB")

print("\nDask DataFrame memory usage (estimated):")
print(dask_df.memory_usage(deep=True).sum().compute() / 1e6, "MB")
```

Slide 6: Bài hát xử lý

Dask tận dụng khả năng xử lý bài hát để xử lý các dữ liệu lớn theo cách hiệu quả. Nó có thể phân phối các phép tính trên nhiều lõi hoặc thậm chí một cụm máy.

```python
import pandas as pd
import dask.dataframe as dd
import time

# Create a large DataFrame
large_df = pd.DataFrame({'A': range(10000000), 'B': range(10000000, 20000000)})

# Convert to Dask DataFrame
dask_df = dd.from_pandas(large_df, npartitions=4)

# Measure time for Pandas operation
start_time = time.time()
pandas_result = large_df['A'].mean()
pandas_time = time.time() - start_time

# Measure time for Dask operation
start_time = time.time()
dask_result = dask_df['A'].mean().compute()
dask_time = time.time() - start_time

print(f"Pandas time: {pandas_time:.2f} seconds")
print(f"Dask time: {dask_time:.2f} seconds")
print(f"Speedup: {pandas_time / dask_time:.2f}x")
```

Slide 7:Đánh giá lười biếng

Dask use đánh giá lười biếng, nghĩa là các thao tác được thực hiện chỉ khi kết quả được yêu cầu rõ ràng. Điều này cho phép các kế hoạch được tối ưu hóa và sử dụng kết quả hiệu ứng bộ nhớ.

```python
import pandas as pd
import dask.dataframe as dd

# Create DataFrames
pandas_df = pd.DataFrame({'A': range(10), 'B': range(10, 20)})
dask_df = dd.from_pandas(pandas_df, npartitions=2)

# Define operations
pandas_result = (pandas_df['A'] * 2).mean()
dask_result = (dask_df['A'] * 2).mean()

print("Pandas result (computed immediately):")
print(pandas_result)

print("\nDask result (not computed yet):")
print(dask_result)

print("\nDask result (after computation):")
print(dask_result.compute())
```

Slide 8: Xử lý thời gian chuỗi dữ liệu

Cả Pandas và Dask đều cung cấp các công cụ mạnh mẽ để làm việc với thời gian chuỗi dữ liệu, nhưng Dask có thể xử lý nhiều thời gian chuỗi dữ liệu hơn.

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np

# Create a time series DataFrame
dates = pd.date_range('2023-01-01', periods=1000000, freq='T')
pandas_ts = pd.DataFrame({'timestamp': dates, 'value': np.random.randn(1000000)})

# Convert to Dask DataFrame
dask_ts = dd.from_pandas(pandas_ts, npartitions=10)

# Resample and compute mean
pandas_result = pandas_ts.set_index('timestamp').resample('D').mean()
dask_result = dask_ts.set_index('timestamp').resample('D').mean().compute()

print("Pandas result:")
print(pandas_result.head())
print("\nDask result:")
print(dask_result.head())
```

Slide 9: Thiếu dữ liệu xử lý

Cả Pandas và Dask đều cung cấp các phương pháp xử lý dữ liệu bị thiếu, nhưng Dask có thể xử lý các dữ liệu lớn hơn và thiếu kết quả hiệu quả hơn.

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np

# Create DataFrames with missing values
pandas_df = pd.DataFrame({'A': [1, 2, np.nan, 4, 5], 'B': [np.nan, 2, 3, np.nan, 5]})
dask_df = dd.from_pandas(pandas_df, npartitions=2)

# Fill missing values
pandas_filled = pandas_df.fillna(0)
dask_filled = dask_df.fillna(0)

print("Pandas DataFrame with filled values:")
print(pandas_filled)

print("\nDask DataFrame with filled values:")
print(dask_filled.compute())
```

Slide 10: Phân nhóm và tổng hợp

Cả hai thư viện đều hỗ trợ các nhóm hoạt động và tổng hợp, nhưng Dask có thể xử lý các hoạt động này trên nhiều dữ liệu hơn.

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np

# Create sample data
pandas_df = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C'], 1000000),
    'value': np.random.randn(1000000)
})
dask_df = dd.from_pandas(pandas_df, npartitions=10)

# Perform groupby and aggregation
pandas_result = pandas_df.groupby('category')['value'].mean()
dask_result = dask_df.groupby('category')['value'].mean().compute()

print("Pandas groupby result:")
print(pandas_result)
print("\nDask groupby result:")
print(dask_result)
```

Slide 11: Trực quan hóa

Pandas tích hợp tốt với các thư viện vẽ đồ thị như Matplotlib, trong khi Dask yêu cầu tính toán trước khi trực quan hóa. Tuy nhiên, Dask có thể xử lý tiền xử lý dữ liệu lớn hơn để trực tuyến hóa.

```python
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt

# Create sample data
pandas_df = pd.DataFrame({'x': range(1000), 'y': np.random.randn(1000)})
dask_df = dd.from_pandas(pandas_df, npartitions=10)

# Pandas plot
plt.figure(figsize=(10, 5))
pandas_df.plot(x='x', y='y', ax=plt.subplot(121), title='Pandas Plot')

# Dask plot (compute first)
dask_result = dask_df.compute()
dask_result.plot(x='x', y='y', ax=plt.subplot(122), title='Dask Plot')

plt.tight_layout()
plt.show()
```

Trang trình bày 12: Ví dụ thực tế: Phân tích nhật ký

Phân tích nhật ký máy chủ là một nhiệm vụ phổ biến có thể được hưởng lợi từ cả Pandas và Dask, tùy thuộc vào kích thước của nhật ký tệp.

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np

# Generate sample log data
log_entries = [
    f"{i},2023-05-{np.random.randint(1, 32):02d} {np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}:{np.random.randint(0, 60):02d},{'GET' if np.random.random() > 0.3 else 'POST'},{np.random.choice([200, 404, 500])}"
    for i in range(1000000)
]

# Write to CSV
with open('server_logs.csv', 'w') as f:
    f.write("id,timestamp,method,status\n")
    f.write("\n".join(log_entries))

# Read with Pandas
pandas_logs = pd.read_csv('server_logs.csv', parse_dates=['timestamp'])

# Read with Dask
dask_logs = dd.read_csv('server_logs.csv', parse_dates=['timestamp'])

# Analyze HTTP status codes
pandas_status = pandas_logs['status'].value_counts()
dask_status = dask_logs['status'].value_counts().compute()

print("Pandas status code counts:")
print(pandas_status)
print("\nDask status code counts:")
print(dask_status)
```

Slide 13: Ví dụ thực tế: Phân tích không gian địa lý

Phân tích không gian không gian thường liên kết đến các dữ liệu lớn, tạo ra nó trở thành trường hợp sử dụng lý tưởng để so sánh hiệu suất của Pandas và Dask.

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np
import time

# Generate sample geospatial data
num_points = 1000000
lat = np.random.uniform(30, 50, num_points)
lon = np.random.uniform(-120, -70, num_points)
data = pd.DataFrame({'lat': lat, 'lon': lon})

# Function to check if a point is within a bounding box
def within_bbox(row, bbox):
    return (bbox[0] <= row['lon'] <= bbox[2]) and (bbox[1] <= row['lat'] <= bbox[3])

# Define a bounding box (min_lon, min_lat, max_lon, max_lat)
bbox = (-100, 35, -90, 45)

# Pandas analysis
start_time = time.time()
pandas_result = data[data.apply(within_bbox, axis=1, bbox=bbox)]
pandas_time = time.time() - start_time

# Dask analysis
dask_data = dd.from_pandas(data, npartitions=10)
start_time = time.time()
dask_result = dask_data[dask_data.apply(within_bbox, axis=1, bbox=bbox, meta=('bool')).compute()]
dask_time = time.time() - start_time

print(f"Pandas processing time: {pandas_time:.2f} seconds")
print(f"Dask processing time: {dask_time:.2f} seconds")
print(f"Points within bounding box (Pandas): {len(pandas_result)}")
print(f"Points within bounding box (Dask): {len(dask_result)}")
```

Slide 14: Kết luận

Pandas và Dask đều là những công cụ mạnh mẽ để phân tích dữ liệu bằng Python. Pandas rất lý tưởng cho các tập dữ liệu nhỏ hơn, phù hợp với bộ nhớ và cung cấp nhiều tính năng phong phú để thao tác dữ liệu. Khả năng mở rộng của Dask này đã tăng cường các dữ liệu tập tin bằng cách cho phép xử lý bài hát và tính toán bên ngoài cốt lõi. Chọn Pandas để phân tích nhanh hơn các dữ liệu và mẫu nguyên, đồng thời xem xét Dask khi xử lý các tác vụ xử lý mô-đun dữ liệu lớn hoặc khi bạn cần sử dụng phân tích tài chính điện tử.

```python
import pandas as pd
import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
sizes = [1e3, 1e4, 1e5, 1e6]
pandas_times = []
dask_times = []

for size in sizes:
    data = pd.DataFrame({'A': np.random.randn(int(size)), 'B': np.random.randn(int(size))})

    # Pandas performance
    start = time.time()
    _ = data.groupby('A').B.mean()
    pandas_times.append(time.time() - start)

    # Dask performance
    dask_data = dd.from_pandas(data, npartitions=4)
    start = time.time()
    _ = dask_data.groupby('A').B.mean().compute()
    dask_times.append(time.time() - start)

# Plot performance comparison
plt.figure(figsize=(10, 6))
plt.plot(sizes, pandas_times, label='Pandas')
plt.plot(sizes, dask_times, label='Dask')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Dataset Size')
plt.ylabel('Execution Time (s)')
plt.title('Pandas vs Dask Performance')
plt.legend()
plt.grid(True)
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Để biết thêm thông tin về Pandas và Dask, hãy xem xét khám phá các tài nguyên sau:

1. Tài liệu về gấu trúc: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
2. Tài liệu về Dask: [https://docs.dask.org/en/latest/](https://docs.dask.org/en/latest/)
3. "Mở rộng Pandas: So sánh Dask, Ray, Modin, Vaex và RAPIDS" (arXiv:2202.04935): [https://arxiv.org/abs/2202.04935](https://arxiv.org/abs/2202.04935)
4. "Nghiên cứu so sánh các phân tán dữ liệu khung hệ thống" (arXiv:2011.00719): [https://arxiv.org/abs/2011.00719](https://arxiv.org/abs/2011.00719)

Tài nguyên này cung cấp thông tin chuyên sâu về cả thư viện và nghiên cứu so sánh về phân tích DataFrame hệ thống, có thể giúp bạn đưa ra quyết định sáng suốt về cách sử dụng công cụ nào cho nhu cầu phân tích dữ liệu cụ thể của mình.
