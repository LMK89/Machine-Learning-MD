## Kỳ vọng về hoạt động tại chỗ của Pandas rất thực tế
Trang trình bày 1: Tìm hiểu hoạt động tại trang Pandas

Tham số tại chỗ hoạt động của Pandas thường bị hiểu nhầm. Mặc dù các nhà phát triển mong đợi nó sẽ sửa đổi cấu trúc dữ liệu một cách trực tiếp mà không cần tạo bản sao, nhưng thực tế lại phức tạp hơn. Các hoạt động tại chỗ thực hiện tạo ra các bản sao tạm thời trước khi được phân bổ, có khả năng ảnh hưởng đến hiệu suất.

```python
import pandas as pd
import numpy as np
import time

# Create sample DataFrame
df = pd.DataFrame(np.random.randn(1000000, 5), columns=['A', 'B', 'C', 'D', 'E'])

# Compare performance of inplace vs regular operation
start = time.time()
df.sort_values('A', inplace=True)
inplace_time = time.time() - start

df_copy = df.copy()
start = time.time()
df_sorted = df_copy.sort_values('A')
regular_time = time.time() - start

print(f"Inplace operation time: {inplace_time:.4f} seconds")
print(f"Regular operation time: {regular_time:.4f} seconds")
```

Trình bày 2: Phân tích sử dụng bộ nhớ

Làm rõ ý nghĩa bộ nhớ của hoạt động tại chỗ Yêu cầu phải giám sát bộ nhớ bổ sung. Ngược lại với trực tuyến, các thao tác tại bộ nhớ tiêu tốn tương đương hoặc nhiều hơn so với các thao tác không diễn ra tại phòng để tạo bản sao tạm thời.

```python
import memory_profiler
import pandas as pd

@memory_profiler.profile
def inplace_operation():
    df = pd.DataFrame({'A': range(1000000)})
    df.sort_values('A', inplace=True)
    return df

@memory_profiler.profile
def regular_operation():
    df = pd.DataFrame({'A': range(1000000)})
    return df.sort_values('A')

# Execute both functions to compare memory usage
_ = inplace_operation()
_ = regular_operation()
```

Trang trình bày 3: Phân tích cảnh báo SettingWithCopy

Pandas thực hiện kiểm tra tiện ích bổ sung trong các hoạt động tại phòng để đảm bảo an toàn cho dữ liệu, bao gồm cơ chế cảnh báo SettingWithCopy. Những lần kiểm tra này có thể gây ra chi phí đáng kể, đặc biệt là khi làm việc với các DataFrame lớn hoặc các hoạt động phức tạp.

```python
import pandas as pd

# Create a DataFrame with chained operations
df = pd.DataFrame({'A': range(10), 'B': range(10)})

# Example triggering SettingWithCopy warning
def demonstrate_warning():
    subset = df[df['A'] > 5]    # Creates a view
    subset['B'] = 999           # Triggers warning

# Proper way to modify data
def proper_modification():
    df.loc[df['A'] > 5, 'B'] = 999

# Execute both approaches
demonstrate_warning()
proper_modification()
```

Trang trình bày 4: Khung đánh giá hiệu suất

Để đánh giá cách thức có được hệ thống hoạt động tại chỗ, chúng tôi cần có một khuôn khổ đo lường tiêu chuẩn. Công việc khai báo này thực hiện thời gian thực hiện và khả năng sử dụng bộ nhớ trên nhiều DataFrame kích thước và các loại hoạt động khác nhau.

```python
import pandas as pd
import numpy as np
from memory_profiler import memory_usage
import time

def benchmark_operation(operation, df_size, operation_type, inplace=False):
    df = pd.DataFrame(np.random.randn(df_size, 5),
                     columns=['A', 'B', 'C', 'D', 'E'])

    start_time = time.time()
    mem_usage = memory_usage((operation, (df, inplace), {}))
    execution_time = time.time() - start_time

    return {
        'operation': operation_type,
        'size': df_size,
        'inplace': inplace,
        'time': execution_time,
        'max_memory': max(mem_usage)
    }
```

Slide 5: So sánh các hoạt động chung

Phân tích khác biệt về hiệu suất giữa các hoạt động Pandas được sử dụng thường xuyên cho các mô hình nhất quán. Công việc khai báo này so sánh các hoạt động sắp xếp, điền, giãn và đặt lại\_index có và không có tham số tại chỗ.

```python
def compare_operations(df_size=1000000):
    operations = {
        'sort_values': lambda df, inplace: df.sort_values('A', inplace=inplace),
        'fillna': lambda df, inplace: df.fillna(0, inplace=inplace),
        'drop': lambda df, inplace: df.drop('B', axis=1, inplace=inplace),
        'reset_index': lambda df, inplace: df.reset_index(inplace=inplace)
    }

    results = []
    for op_name, op_func in operations.items():
        for inplace in [True, False]:
            result = benchmark_operation(op_func, df_size, op_name, inplace)
            results.append(result)

    return pd.DataFrame(results)
```

Slide 6: Alternative Approaches

Instead of relying on inplace operations, we can implement more efficient approaches using direct assignment or method chaining. These alternatives often provide better performance while maintaining code readability.

```python
import pandas as pd
import numpy as np

# Create sample DataFrame
df = pd.DataFrame(np.random.randn(1000000, 3), columns=['A', 'B', 'C'])

# Method 1: Direct assignment
start = time.time()
df = df.sort_values('A')
time1 = time.time() - start

# Method 2: Method chaining
start = time.time()
df = (df
      .sort_values('A')
      .reset_index(drop=True)
      .fillna(0))
time2 = time.time() - start

print(f"Direct assignment: {time1:.4f}s")
print(f"Method chaining: {time2:.4f}s")
```

Trang trình bày 7: Phân tích hành động sao chép

Hiểu cách Pandas quản lý các bản dữ liệu là rất quan trọng để tối ưu hóa hiệu suất. Việc phát triển này có thể tạo ra các bản sao chép khác nhau và các hoạt động của chúng để sử dụng bộ nhớ và thời gian thực thi.

```python
import pandas as pd
import numpy as np
from memory_profiler import profile

@profile
def analyze_copy_behavior():
    # Original DataFrame
    df = pd.DataFrame(np.random.randn(100000, 3))

    # View creation
    view = df[df > 0]

    # Copy creation
    copy = df.copy()

    # Inplace operation
    df.fillna(0, inplace=True)

    return df, view, copy

# Execute analysis
result_df, result_view, result_copy = analyze_copy_behavior()
```

Trang trình bày 8: Ví dụ thực tế: Quy trình làm sạch dữ liệu

Việc phát triển quy trình làm sạch dữ liệu thực tế cho thấy hoạt động của hoạt động tại chỗ trong các vấn đề sản xuất. Ví dụ này xử lý một tệp lớn với nhiều chuyển đổi bước.

```python
import pandas as pd
import numpy as np
from datetime import datetime

def efficient_data_cleaning(file_path):
    # Load dataset
    df = pd.read_csv(file_path)

    # Chain operations instead of using inplace
    df = (df
          .drop_duplicates()
          .fillna({'numeric_col': 0, 'string_col': 'unknown'})
          .sort_values('date_col')
          .reset_index(drop=True))

    # Calculate derived columns
    df['processed_date'] = datetime.now()
    df['row_number'] = np.arange(len(df))

    return df

# Example usage with timing
start = time.time()
clean_df = efficient_data_cleaning('large_dataset.csv')
print(f"Processing time: {time.time() - start:.4f}s")
```

Trang trình bày 9: Chiến lược tối ưu hóa hiệu suất

Khi làm việc với các dữ liệu lớn, công việc tối ưu hóa hoạt động của Pandas trở nên quan trọng. Việc phát triển này có thể thực hiện các chiến lược khác nhau để cải thiện hiệu quả hoạt động ngoài quyết định tại chỗ và không tại phòng.

```python
import pandas as pd
import numpy as np

def optimize_operations(df):
    # Strategy 1: Use numpy operations where possible
    df['numpy_calc'] = df.values.sum(axis=1)

    # Strategy 2: Vectorized operations
    df['categorical'] = pd.Categorical(df['string_column'])

    # Strategy 3: Bulk updates
    mask = df['value'] > 0
    df.loc[mask, ['col1', 'col2']] = df.loc[mask, ['col1', 'col2']] * 2

    # Strategy 4: Use efficient dtypes
    df['integer_col'] = df['integer_col'].astype('int32')

    return df
```

Trang trình bày 10: Mã nguồn cho chiến lược tối ưu hóa hiệu suất

```python
def measure_optimization_impact():
    # Create test DataFrame
    df = pd.DataFrame({
        'string_column': np.random.choice(['A', 'B', 'C'], 1000000),
        'value': np.random.randn(1000000),
        'col1': np.random.randn(1000000),
        'col2': np.random.randn(1000000),
        'integer_col': np.random.randint(0, 100, 1000000)
    })

    # Measure original memory
    original_memory = df.memory_usage().sum() / 1024**2

    # Apply optimizations
    start = time.time()
    df = optimize_operations(df)
    optimization_time = time.time() - start

    # Measure optimized memory
    optimized_memory = df.memory_usage().sum() / 1024**2

    return {
        'original_memory_mb': original_memory,
        'optimized_memory_mb': optimized_memory,
        'optimization_time_s': optimization_time
    }

# Execute and print results
results = measure_optimization_impact()
print(f"Memory reduction: {results['original_memory_mb'] - results['optimized_memory_mb']:.2f} MB")
print(f"Optimization time: {results['optimization_time_s']:.4f} s")
```

Trang 11: Triển khai chuỗi hành động

Việc phát triển các hoạt động cung cấp chuỗi cung cấp một giải pháp thay thế sạch hơn và thường mang lại hiệu quả hơn cho các hoạt động tại chỗ. Mẫu này duy trì tính chất bất kỳ biến thể nào khi có khả năng cải thiện hiệu suất thông qua việc thực thi các đường dẫn được tối ưu hóa.

```python
class DataFrameChain:
    def __init__(self, df):
        self.df = df

    def transform(self, func):
        self.df = func(self.df)
        return self

    def result(self):
        return self.df

# Example usage
def process_dataframe(df):
    return (DataFrameChain(df)
            .transform(lambda x: x.sort_values('A'))
            .transform(lambda x: x.fillna(0))
            .transform(lambda x: x.reset_index(drop=True))
            .result())

# Benchmark
df = pd.DataFrame(np.random.randn(100000, 3), columns=['A', 'B', 'C'])
start = time.time()
result = process_dataframe(df)
print(f"Chain operation time: {time.time() - start:.4f}s")
```

Trang trình bày 12: Kết quả hoạt động của bộ nhớ

Việc phát triển các hoạt động sử dụng kết quả bộ nhớ Yêu cầu phải hiểu cách quản lý bộ nhớ trong Pandas. Việc phát triển này có thể hiện thực hóa các kỹ thuật xử lý dữ liệu lớn với mức tối thiểu bộ nhớ.

```python
import pandas as pd
import numpy as np
from contextlib import contextmanager

@contextmanager
def track_memory():
    import psutil
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024
    yield
    mem_after = process.memory_info().rss / 1024 / 1024
    print(f"Memory change: {mem_after - mem_before:.2f} MB")

def memory_efficient_processing(df):
    # Use generators for memory efficiency
    def process_chunks():
        for chunk in np.array_split(df, 10):
            yield chunk.mean()

    # Process in chunks
    with track_memory():
        results = pd.concat(process_chunks())

    return results
```

Trang trình bày 13: Hiệu suất số liệu trực quan

Tạo ra hiệu suất số liệu toàn diện giúp hiểu được hoạt động của các hoạt động chiến lược khác nhau. Việc phát triển khai báo này tạo ra hình ảnh trực tiếp so sánh các hoạt động tại chỗ và không tại chỗ trong nhiều vấn đề khác nhau.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_performance(sizes=[1000, 10000, 100000, 1000000]):
    results = []

    for size in sizes:
        # Measure inplace operations
        df = pd.DataFrame(np.random.randn(size, 3))
        start = time.time()
        df.sort_values(0, inplace=True)
        inplace_time = time.time() - start

        # Measure regular operations
        df = pd.DataFrame(np.random.randn(size, 3))
        start = time.time()
        _ = df.sort_values(0)
        regular_time = time.time() - start

        results.append({
            'size': size,
            'inplace_time': inplace_time,
            'regular_time': regular_time
        })

    # Create visualization
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['size'], results_df['inplace_time'], label='Inplace')
    plt.plot(results_df['size'], results_df['regular_time'], label='Regular')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('DataFrame Size')
    plt.ylabel('Execution Time (s)')
    plt.legend()
    plt.title('Performance Comparison: Inplace vs Regular Operations')
    plt.show()
```

Trang trình bày 14: Tài nguyên bổ sung

* [https://arxiv.org/abs/1709.03429](https://arxiv.org/abs/1709.03429) - "Tối ưu hóa phân tích dữ liệu với Pandas: Một nghiên cứu toàn diện"
* [https://arxiv.org/abs/1801.07010](https://arxiv.org/abs/1801.07010) - "Xử lý hiệu ứng dữ liệu về bộ nhớ trong Python"
* [https://arxiv.org/abs/1907.08385](https://arxiv.org/abs/1907.08385) - "Phân tích hiệu suất của khung dữ liệu hoạt động trong khoa học dữ liệu"
