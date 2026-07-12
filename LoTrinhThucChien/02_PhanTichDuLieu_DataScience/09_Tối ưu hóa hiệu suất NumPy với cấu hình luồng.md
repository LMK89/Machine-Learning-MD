## Tối ưu hóa hiệu suất NumPy với cấu hình luồng
Trang trình bày 1: Tìm hiểu cấu hình luồng NumPy

Hiệu suất của NumPy phụ thuộc rất nhiều vào việc triển khai BLAS (Chương trình con đại số tuyến tính cơ bản) và cấu hình luồng cơ bản của nó. Ba biến môi trường chính kiểm soát hành vi phân luồng là MKL\_NUM\_THREADS, OPENBLAS\_NUM\_THREADS và OMP\_NUM\_THREADS, mỗi biến tương ứng với các chương trình phụ trợ BLAS khác nhau.

```python
import os
import numpy as np

# Set environment variables for thread control
os.environ['MKL_NUM_THREADS'] = '4'  # Intel MKL
os.environ['OPENBLAS_NUM_THREADS'] = '4'  # OpenBLAS
os.environ['OMP_NUM_THREADS'] = '4'  # OpenMP

# Check NumPy configuration
np.show_config()  # Shows BLAS implementation details
```

Trang trình bày 2: Phân tích tác động của cấu hình luồng

Việc hiểu tác động của cấu hình luồng yêu cầu các hoạt động ma trận điểm chuẩn trong các cài đặt khác nhau. Mã này trình bày cách đo các biến thể hiệu suất với số lượng luồng khác nhau bằng cách sử dụng phép nhân ma trận làm ví dụ.

```python
import time
import numpy as np
from contextlib import contextmanager

@contextmanager
def thread_config(num_threads):
    # Store original settings
    original_mkl = os.environ.get('MKL_NUM_THREADS')
    original_openblas = os.environ.get('OPENBLAS_NUM_THREADS')
    original_omp = os.environ.get('OMP_NUM_THREADS')

    # Set new thread count
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['OMP_NUM_THREADS'] = str(num_threads)

    try:
        yield
    finally:
        # Restore original settings
        if original_mkl:
            os.environ['MKL_NUM_THREADS'] = original_mkl
        if original_openblas:
            os.environ['OPENBLAS_NUM_THREADS'] = original_openblas
        if original_omp:
            os.environ['OMP_NUM_THREADS'] = original_omp
```

Slide 3: Chức năng đo điểm chuẩn hiệu suất

Chức năng đo điểm chuẩn toàn diện giúp xác định cấu hình luồng tối ưu cho các hoạt động ma trận cụ thể. Việc triển khai này đo lường thời gian thực hiện trên số lượng luồng và kích thước ma trận khác nhau.

```python
def benchmark_matrix_operation(sizes, thread_counts):
    results = {}

    for size in sizes:
        results[size] = {}
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)

        for threads in thread_counts:
            with thread_config(threads):
                start_time = time.perf_counter()
                C = np.dot(A, B)  # Matrix multiplication
                duration = time.perf_counter() - start_time
                results[size][threads] = duration

    return results

# Example usage
sizes = [1000, 2000, 4000]
thread_counts = [1, 2, 4, 8]
results = benchmark_matrix_operation(sizes, thread_counts)
```

Trang trình bày 4: Trực quan hóa hiệu suất của luồng

Tạo trực quan hóa giúp hiểu mối quan hệ giữa số lượng luồng và hiệu suất cho các kích thước ma trận khác nhau. Việc triển khai này sử dụng matplotlib để tạo ra các biểu đồ hiệu suất so sánh.

```python
import matplotlib.pyplot as plt

def plot_thread_performance(results):
    plt.figure(figsize=(12, 6))

    for size in results:
        threads = list(results[size].keys())
        times = list(results[size].values())
        plt.plot(threads, times, marker='o', label=f'Matrix Size: {size}x{size}')

    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Matrix Multiplication Performance vs Thread Count')
    plt.legend()
    plt.grid(True)
    plt.show()

# Generate performance visualization
plot_thread_performance(results)
```

Trang trình bày 5: Trình quản lý bối cảnh nhận biết luồng

Việc triển khai trình quản lý bối cảnh để thay đổi cấu hình luồng tạm thời cho phép thử nghiệm an toàn với các cài đặt luồng khác nhau mà không ảnh hưởng đến môi trường chung.

```python
class ThreadConfiguration:
    def __init__(self, num_threads):
        self.num_threads = str(num_threads)
        self.original_config = {}

    def __enter__(self):
        # Save current configuration
        thread_vars = ['MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']
        for var in thread_vars:
            self.original_config[var] = os.environ.get(var)
            os.environ[var] = self.num_threads
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original configuration
        for var, value in self.original_config.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
```

Slide 6: Real-world Example: Image Processing Pipeline

A practical demonstration of thread optimization in image processing tasks using NumPy. This implementation shows how different thread configurations affect the performance of common image operations.

```python
import numpy as np
from PIL import Image
import time

def process_image_batch(images, thread_counts):
    results = {}

    for threads in thread_counts:
        with ThreadConfiguration(threads):
            start_time = time.perf_counter()

            # Simulate batch processing
            processed = []
            for img in images:
                # Convert to numpy array
                img_array = np.array(img)

                # Apply various transformations
                filtered = np.fft.fft2(img_array)  # FFT
                inverse = np.fft.ifft2(filtered)   # Inverse FFT
                processed.append(inverse.real)

            duration = time.perf_counter() - start_time
            results[threads] = duration

    return results
```

Trang trình bày 7: Mã nguồn phân tích xử lý ảnh

```python
def analyze_image_processing_performance():
    # Generate synthetic images
    images = []
    for _ in range(10):
        img_array = np.random.rand(1024, 1024)
        images.append(img_array)

    # Test different thread configurations
    thread_counts = [1, 2, 4, 8, 16]
    results = process_image_batch(images, thread_counts)

    # Print results
    print("\nImage Processing Performance Results:")
    print("-" * 40)
    for threads, duration in results.items():
        print(f"Threads: {threads:2d} | Time: {duration:.4f} seconds")

    return results

# Example usage
performance_results = analyze_image_processing_performance()
```

Trang trình bày 8: Hoạt động giới hạn bộ nhớ và giới hạn CPU

Hiểu được sự khác biệt giữa các hoạt động giới hạn bộ nhớ và giới hạn CPU là rất quan trọng để có cấu hình luồng tối ưu. Các loại hoạt động khác nhau được hưởng lợi từ số lượng luồng khác nhau dựa trên yêu cầu tài nguyên của chúng.

```python
def compare_operation_types(size=5000):
    operations = {
        'memory_bound': lambda: np.sum(np.random.rand(size, size)),
        'cpu_bound': lambda: np.linalg.svd(np.random.rand(size, size))
    }

    thread_counts = [1, 2, 4, 8]
    results = {op: {} for op in operations}

    for op_name, operation in operations.items():
        for threads in thread_counts:
            with ThreadConfiguration(threads):
                start_time = time.perf_counter()
                operation()
                duration = time.perf_counter() - start_time
                results[op_name][threads] = duration

    return results
```

Slide 9: Automatic Thread Optimization

Implementing an automatic thread optimizer that determines the optimal thread count for specific operations through iterative testing and performance measurement.

```python
class ThreadOptimizer:
    def __init__(self, max_threads=16):
        self.max_threads = max_threads
        self.thread_cache = {}

    def find_optimal_threads(self, operation, *args, **kwargs):
        # Create a unique key for this operation
        op_key = f"{operation.__name__}_{hash(str(args))}"

        if op_key in self.thread_cache:
            return self.thread_cache[op_key]

        best_time = float('inf')
        optimal_threads = 1

        for threads in range(1, self.max_threads + 1):
            with ThreadConfiguration(threads):
                times = []
                # Run multiple trials
                for _ in range(3):
                    start = time.perf_counter()
                    operation(*args, **kwargs)
                    times.append(time.perf_counter() - start)

                avg_time = np.mean(times)
                if avg_time < best_time:
                    best_time = avg_time
                    optimal_threads = threads

        self.thread_cache[op_key] = optimal_threads
        return optimal_threads
```

Trang trình bày 10: Triển khai quy mô luồng động

Một hệ thống điều chỉnh linh hoạt số lượng luồng dựa trên tải hệ thống và tài nguyên sẵn có, đảm bảo hiệu suất tối ưu trong các điều kiện khối lượng công việc khác nhau.

```python
import psutil
import multiprocessing

class DynamicThreadScaler:
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()

    def get_optimal_threads(self, operation_type='default'):
        # Get current CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent

        # Base calculation of available threads
        available_threads = max(1, self.cpu_count - int(cpu_usage / 100 * self.cpu_count))

        # Adjust based on operation type
        if operation_type == 'memory_bound':
            # Reduce threads if memory usage is high
            if memory_usage > 80:
                available_threads = max(1, available_threads // 2)
        elif operation_type == 'cpu_bound':
            # Ensure minimum threads for CPU-bound operations
            available_threads = max(2, available_threads)

        return available_threads
```

Trang trình bày 11: Giám sát và ghi nhật ký hiệu suất

Việc triển khai hệ thống giám sát toàn diện để theo dõi hiệu suất luồng theo thời gian giúp đưa ra quyết định sáng suốt về điều chỉnh cấu hình luồng cho các loại hoạt động khác nhau.

```python
import logging
from datetime import datetime
import json

class ThreadPerformanceMonitor:
    def __init__(self, log_file='thread_performance.log'):
        self.log_file = log_file
        self.setup_logging()
        self.performance_history = {}

    def setup_logging(self):
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )

    def record_performance(self, operation_name, matrix_size, thread_count, execution_time):
        timestamp = datetime.now().isoformat()
        entry = {
            'timestamp': timestamp,
            'operation': operation_name,
            'matrix_size': matrix_size,
            'threads': thread_count,
            'execution_time': execution_time
        }

        # Log the entry
        logging.info(json.dumps(entry))

        # Update performance history
        key = f"{operation_name}_{matrix_size}"
        if key not in self.performance_history:
            self.performance_history[key] = []
        self.performance_history[key].append(entry)
```

Slide 12: Bộ công cụ phân tích luồng nâng cao

Bộ công cụ toàn diện để phân tích hiệu suất luồng trên các loại hoạt động NumPy khác nhau, cung cấp thông tin chi tiết về cấu hình luồng tối ưu.

```python
class ThreadAnalysisToolkit:
    def __init__(self):
        self.monitor = ThreadPerformanceMonitor()
        self.optimizer = ThreadOptimizer()
        self.scaler = DynamicThreadScaler()

    def analyze_operation(self, operation, sizes, thread_range):
        results = {}

        for size in sizes:
            results[size] = {}
            for threads in thread_range:
                with ThreadConfiguration(threads):
                    # Warm-up run
                    operation(size)

                    # Actual measurement
                    start_time = time.perf_counter()
                    operation(size)
                    duration = time.perf_counter() - start_time

                    # Record performance
                    self.monitor.record_performance(
                        operation.__name__,
                        size,
                        threads,
                        duration
                    )
                    results[size][threads] = duration

        return results
```

Slide 13: Results Analysis and Visualization

```python
def visualize_thread_analysis(results):
    import seaborn as sns

    # Prepare data for plotting
    plot_data = []
    for size in results:
        for threads, time in results[size].items():
            plot_data.append({
                'Matrix Size': size,
                'Threads': threads,
                'Execution Time': time
            })

    # Create heatmap
    df = pd.DataFrame(plot_data)
    pivot_table = df.pivot('Matrix Size', 'Threads', 'Execution Time')

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Thread Performance Analysis')
    plt.xlabel('Number of Threads')
    plt.ylabel('Matrix Size')
    plt.show()
```

Trang trình bày 14: Tài nguyên bổ sung

* Tối ưu hóa các hoạt động của mảng NumPy: Tìm kiếm "Tối ưu hóa NumPy: Mẹo và thủ thuật cho tính toán số"
* Phân tích hiệu suất của các hoạt động NumPy đa luồng: [http://www.google.com/search?q=numpy+multithreading+performance+analysis](http://www.google.com/search?q=numpy+multithreading+performance+analysis)
* Kỹ thuật tối ưu luồng trong tính toán khoa học:
    * [https://scicomp.stackexchange.com/questions/tagged/numpy+parallel-computing](https://scicomp.stackexchange.com/questions/tagged/numpy+parallel-computing)
    * [https://numpy.org/doc/stable/reference/routines.linalg.html](https://numpy.org/doc/stable/reference/routines.linalg.html)
    * [https://scipy-lectures.org/advanced/optimizing/](https://scipy-lectures.org/advanced/optimizing/)
* Cụm từ tìm kiếm được đề xuất để nghiên cứu thêm:
    * "Kỹ thuật tối ưu hóa NumPy BLAS"
    * "Hiệu suất đại số tuyến tính đa luồng"
    * "So sánh luồng OpenBLAS và MKL"
