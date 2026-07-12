## Giải thích Trình tạo trong Python bằng Fibonacci
Slide 1: Khái niệm nhà máy phát điện

Trình tạo trong Python là một loại hàm đặc biệt cho phép đánh giá các giá trị một cách lười biếng, chỉ tạo các phần tử khi cần thiết. Không giống như các hàm thông thường trả về tất cả các giá trị cùng một lúc, các trình tạo tạo ra từng giá trị một, giúp chúng tiết kiệm bộ nhớ cho các chuỗi lớn.

```python
def simple_generator(n):
    """Basic generator that yields numbers from 0 to n-1"""
    current = 0
    while current < n:
        yield current
        current += 1

# Example usage
gen = simple_generator(3)
print(next(gen))  # Output: 0
print(next(gen))  # Output: 1
print(next(gen))  # Output: 2
```

Trang trình bày 2: Triển khai Trình tạo Fibonacci

Trình tạo Fibonacci thể hiện sức mạnh của việc đánh giá lười biếng bằng cách tạo ra các số Fibonacci theo yêu cầu. Cách tiếp cận này đặc biệt hiệu quả vì nó chỉ tính toán các giá trị khi được yêu cầu, duy trì mức sử dụng bộ nhớ tối thiểu bất kể độ dài chuỗi.

```python
def fibonacci_generator(n):
    """Generates n Fibonacci numbers"""
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

# Example usage
fib = fibonacci_generator(5)
sequence = [next(fib) for _ in range(5)]
print(sequence)  # Output: [0, 1, 1, 2, 3]
```

Slide 3: Quản lý trạng thái máy phát điện

Trình tạo duy trì trạng thái nội bộ giữa các lệnh gọi, ghi nhớ vị trí cuối cùng và tất cả các biến cục bộ. Tính năng này khiến chúng trở nên lý tưởng để triển khai các thuật toán lặp trong đó việc bảo toàn trạng thái là rất quan trọng để tạo ra giá trị tiếp theo.

```python
def stateful_generator():
    """Demonstrates state preservation in generators"""
    state = 0
    while True:
        received = yield state
        state += 10 if received else 1

# Example usage
gen = stateful_generator()
print(next(gen))      # Output: 0
print(gen.send(True)) # Output: 10
print(gen.send(False))# Output: 11
```

Slide 4: Trình tạo chuỗi toán học

Một ứng dụng thực tế của máy phát điện để tính toán chuỗi toán học cho thấy tiện ích của chúng trong tính toán toán học. Việc triển khai này cho thấy cách tạo ra các số hạng của chuỗi lũy thừa một cách hiệu quả.

```python
def power_series_generator(x, terms):
    """Generates terms of the power series for e^x"""
    n = 0
    factorial = 1
    while n < terms:
        term = (x ** n) / factorial
        yield term
        n += 1
        factorial *= (n + 1)

# Calculate first 5 terms of e^2
series = power_series_generator(2, 5)
partial_sum = sum([next(series) for _ in range(5)])
print(f"Partial sum: {partial_sum}")  # Output approximates e^2
```

Trang trình bày 5: Tối ưu hóa biểu thức trình tạo

Biểu thức trình tạo cung cấp cú pháp ngắn gọn hơn để tạo trình tạo, mang lại hiệu quả bộ nhớ so với việc hiểu danh sách. Điều này đặc biệt hữu ích khi làm việc với tập dữ liệu lớn hoặc chuỗi vô hạn.

```python
# Memory-efficient generator expression
gen_exp = (x**2 for x in range(10**6))
print(next(gen_exp))  # Output: 0
print(next(gen_exp))  # Output: 1

# Memory comparison
import sys
list_comp = [x**2 for x in range(10**6)]
gen_exp = (x**2 for x in range(10**6))
print(f"List size: {sys.getsizeof(list_comp)}")
print(f"Generator size: {sys.getsizeof(gen_exp)}")
```

Trang trình bày 6: Trình tạo chuỗi vô hạn

Trình tạo vượt trội trong việc xử lý các chuỗi vô hạn vì chúng chỉ tạo ra các giá trị khi được yêu cầu. Việc triển khai này cho thấy cách tạo trình tạo chuỗi vô hạn trong khi vẫn duy trì mức sử dụng bộ nhớ liên tục.

```python
def infinite_primes():
    """Generates an infinite sequence of prime numbers"""
    def is_prime(n):
        return all(n % i != 0 for i in range(2, int(n**0.5) + 1))

    n = 2
    while True:
        if is_prime(n):
            yield n
        n += 1

# Example usage
primes = infinite_primes()
first_five = [next(primes) for _ in range(5)]
print(first_five)  # Output: [2, 3, 5, 7, 11]
```

Slide 7: Thi công đường ống máy phát điện

Quy trình tạo dữ liệu cho phép tạo quy trình xử lý dữ liệu phức tạp trong đó mỗi trình tạo chuyển đổi dữ liệu từ dữ liệu trước đó. Mẫu này cho phép xử lý các tập dữ liệu lớn một cách hiệu quả về mặt bộ nhớ thông qua việc kết hợp.

```python
def read_data():
    """Simulates reading large data"""
    for i in range(1000):
        yield i

def filter_even(numbers):
    """Filters even numbers"""
    for num in numbers:
        if num % 2 == 0:
            yield num

def multiply_by_three(numbers):
    """Multiplies each number by 3"""
    for num in numbers:
        yield num * 3

# Pipeline construction
data = read_data()
filtered = filter_even(data)
result = multiply_by_three(filtered)

# Process first 5 results
print([next(result) for _ in range(5)])  # Output: [0, 6, 12, 18, 24]
```

Trang trình bày 8: Ví dụ thực tế - Truyền dữ liệu

Việc triển khai thực tế này thể hiện việc sử dụng trình tạo để xử lý các tập dữ liệu lớn theo khối, mô phỏng các kịch bản truyền dữ liệu theo thời gian thực phổ biến trong các ứng dụng kỹ thuật dữ liệu.

```python
def stream_data_processor(chunk_size=1000):
    """Simulates processing streaming data in chunks"""
    def generate_data():
        for i in range(10000):
            yield {'id': i, 'value': i * 2}

    def process_chunk(chunk):
        return sum(item['value'] for item in chunk)

    current_chunk = []
    for item in generate_data():
        current_chunk.append(item)
        if len(current_chunk) == chunk_size:
            yield process_chunk(current_chunk)
            current_chunk = []

    if current_chunk:  # Process remaining items
        yield process_chunk(current_chunk)

# Example usage
processor = stream_data_processor(chunk_size=2500)
chunk_sums = list(processor)
print(f"Number of chunks processed: {len(chunk_sums)}")
print(f"Sum of all chunks: {sum(chunk_sums)}")
```

Trang trình bày 9: Phân tích chuỗi thời gian dựa trên máy phát điện

Triển khai thực tế cho thấy cách sử dụng trình tạo để phân tích chuỗi thời gian, thể hiện phép tính trung bình động với mức sử dụng bộ nhớ tối thiểu.

```python
from collections import deque
from datetime import datetime, timedelta

def moving_average_generator(window_size):
    """Generates moving averages for streaming time series data"""
    window = deque(maxlen=window_size)

    while True:
        new_value = yield None if len(window) < window_size else sum(window)/window_size
        window.append(new_value)

# Example usage with time series data
def simulate_time_series():
    start_time = datetime.now()
    for i in range(100):
        yield (start_time + timedelta(minutes=i), i + (i % 5))

# Process time series
ma_gen = moving_average_generator(5)
next(ma_gen)  # Initialize generator

for timestamp, value in simulate_time_series():
    ma = ma_gen.send(value)
    if ma is not None:
        print(f"{timestamp}: MA = {ma:.2f}")
```

Trang trình bày 10: Xử lý dữ liệu hiệu quả về bộ nhớ

Triển khai này cho thấy cách sử dụng trình tạo để xử lý các tập dữ liệu lớn một cách hiệu quả, thể hiện sự tối ưu hóa mức sử dụng bộ nhớ cho các kịch bản dữ liệu lớn.

```python
def process_large_dataset(filename, chunk_size=1000):
    """Process large datasets in memory-efficient chunks"""
    def read_chunks():
        with open(filename, 'r') as f:
            chunk = []
            for line in f:
                chunk.append(float(line.strip()))
                if len(chunk) == chunk_size:
                    yield chunk
                    chunk = []
            if chunk:
                yield chunk

    def calculate_statistics(chunk):
        return {
            'count': len(chunk),
            'mean': sum(chunk) / len(chunk),
            'max': max(chunk),
            'min': min(chunk)
        }

    for chunk in read_chunks():
        yield calculate_statistics(chunk)

# Example usage with file creation
import random

# Create sample data file
with open('large_dataset.txt', 'w') as f:
    for _ in range(10000):
        f.write(f"{random.random()}\n")

# Process data
stats_generator = process_large_dataset('large_dataset.txt')
chunk_stats = list(stats_generator)
print(f"Processed {len(chunk_stats)} chunks")
```

Trang trình bày 11: Mẫu vòng lặp tùy chỉnh dựa trên trình tạo

Trình tạo cung cấp một cách thức dễ dàng để triển khai các trình vòng lặp tùy chỉnh, đơn giản hóa việc triển khai các mẫu lặp phức tạp trong khi vẫn duy trì mã rõ ràng, dễ đọc. Ví dụ này minh họa một trình vòng lặp phạm vi tùy chỉnh có điều khiển bước.

```python
def custom_range_iterator(start, end, step=1):
    """Custom range iterator with dynamic step control"""
    current = start
    step_size = step

    while current < end:
        step_control = yield current
        if step_control is not None:
            step_size = step_control
        current += step_size

# Example usage
def demonstrate_custom_range():
    iterator = custom_range_iterator(0, 10)
    next(iterator)  # Initialize

    print(iterator.send(None))    # Output: 0
    print(iterator.send(2))       # Change step to 2
    print(iterator.send(None))    # Continue with step 2
    print(iterator.send(0.5))     # Change step to 0.5

demonstrate_custom_range()
```

Slide 12: Xử lý dữ liệu thời gian thực bằng Generator

Việc triển khai này cho thấy cách sử dụng trình tạo để xử lý dữ liệu theo thời gian thực, cho thấy ứng dụng thực tế trong việc giám sát và phân tích dữ liệu truyền phát với độ trễ tối thiểu.

```python
import time
from collections import deque

def sensor_data_generator(sampling_rate=0.1):
    """Simulates continuous sensor data stream"""
    while True:
        yield {
            'timestamp': time.time(),
            'temperature': 20 + (time.time() % 5),
            'humidity': 50 + (time.time() % 10)
        }
        time.sleep(sampling_rate)

def analyze_sensor_data(window_size=5):
    """Analyzes streaming sensor data"""
    temp_window = deque(maxlen=window_size)
    humid_window = deque(maxlen=window_size)

    sensor = sensor_data_generator()

    while True:
        data = next(sensor)
        temp_window.append(data['temperature'])
        humid_window.append(data['humidity'])

        if len(temp_window) == window_size:
            yield {
                'avg_temp': sum(temp_window) / window_size,
                'avg_humidity': sum(humid_window) / window_size,
                'timestamp': data['timestamp']
            }

# Example usage
analyzer = analyze_sensor_data()
for _ in range(3):  # Analyze 3 windows of data
    analysis = next(analyzer)
    print(f"Time: {analysis['timestamp']:.2f}, "
          f"Avg Temp: {analysis['avg_temp']:.2f}, "
          f"Avg Humidity: {analysis['avg_humidity']:.2f}")
```

Slide 13: Trình tạo chuỗi toán học nâng cao

Một triển khai phức tạp của bộ tạo chuỗi toán học có thể xử lý nhiều loại chuỗi khác nhau với các tham số có thể định cấu hình và tiêu chí hội tụ.

```python
def math_series_generator(series_type='geometric', first_term=1, ratio=0.5, tolerance=1e-10):
    """
    Generates terms of various mathematical series
    Supports: geometric, arithmetic, and harmonic series
    """
    current_term = first_term
    position = 1

    while abs(current_term) > tolerance:
        yield current_term

        if series_type == 'geometric':
            current_term *= ratio
        elif series_type == 'arithmetic':
            current_term += ratio
        elif series_type == 'harmonic':
            position += 1
            current_term = first_term / position

# Example usage
def demonstrate_series():
    # Geometric series
    geometric = math_series_generator(series_type='geometric')
    geometric_terms = [next(geometric) for _ in range(5)]

    # Harmonic series
    harmonic = math_series_generator(series_type='harmonic')
    harmonic_terms = [next(harmonic) for _ in range(5)]

    print(f"Geometric series: {geometric_terms}")
    print(f"Harmonic series: {harmonic_terms}")

demonstrate_series()
```

Trang trình bày 14: Tài nguyên bổ sung

* "Các mẫu trình tạo trong Python" - [https://www.python.org/dev/peps/pep-0255/](https://www.python.org/dev/peps/pep-0255/)
* "Tìm hiểu về trình tạo Python" - [https://docs.python.org/3/howto/function.html](https://docs.python.org/3/howto/function.html)
* "Quản lý bộ nhớ bằng Python" - [https://docs.python.org/3/c-api/memory.html](https://docs.python.org/3/c-api/memory.html)
* "Xử lý dữ liệu hiệu quả trong Python" - [https://realpython.com/introduction-to-python-generators/](https://realpython.com/introduction-to-python-generators/)
* "Mẫu lặp nâng cao" - [https://www.python.org/dev/peps/pep-0289/](https://www.python.org/dev/peps/pep-0289/)
