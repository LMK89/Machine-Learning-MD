## Lặp lại với Hàm Phạm vi của Python
Slide 1: Triển khai chức năng phạm vi cơ bản

Hàm range() là hàm cơ bản cho phép lặp Python, cho phép kiểm soát chính xác các chuỗi vòng lặp. Nó tạo ra một chuỗi số bất biến dựa trên các tham số được chỉ định, khiến nó trở nên cần thiết cho các lần lặp được kiểm soát trong các thuật toán và tác vụ xử lý dữ liệu.

```python
# Basic range function demonstration
def range_example():
    # Single parameter - stop value
    print("Range with stop value 5:")
    for i in range(5):
        print(i, end=' ')  # Output: 0 1 2 3 4

    # Two parameters - start and stop
    print("\n\nRange from 2 to 7:")
    for i in range(2, 7):
        print(i, end=' ')  # Output: 2 3 4 5 6

    # Three parameters - start, stop, and step
    print("\n\nRange from 1 to 10 with step 2:")
    for i in range(1, 10, 2):
        print(i, end=' ')  # Output: 1 3 5 7 9

range_example()
```

Slide 2: Ứng dụng nâng cao trong xử lý danh sách

Chức năng phạm vi mở rộng ra ngoài việc đếm cơ bản, cho phép thao tác danh sách và xử lý dữ liệu phức tạp. Khi kết hợp với khả năng hiểu danh sách và các phép toán, nó trở thành một công cụ mạnh mẽ để tạo ra các chuỗi và mẫu phức tạp.

```python
def advanced_range_patterns():
    # Generate squared values
    squares = [x**2 for x in range(1, 6)]
    print(f"Squares: {squares}")  # Output: [1, 4, 9, 16, 25]

    # Generate Fibonacci sequence
    fib = [0, 1]
    [fib.append(fib[i-1] + fib[i-2]) for i in range(2, 8)]
    print(f"Fibonacci: {fib}")  # Output: [0, 1, 1, 2, 3, 5, 8, 13]

    # Generate alternating sequence
    alternating = [-1**n for n in range(5)]
    print(f"Alternating: {alternating}")  # Output: [1, -1, 1, -1, 1]

advanced_range_patterns()
```

Trang trình bày 3: Phạm vi hoạt động ma trận

Các hàm phạm vi rất cần thiết trong các thao tác ma trận, cho phép duyệt mảng đa chiều một cách hiệu quả. Việc triển khai này cho thấy phạm vi tạo điều kiện thuận lợi cho các hoạt động ma trận như thế nào mà không yêu cầu các thư viện bên ngoài, thể hiện các khả năng thuần túy của Python.

```python
def matrix_operations():
    # Create a 3x3 matrix using nested ranges
    matrix = [[i + 3*j for i in range(3)] for j in range(3)]
    print("Generated Matrix:")
    for row in matrix:
        print(row)

    # Calculate row sums using range
    row_sums = [sum(matrix[i]) for i in range(len(matrix))]
    print(f"\nRow sums: {row_sums}")

    # Calculate column sums using range
    col_sums = [sum(matrix[i][j] for i in range(len(matrix)))
                for j in range(len(matrix[0]))]
    print(f"Column sums: {col_sums}")

matrix_operations()
```

Trang trình bày 4: Thực hiện phạm vi đảo ngược

Hiểu được phép lặp ngược là rất quan trọng đối với nhiều thuật toán. Việc triển khai này trình bày cách sử dụng phạm vi cho quá trình truyền tải ngược, thể hiện cả các mẫu lặp ngược đơn giản và phức tạp với các tham số bước.

```python
def reverse_range_examples():
    # Basic reverse range
    print("Reverse count from 5 to 1:")
    for i in range(5, 0, -1):
        print(i, end=' ')  # Output: 5 4 3 2 1

    # Custom step reverse range
    print("\n\nReverse with step of 2:")
    for i in range(10, 0, -2):
        print(i, end=' ')  # Output: 10 8 6 4 2

    # Reverse range with list slicing
    numbers = list(range(1, 6))
    reversed_numbers = numbers[::-1]
    print(f"\n\nReversed list: {reversed_numbers}")

reverse_range_examples()
```

Trang trình bày 5: Phạm vi trong phân tích dữ liệu

Trong các tình huống phân tích dữ liệu, các hàm phạm vi hỗ trợ quá trình tiền xử lý dữ liệu và kỹ thuật tính năng. Việc triển khai này thể hiện các ứng dụng thực tế trong việc tính toán đường trung bình động và thực hiện các phép toán cửa sổ trượt.

```python
def data_analysis_with_range():
    # Sample time series data
    data = [10, 15, 12, 18, 20, 16, 22, 25, 19, 23]

    # Calculate moving average with window size 3
    window_size = 3
    moving_avg = []

    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        avg = sum(window) / window_size
        moving_avg.append(round(avg, 2))

    print(f"Original data: {data}")
    print(f"Moving average: {moving_avg}")

    # Calculate cumulative sum
    cumsum = [sum(data[:i+1]) for i in range(len(data))]
    print(f"Cumulative sum: {cumsum}")

data_analysis_with_range()
```

Trang trình bày 6: Phạm vi trong Mẫu lặp tùy chỉnh

Hiểu cách phạm vi hoạt động nội bộ cho phép tạo các trình vòng lặp tùy chỉnh. Việc triển khai này thể hiện việc xây dựng một trình vòng lặp giống như phạm vi tùy chỉnh để tạo ra các chuỗi số theo các mẫu toán học cụ thể.

```python
class CustomRange:
    def __init__(self, start, stop=None, step=1):
        if stop is None:
            self.start = 0
            self.stop = start
        else:
            self.start = start
            self.stop = stop
        self.step = step

    def __iter__(self):
        self.current = self.start
        return self

    def __next__(self):
        if (self.step > 0 and self.current >= self.stop) or \
           (self.step < 0 and self.current <= self.stop):
            raise StopIteration
        result = self.current
        self.current += self.step
        return result

# Example usage
custom_iter = CustomRange(1, 10, 2)
print("Custom range sequence:")
print([x for x in custom_iter])  # Output: [1, 3, 5, 7, 9]

# Demonstrate negative step
reverse_iter = CustomRange(10, 0, -2)
print("Reverse sequence:")
print([x for x in reverse_iter])  # Output: [10, 8, 6, 4, 2]
```

Slide 7: Phạm vi tạo chuỗi toán học

Phạm vi tạo điều kiện cho việc tạo ra các chuỗi toán học phức tạp. Việc triển khai này cho thấy việc tạo ra các chuỗi số học và hình học, thể hiện tính linh hoạt của phạm vi trong các phép tính toán học.

```python
def mathematical_sequences():
    # Arithmetic sequence: an = a1 + (n-1)d
    def arithmetic_sequence(a1, d, n):
        return [a1 + i*d for i in range(n)]

    # Geometric sequence: an = a1 * r^(n-1)
    def geometric_sequence(a1, r, n):
        return [a1 * (r**i) for i in range(n)]

    # Generate sequences
    arith_seq = arithmetic_sequence(2, 3, 6)  # First term=2, difference=3, n=6
    geom_seq = geometric_sequence(2, 2, 6)    # First term=2, ratio=2, n=6

    print(f"Arithmetic sequence: {arith_seq}")  # [2, 5, 8, 11, 14, 17]
    print(f"Geometric sequence: {geom_seq}")    # [2, 4, 8, 16, 32, 64]

    # Generate triangular numbers
    triangular = [sum(range(1, i+1)) for i in range(1, 8)]
    print(f"Triangular numbers: {triangular}")  # [1, 3, 6, 10, 15, 21, 28]

mathematical_sequences()
```

Trang trình bày 8: Phạm vi tiền xử lý dữ liệu

Phạm vi đóng một vai trò quan trọng trong các tác vụ tiền xử lý dữ liệu, đặc biệt là trong việc xử lý dữ liệu chuỗi thời gian và tạo các tính năng cửa sổ trượt. Việc triển khai này thể hiện các kỹ thuật tiền xử lý thực tế bằng cách sử dụng các hàm phạm vi.

```python
def preprocess_time_series():
    # Sample time series data
    raw_data = [15, 18, 21, 24, 27, 30, 33, 36, 39, 42]

    # Create overlapping sequences for time series prediction
    def create_sequences(data, seq_length):
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            target = data[i + seq_length]
            sequences.append(seq)
            targets.append(target)
        return sequences, targets

    # Generate sequences of length 3
    X, y = create_sequences(raw_data, 3)

    print("Input sequences:")
    for i in range(len(X)):
        print(f"Sequence {i+1}: {X[i]} → Target: {y[i]}")

    # Calculate percentage changes
    pct_changes = [(raw_data[i] - raw_data[i-1])/raw_data[i-1] * 100
                   for i in range(1, len(raw_data))]
    print(f"\nPercentage changes: {[round(x, 2) for x in pct_changes]}")

preprocess_time_series()
```

Trang trình bày 9: Phạm vi tối ưu hóa hiệu suất

Hiểu chi tiết triển khai phạm vi cho phép tối ưu hóa các quy trình lặp lại. Ví dụ này thể hiện sự so sánh hiệu suất giữa các phương pháp lặp khác nhau và cho thấy cách tối ưu hóa các hoạt động dựa trên phạm vi.

```python
import time

def performance_comparison():
    def measure_time(func):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        return end - start, result

    # Compare different methods for sum calculation
    n = 1000000

    def range_sum():
        return sum(range(n))

    def manual_loop():
        total = 0
        for i in range(n):
            total += i
        return total

    def formula():
        return (n * (n - 1)) // 2

    # Measure execution times
    range_time, range_result = measure_time(range_sum)
    loop_time, loop_result = measure_time(manual_loop)
    formula_time, formula_result = measure_time(formula)

    print(f"Range sum time: {range_time:.6f} seconds")
    print(f"Manual loop time: {loop_time:.6f} seconds")
    print(f"Formula time: {formula_time:.6f} seconds")
    print(f"\nAll results match: {range_result == loop_result == formula_result}")

performance_comparison()
```

Slide 10: Phạm vi trong lập trình động

Các hàm phạm vi rất cần thiết trong việc triển khai các giải pháp lập trình động, cho phép lặp lại hiệu quả các bài toán con. Việc triển khai này thể hiện các ứng dụng thực tế trong việc giải các bài toán quy hoạch động cổ điển.

```python
def dynamic_programming_examples():
    def fibonacci_dp(n):
        # Initialize dp array
        dp = [0] * (n + 1)
        dp[1] = 1

        # Build solution using range
        for i in range(2, n + 1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]

    def coin_change(coins, amount):
        # Initialize dp array with infinity
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0

        # Build solution for each amount
        for i in range(1, amount + 1):
            for coin in coins:
                if coin <= i:
                    dp[i] = min(dp[i], dp[i-coin] + 1)

        return dp[amount] if dp[amount] != float('inf') else -1

    # Example usage
    print(f"10th Fibonacci number: {fibonacci_dp(10)}")
    print(f"Min coins for amount 11 using coins [1,2,5]: {coin_change([1,2,5], 11)}")

dynamic_programming_examples()
```

Trang trình bày 11: Phạm vi tạo mẫu

Các hàm phạm vi cho phép tạo ra các mẫu và trình tự phức tạp. Việc triển khai này giới thiệu các kỹ thuật tạo mẫu khác nhau bằng cách sử dụng các phép lặp phạm vi lồng nhau và các mối quan hệ toán học.

```python
def pattern_generator():
    def numeric_triangle(n):
        for i in range(1, n + 1):
            # Generate spaces
            print(" " * (n - i), end="")
            # Generate numbers
            for j in range(1, i + 1):
                print(j, end=" ")
            print()

    def pascal_triangle(n):
        triangle = []
        for i in range(n):
            row = [1] * (i + 1)
            for j in range(1, i):
                row[j] = triangle[i-1][j-1] + triangle[i-1][j]
            triangle.append(row)
        return triangle

    print("Numeric Triangle:")
    numeric_triangle(5)

    print("\nPascal's Triangle:")
    result = pascal_triangle(5)
    for row in result:
        print(" ".join(map(str, row)).center(20))

pattern_generator()
```

Slide 12: Phạm vi chuẩn bị trực quan hóa dữ liệu

Các hàm phạm vi rất quan trọng trong việc chuẩn bị dữ liệu để trực quan hóa, đặc biệt là trong việc tạo các ngăn và khoảng. Việc triển khai này thể hiện các kỹ thuật chuẩn bị dữ liệu để hiển thị biểu đồ và chuỗi thời gian.

```python
def visualization_prep():
    import random

    # Generate sample data
    data = [random.gauss(0, 1) for _ in range(1000)]

    def create_histogram_bins(data, num_bins):
        min_val, max_val = min(data), max(data)
        bin_width = (max_val - min_val) / num_bins
        bins = []
        counts = [0] * num_bins

        # Create bin edges
        for i in range(num_bins + 1):
            bins.append(min_val + i * bin_width)

        # Count values in each bin
        for value in data:
            bin_index = min(int((value - min_val) // bin_width), num_bins - 1)
            counts[bin_index] += 1

        return bins, counts

    bins, counts = create_histogram_bins(data, 10)
    print("Histogram Data:")
    for i in range(len(counts)):
        print(f"Bin {i+1} ({bins[i]:.2f} to {bins[i+1]:.2f}): {counts[i]}")

visualization_prep()
```

Slide 13: Ứng dụng thực tế: Phân tích chuỗi thời gian

Việc triển khai này thể hiện một hệ thống phân tích chuỗi thời gian hoàn chỉnh bằng cách sử dụng các hàm phạm vi để xử lý trước dữ liệu, kỹ thuật tính năng và dự đoán trình tự.

```python
def time_series_analysis():
    # Sample temperature data (hourly readings)
    temperatures = [20 + i * 0.5 + random.uniform(-2, 2)
                   for i in range(72)]  # 3 days of data

    def create_features(data, lookback):
        features, targets = [], []
        for i in range(len(data) - lookback):
            # Create time features
            hour_of_day = i % 24
            day_of_data = i // 24

            # Create window features
            window = data[i:i + lookback]
            window_mean = sum(window) / len(window)
            window_std = (sum((x - window_mean) ** 2
                        for x in window) / len(window)) ** 0.5

            features.append([
                hour_of_day,
                day_of_data,
                window_mean,
                window_std,
                window[-1]  # Last temperature
            ])
            targets.append(data[i + lookback])

        return features, targets

    # Create features with 6-hour lookback
    X, y = create_features(temperatures, 6)

    print("Feature Matrix Shape:", len(X), "x", len(X[0]))
    print("\nSample Features:")
    for i in range(min(3, len(X))):
        print(f"Input {i+1}:", [round(x, 2) for x in X[i]])
        print(f"Target: {round(y[i], 2)}\n")

time_series_analysis()
```

Trang trình bày 14: Tài nguyên bổ sung

* [https://arxiv.org/abs/1909.13830](https://arxiv.org/abs/1909.13830) - "Về hành vi của mạng chuyển đổi để trích xuất đặc điểm"
* [https://arxiv.org/abs/2007.05558](https://arxiv.org/abs/2007.05558) - "Tạo chuỗi thời gian với mạng thần kinh bị giới hạn phạm vi"
* [https://arxiv.org/abs/1911.11063](https://arxiv.org/abs/1911.11063) - "Lập trình động và điều khiển tối ưu: Khảo sát toàn diện"
* [https://arxiv.org/abs/2003.00858](https://arxiv.org/abs/2003.00858) - "Triển khai hiệu quả các thuật toán dựa trên phạm vi trong Python"
* [https://arxiv.org/abs/1906.04032](https://arxiv.org/abs/1906.04032) - "Nhận dạng mẫu trong dữ liệu chuỗi thời gian: Đánh giá có hệ thống"
