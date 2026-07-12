## Trình trang trí dựa trên lớp trong Python
Trang trình bày 1: Giới thiệu về Trang trí dựa trên lớp

Các trình trang trí dựa trên lớp đại diện cho một mẫu triển khai nâng cao trong Python cho phép các lớp sửa đổi hoặc nâng cao hoạt động của các hàm hoặc phương thức. Không giống như các trình trang trí hàm, các trình trang trí lớp duy trì trạng thái và cung cấp một cách tiếp cận hướng đối tượng hơn để mở rộng chức năng.

```python
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Call count: {self.count}")
        return self.func(*args, **kwargs)

@CountCalls
def example():
    return "Function executed"

# Usage example
print(example())  # Output: Call count: 1 \n Function executed
print(example())  # Output: Call count: 2 \n Function executed
```

Slide 2: Quản lý trạng thái trong trang trí

Trình trang trí lớp vượt trội trong việc duy trì trạng thái giữa các lệnh gọi hàm, cung cấp cơ chế mạnh mẽ để theo dõi lịch sử thực thi, lưu kết quả vào bộ đệm hoặc triển khai các mẫu hành vi phức tạp tồn tại qua nhiều lệnh gọi.

```python
class Memoize:
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args not in self.cache:
            self.cache[args] = self.func(*args)
        return self.cache[args]

@Memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Usage example
print(fibonacci(10))  # Output: 55 (computed once)
print(fibonacci(10))  # Output: 55 (retrieved from cache)
```

Slide 3: Phương pháp trang trí

Trình trang trí dựa trên lớp có thể sửa đổi cả phương thức phiên bản và phương thức lớp, yêu cầu xử lý cẩn thận tham số self và liên kết phương thức thích hợp để duy trì ngữ cảnh chính xác và khả năng truy cập của các thuộc tính phiên bản.

```python
class ValidateArguments:
    def __init__(self, func):
        self.func = func

    def __call__(self, instance, *args, **kwargs):
        if not args:
            raise ValueError("Method requires at least one argument")
        return self.func(instance, *args, **kwargs)

    def __get__(self, obj, objtype):
        import functools
        return functools.partial(self.__call__, obj)

class Example:
    @ValidateArguments
    def process_data(self, data):
        return f"Processing: {data}"

# Usage example
e = Example()
print(e.process_data("test"))  # Output: Processing: test
```

Slide 4: Trang trí nhà xưởng

Mẫu nhà máy trang trí cho phép các trình trang trí dựa trên lớp có thể tùy chỉnh có thể chấp nhận các tham số để sửa đổi hành vi của chúng, cung cấp một khung linh hoạt để tạo các trình trang trí chuyên dụng với các tính năng có thể định cấu hình.

```python
class Retry:
    def __init__(self, max_attempts=3, delay=1):
        self.max_attempts = max_attempts
        self.delay = delay

    def __call__(self, func):
        class Wrapper:
            def __init__(self, f):
                self.func = f

            def __call__(self, *args, **kwargs):
                import time
                attempts = 0
                while attempts < self.max_attempts:
                    try:
                        return self.func(*args, **kwargs)
                    except Exception as e:
                        attempts += 1
                        if attempts == self.max_attempts:
                            raise e
                        time.sleep(self.delay)
        return Wrapper(func)

@Retry(max_attempts=2, delay=0.1)
def unstable_operation():
    import random
    if random.random() < 0.5:
        raise ValueError("Random failure")
    return "Success"
```

Slide 5: Trang trí lớp có tham số

Trình trang trí lớp có thể chấp nhận các tham số khởi tạo để tùy chỉnh hành vi của chúng trong khi vẫn duy trì khả năng truy cập và sửa đổi chức năng được trang trí. Mẫu này cho phép cấu hình linh hoạt hành vi trang trí tại thời điểm trang trí.

```python
class RateLimiter:
    def __init__(self, calls_per_second=1):
        self.calls_per_second = calls_per_second
        self.last_called = {}

    def __call__(self, func):
        def wrapped(*args, **kwargs):
            import time
            now = time.time()
            if func in self.last_called:
                elapsed = now - self.last_called[func]
                if elapsed < (1.0 / self.calls_per_second):
                    time.sleep((1.0 / self.calls_per_second) - elapsed)
            result = func(*args, **kwargs)
            self.last_called[func] = time.time()
            return result
        return wrapped

@RateLimiter(calls_per_second=2)
def fast_operation():
    return "Operation executed"

# Usage
import time
start = time.time()
for _ in range(3):
    print(fast_operation())
print(f"Execution time: {time.time() - start:.2f} seconds")
```

Trang trình bày 6: Trang trí lớp lồng nhau

Nhiều trình trang trí lớp có thể được áp dụng cho một hàm duy nhất, tạo ra một chuỗi sửa đổi trong đó mỗi trình trang trí thêm chức năng riêng của nó. Hiểu thứ tự thực hiện là rất quan trọng để thực hiện đúng.

```python
class LogCalls:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print(f"Calling {self.func.__name__} with args: {args}, kwargs: {kwargs}")
        result = self.func(*args, **kwargs)
        print(f"Returned: {result}")
        return result

class TimeExecution:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        import time
        start = time.time()
        result = self.func(*args, **kwargs)
        print(f"Execution time: {time.time() - start:.4f} seconds")
        return result

@LogCalls
@TimeExecution
def complex_operation(x, y):
    import time
    time.sleep(0.1)
    return x + y

# Usage
result = complex_operation(3, 4)
```

Trang trình bày 7: Triển khai trang trí lớp để xác thực dữ liệu

Trình trang trí lớp cung cấp một giải pháp tinh tế để triển khai xác thực dữ liệu và kiểm tra kiểu, đảm bảo rằng các đối số của hàm đáp ứng các tiêu chí cụ thể trước khi tiến hành thực thi.

```python
class ValidateTypes:
    def __init__(self, *types):
        self.types = types

    def __call__(self, func):
        def wrapped(*args, **kwargs):
            if len(args) != len(self.types):
                raise ValueError("Invalid number of arguments")

            for arg, expected_type in zip(args, self.types):
                if not isinstance(arg, expected_type):
                    raise TypeError(f"Expected {expected_type}, got {type(arg)}")

            return func(*args, **kwargs)
        return wrapped

@ValidateTypes(int, int)
def add_numbers(x, y):
    return x + y

# Usage examples
try:
    print(add_numbers(1, 2))        # Works: 3
    print(add_numbers("1", 2))      # Raises TypeError
except TypeError as e:
    print(f"Error: {e}")
```

Trang trình bày 8: Trình trang trí lớp nhận biết ngữ cảnh

Trình trang trí nhận biết ngữ cảnh có thể sửa đổi hành vi của chúng dựa trên môi trường thời gian chạy hoặc trạng thái của đối tượng được trang trí, cung cấp khả năng thích ứng chức năng động.

```python
class EnvironmentAware:
    def __init__(self, func):
        self.func = func
        self.env_checks = {
            'development': lambda: self._dev_wrapper,
            'production': lambda: self._prod_wrapper
        }

    def __call__(self, *args, **kwargs):
        import os
        env = os.getenv('ENV', 'development')
        wrapper = self.env_checks.get(env, lambda: self.func)()
        return wrapper(*args, **kwargs)

    def _dev_wrapper(self, *args, **kwargs):
        print(f"DEBUG: Calling {self.func.__name__}")
        result = self.func(*args, **kwargs)
        print(f"DEBUG: Result = {result}")
        return result

    def _prod_wrapper(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except Exception as e:
            import logging
            logging.error(f"Error in {self.func.__name__}: {e}")
            raise

@EnvironmentAware
def process_data(data):
    return data.upper()

# Usage
import os
os.environ['ENV'] = 'development'
print(process_data("test"))
```

Slide 9: Giám sát hiệu suất với Class Decorators

Trình trang trí lớp có thể triển khai giám sát hiệu suất phức tạp bằng cách theo dõi số liệu thực thi qua nhiều lệnh gọi, cung cấp thông tin chi tiết có giá trị về hành vi chức năng và mô hình sử dụng tài nguyên.

```python
class PerformanceMonitor:
    def __init__(self, func):
        self.func = func
        self.calls = 0
        self.total_time = 0
        self.min_time = float('inf')
        self.max_time = float('-inf')

    def __call__(self, *args, **kwargs):
        import time
        start = time.perf_counter()
        result = self.func(*args, **kwargs)
        execution_time = time.perf_counter() - start

        self.calls += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)

        print(f"\nPerformance Stats for {self.func.__name__}:")
        print(f"Avg Time: {self.total_time/self.calls:.4f}s")
        print(f"Min Time: {self.min_time:.4f}s")
        print(f"Max Time: {self.max_time:.4f}s")
        return result

@PerformanceMonitor
def complex_calculation(n):
    import time
    time.sleep(0.1)  # Simulate work
    return sum(i * i for i in range(n))

# Usage example
for i in range(5):
    result = complex_calculation(1000 * i)
```

Slide 10: An toàn chỉ trong trang trí lớp học

Việc triển khai các trình trang trí lớp an toàn theo luồng đòi hỏi phải xem xét cẩn thận trạng thái chia sẻ và các mẫu truy cập đồng thời để đảm bảo hoạt động đáng tin cậy trong môi trường đa luồng.

```python
import threading
from functools import wraps

class ThreadSafeDecorator:
    def __init__(self, func):
        self.func = func
        self.lock = threading.Lock()
        self.local = threading.local()
        self.results = {}

    def __call__(self, *args, **kwargs):
        with self.lock:
            thread_id = threading.get_ident()
            if thread_id not in self.results:
                self.results[thread_id] = []

            result = self.func(*args, **kwargs)
            self.results[thread_id].append(result)

            return result

    def get_thread_results(self):
        return self.results.get(threading.get_ident(), [])

@ThreadSafeDecorator
def process_value(x):
    import time, random
    time.sleep(random.random() * 0.1)
    return x * 2

# Usage with multiple threads
def worker():
    for i in range(3):
        print(f"Thread {threading.get_ident()}: {process_value(i)}")

threads = [threading.Thread(target=worker) for _ in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

Trang trình bày 11: Ứng dụng thực tế: Giới hạn tốc độ API

Việc triển khai này thể hiện ứng dụng thực tế của trình trang trí lớp để giới hạn tốc độ API, bao gồm theo dõi yêu cầu và điều chỉnh tự động cho nhiều điểm cuối.

```python
from time import time
from collections import defaultdict
import threading

class APIRateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.lock = threading.Lock()

    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with self.lock:
                now = time()
                self.cleanup_old_requests(now)

                if self.can_make_request(now):
                    self.requests[func.__name__].append(now)
                    return func(*args, **kwargs)
                else:
                    raise Exception("Rate limit exceeded")

        return wrapped

    def cleanup_old_requests(self, now):
        window = 60  # 1 minute window
        for endpoint in self.requests:
            self.requests[endpoint] = [
                req_time for req_time in self.requests[endpoint]
                if now - req_time <= window
            ]

    def can_make_request(self, now):
        return len(self.requests) < self.requests_per_minute

# Example API endpoints
@APIRateLimiter(requests_per_minute=2)
def get_user_data(user_id):
    return f"Data for user {user_id}"

# Usage demonstration
for i in range(4):
    try:
        print(get_user_data(i))
    except Exception as e:
        print(f"Request {i}: {str(e)}")
```

Slide 12: Ứng dụng thực tế: Nhóm kết nối cơ sở dữ liệu

Việc triển khai này giới thiệu một hệ thống tổng hợp kết nối cơ sở dữ liệu thực tế bằng cách sử dụng các trình trang trí lớp, quản lý vòng đời kết nối và đảm bảo sử dụng tài nguyên hiệu quả.

```python
import queue
import threading
import time

class DatabaseConnectionPool:
    def __init__(self, pool_size=5, timeout=30):
        self.pool_size = pool_size
        self.timeout = timeout
        self.connections = queue.Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        for _ in range(self.pool_size):
            self.connections.put(self._create_connection())

    def _create_connection(self):
        # Simulated database connection
        return {
            'id': threading.get_ident(),
            'created_at': time.time(),
            'in_use': False
        }

    def __call__(self, func):
        def wrapped(*args, **kwargs):
            conn = self.acquire_connection()
            try:
                return func(conn, *args, **kwargs)
            finally:
                self.release_connection(conn)
        return wrapped

    def acquire_connection(self):
        try:
            conn = self.connections.get(timeout=self.timeout)
            with self.lock:
                conn['in_use'] = True
            return conn
        except queue.Empty:
            raise TimeoutError("No available database connections")

    def release_connection(self, conn):
        with self.lock:
            conn['in_use'] = False
        self.connections.put(conn)

# Example usage
pool = DatabaseConnectionPool(pool_size=2)

@pool
def execute_query(conn, query):
    print(f"Executing query with connection {conn['id']}: {query}")
    time.sleep(0.1)  # Simulate query execution
    return f"Result for {query}"

# Demonstration
def worker(query):
    try:
        result = execute_query(query)
        print(f"Query result: {result}")
    except TimeoutError as e:
        print(f"Error: {e}")

# Create multiple threads to test connection pool
threads = [
    threading.Thread(target=worker, args=(f"SELECT * FROM table_{i}",))
    for i in range(4)
]

for t in threads:
    t.start()
for t in threads:
    t.join()
```

Slide 13: Xử lý lỗi nâng cao với Class Decorators

Việc triển khai này thể hiện các cơ chế phục hồi và xử lý lỗi phức tạp bằng cách sử dụng các trình trang trí lớp, bao gồm xử lý ngoại lệ tùy chỉnh, ghi nhật ký và logic thử lại tự động.

```python
import functools
import logging
import time
from typing import Type, Tuple, Optional

class ErrorHandler:
    def __init__(self,
                 exceptions: Tuple[Type[Exception], ...],
                 retries: int = 3,
                 delay: float = 1.0,
                 backoff: float = 2.0,
                 logger: Optional[logging.Logger] = None):
        self.exceptions = exceptions
        self.retries = retries
        self.delay = delay
        self.backoff = backoff
        self.logger = logger or logging.getLogger(__name__)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            last_exception = None
            delay = self.delay

            for attempt in range(self.retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        self.logger.info(
                            f"Succeeded after {attempt} retries")
                    return result

                except self.exceptions as e:
                    last_exception = e
                    if attempt < self.retries:
                        self.logger.warning(
                            f"Attempt {attempt + 1} failed: {str(e)}")
                        time.sleep(delay)
                        delay *= self.backoff
                    else:
                        self.logger.error(
                            f"All {self.retries} retries failed")

            raise last_exception

        return wrapped

# Example usage
logging.basicConfig(level=logging.INFO)

@ErrorHandler(
    exceptions=(ConnectionError, TimeoutError),
    retries=2,
    delay=0.1
)
def unstable_network_call(url: str) -> str:
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Network unstable")
    return f"Success: {url}"

# Demonstration
try:
    result = unstable_network_call("http://example.com")
    print(f"Final result: {result}")
except ConnectionError as e:
    print(f"Final error: {e}")
```

Trang trình bày 14: Tài nguyên bổ sung

* Trình trang trí Python nâng cao: Các trường hợp và mẫu sử dụng trong thế giới thực
    * Tìm kiếm: "trang web mẫu python trang trí dựa trên lớp: arxiv.org"
* Tối ưu hóa hiệu suất với Python Decorators
    * [https://docs.python.org/3/howto/descriptor.html](https://docs.python.org/3/howto/descriptor.html)
    * [https://realpython.com/primer-on-python-designators/](https://realpython.com/primer-on-python-designators/)
* Lập trình đồng thời với Python Decorators
    * Tìm kiếm: "trang web lập trình đồng thời python:python.org"
* Các mẫu đồng bộ hóa và an toàn luồng trong Python
    * [https://docs.python.org/3/library/threading.html](https://docs.python.org/3/library/threading.html)
    * Tìm kiếm: "trang web triển khai python trang trí an toàn luồng: github.com"
