## Ưu điểm của việc đăng nhập qua Print() trong Python
Slide 1: Giới thiệu về Logging trong Python

Ghi nhật ký là một công cụ mạnh mẽ để theo dõi các sự kiện trong chương trình Python của bạn. Nó mang lại những lợi thế đáng kể so với việc sử dụng các câu lệnh print() để gỡ lỗi và giám sát.

```python
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Example usage
logging.info("Program started")
logging.warning("Low memory warning")
logging.error("File not found")

# Output:
# 2024-09-28 10:15:30,123 - INFO - Program started
# 2024-09-28 10:15:30,124 - WARNING - Low memory warning
# 2024-09-28 10:15:30,125 - ERROR - File not found
```

Trang trình bày 2: Mức độ nghiêm trọng có thể điều chỉnh

Việc ghi nhật ký cung cấp các mức độ nghiêm trọng khác nhau, cho phép bạn phân loại thư dựa trên tầm quan trọng của chúng. Tính năng này cho phép tổ chức và lọc thông điệp tường trình tốt hơn.

```python
import logging

logging.basicConfig(level=logging.DEBUG)

logging.debug("Detailed information for debugging")
logging.info("General information about program execution")
logging.warning("An indication of a potential problem")
logging.error("A more serious problem")
logging.critical("A critical error - program may be unable to continue")

# Output:
# DEBUG:root:Detailed information for debugging
# INFO:root:General information about program execution
# WARNING:root:An indication of a potential problem
# ERROR:root:A more serious problem
# CRITICAL:root:A critical error - program may be unable to continue
```

Slide 3: Cấu hình linh hoạt

Tính năng ghi nhật ký cung cấp các tùy chọn cấu hình mở rộng, cho phép bạn tùy chỉnh vị trí và cách thức ghi lại tin nhắn của bạn. Bạn có thể dễ dàng hướng đầu ra nhật ký đến các điểm đến khác nhau.

```python
import logging

# Log to a file
logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG)
logging.debug("This message will be written to app.log")

# Log to console and file simultaneously
console = logging.StreamHandler()
file_handler = logging.FileHandler('both.log')
logging.getLogger('').addHandler(console)
logging.getLogger('').addHandler(file_handler)
logging.warning("This will appear in both console and both.log file")

# Output to console and both.log file:
# WARNING:root:This will appear in both console and both.log file
```

Trang trình bày 4: Hiệu suất được cải thiện

Không giống như các câu lệnh print(), việc ghi nhật ký có thể bị vô hiệu hóa hoặc lọc một cách hiệu quả mà không cần sửa đổi mã, mang lại hiệu suất tốt hơn trong môi trường sản xuất.

```python
import logging
import time

def performance_test(log_func):
    start_time = time.time()
    for i in range(100000):
        log_func(f"Iteration {i}")
    return time.time() - start_time

# Test with print()
print_time = performance_test(print)

# Test with logging (INFO level)
logging.basicConfig(level=logging.INFO)
log_time = performance_test(logging.info)

# Test with logging (WARNING level, so INFO messages are ignored)
logging.basicConfig(level=logging.WARNING)
log_ignored_time = performance_test(logging.info)

print(f"Print time: {print_time:.4f} seconds")
print(f"Logging (INFO) time: {log_time:.4f} seconds")
print(f"Logging (ignored) time: {log_ignored_time:.4f} seconds")

# Output (approximate):
# Print time: 0.1500 seconds
# Logging (INFO) time: 0.2000 seconds
# Logging (ignored) time: 0.0100 seconds
```

Trang trình bày 5: Quản lý lỗi hiệu quả

Việc ghi nhật ký cung cấp thông tin chi tiết về các ngoại lệ, giúp chẩn đoán và khắc phục sự cố trong mã của bạn dễ dàng hơn.

```python
import logging

logging.basicConfig(level=logging.ERROR)

def divide(x, y):
    try:
        result = x / y
    except ZeroDivisionError:
        logging.exception("Division by zero attempted")
    else:
        return result

divide(10, 0)

# Output:
# ERROR:root:Division by zero attempted
# Traceback (most recent call last):
#   File "<stdin>", line 3, in divide
# ZeroDivisionError: division by zero
```

Trang trình bày 6: Triển khai đơn giản hóa

Với tính năng ghi nhật ký, bạn có thể dễ dàng điều chỉnh mức độ chi tiết của ứng dụng mà không cần sửa đổi mã, đơn giản hóa quá trình chuyển đổi từ môi trường phát triển sang môi trường sản xuất.

```python
import logging
import sys

# Development configuration
if '--dev' in sys.argv:
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Running in development mode")
else:
    # Production configuration
    logging.basicConfig(level=logging.ERROR)
    logging.info("Running in production mode")

# This will only show in development mode
logging.debug("Database connection established")

# This will show in both modes
logging.error("Critical error occurred")

# Output in development mode:
# INFO:root:Running in development mode
# DEBUG:root:Database connection established
# ERROR:root:Critical error occurred

# Output in production mode:
# ERROR:root:Critical error occurred
```

Slide 7: Tùy chỉnh định dạng nhật ký

Việc ghi nhật ký cho phép bạn tùy chỉnh định dạng của thông điệp tường trình, cung cấp nhiều ngữ cảnh hơn và giúp chúng dễ dàng phân tích và phân tích hơn.

```python
import logging

# Create a custom formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create and configure a handler
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Create a logger and add the handler
logger = logging.getLogger('MyApp')
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Use the logger
logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')

# Output:
# 2024-09-28 10:30:15,123 - MyApp - DEBUG - This is a debug message
# 2024-09-28 10:30:15,124 - MyApp - INFO - This is an info message
# 2024-09-28 10:30:15,125 - MyApp - WARNING - This is a warning message
```

Slide 8: Đăng nhập nhiều module

Tính năng ghi nhật ký có thể được sử dụng hiệu quả trên nhiều mô-đun trong ứng dụng của bạn, cung cấp một cách tập trung để quản lý nhật ký từ các phần khác nhau trong chương trình của bạn.

```python
# module_a.py
import logging

logger = logging.getLogger(__name__)

def function_a():
    logger.info("Function A called")

# module_b.py
import logging

logger = logging.getLogger(__name__)

def function_b():
    logger.warning("Function B called")

# main.py
import logging
import module_a
import module_b

logging.basicConfig(level=logging.INFO)

module_a.function_a()
module_b.function_b()

# Output:
# INFO:module_a:Function A called
# WARNING:module_b:Function B called
```

Slide 9: Trình xử lý tệp xoay

Đối với các ứng dụng chạy lâu, điều quan trọng là phải quản lý kích thước tệp nhật ký. RotatingFileHandler cho phép bạn tự động tạo các tệp nhật ký mới khi tệp hiện tại đạt đến kích thước nhất định.

```python
import logging
from logging.handlers import RotatingFileHandler

# Create a rotating file handler
handler = RotatingFileHandler('app.log', maxBytes=2000, backupCount=5)
logger = logging.getLogger('RotatingLogger')
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Generate some log messages
for i in range(10000):
    logger.info(f"This is log message {i}")

# This will create app.log, app.log.1, app.log.2, etc.
# when app.log reaches 2000 bytes
```

Trang trình chiếu 10: Ví dụ thực tế: Ghi nhật ký máy chủ web

Ghi nhật ký là rất quan trọng để theo dõi và gỡ lỗi các ứng dụng web. Đây là ví dụ về cách bạn có thể thiết lập ghi nhật ký cho một máy chủ web đơn giản.

```python
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler

# Set up logging
logging.basicConfig(filename='webserver.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        logging.info(f"GET request received for path: {self.path}")
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'Hello, World!')

    def log_message(self, format, *args):
        logging.info("%s - - [%s] %s" %
                     (self.client_address[0],
                      self.log_date_time_string(),
                      format%args))

httpd = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
logging.info("Server started on localhost:8000")
httpd.serve_forever()

# Output in webserver.log:
# 2024-09-28 11:00:00,123 - INFO - Server started on localhost:8000
# 2024-09-28 11:00:05,456 - INFO - GET request received for path: /
# 2024-09-28 11:00:05,457 - INFO - 127.0.0.1 - - [28/Sep/2024 11:00:05] "GET / HTTP/1.1" 200 -
```

Trang trình bày 11: Ví dụ thực tế: Quy trình xử lý dữ liệu

Ghi nhật ký là điều cần thiết trong quy trình xử lý dữ liệu để theo dõi tiến trình và phát hiện lỗi. Đây là ví dụ về cách sử dụng tính năng ghi nhật ký trong tập lệnh xử lý dữ liệu đơn giản.

```python
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data(data):
    logging.info(f"Processing {len(data)} items")
    processed = []
    for i, item in enumerate(data):
        try:
            result = item * random.randint(1, 10)
            processed.append(result)
            if i % 100 == 0:
                logging.debug(f"Processed {i} items")
        except Exception as e:
            logging.error(f"Error processing item {i}: {e}")
    logging.info(f"Finished processing. {len(processed)} items successful")
    return processed

# Generate some sample data
data = list(range(1000))

# Process the data
result = process_data(data)

# Output:
# 2024-09-28 11:30:00,123 - INFO - Processing 1000 items
# 2024-09-28 11:30:00,124 - DEBUG - Processed 0 items
# 2024-09-28 11:30:00,125 - DEBUG - Processed 100 items
# ...
# 2024-09-28 11:30:00,234 - DEBUG - Processed 900 items
# 2024-09-28 11:30:00,235 - INFO - Finished processing. 1000 items successful
```

Trang trình bày 12: Các phương pháp hay nhất về ghi nhật ký

Dưới đây là một số phương pháp hay nhất cần tuân theo khi triển khai đăng nhập vào ứng dụng Python của bạn:

1. Sử dụng cấp độ nhật ký phù hợp
2. Bao gồm thông tin theo ngữ cảnh
3. Sử dụng tính năng ghi nhật ký có cấu trúc cho dữ liệu phức tạp
4. Định cấu hình ghi nhật ký trong ứng dụng của bạn càng sớm càng tốt
5. Sử dụng ghi nhật ký ngoại lệ với log.Exception()
6. Tránh thông tin nhạy cảm trong nhật ký

```python
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_user(user_id, action):
    logger.info("Processing user", extra={
        'user_id': user_id,
        'action': action
    })

    # Simulating an error
    try:
        if action == 'delete':
            raise ValueError("Cannot delete user")
        # Process user...
    except Exception as e:
        logger.exception(f"Error processing user {user_id}")

# Usage
process_user(12345, 'update')
process_user(67890, 'delete')

# Output:
# 2024-09-28 12:00:00,123 - __main__ - INFO - Processing user
# 2024-09-28 12:00:00,124 - __main__ - ERROR - Error processing user 67890
# Traceback (most recent call last):
#   File "<stdin>", line 4, in process_user
# ValueError: Cannot delete user
```

Trang trình bày 13: Ghi nhật ký và In: So sánh

Hãy so sánh việc ghi nhật ký và in() để hiểu tại sao việc ghi nhật ký thường được ưa thích hơn để gỡ lỗi và giám sát các ứng dụng.

```python
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def using_print():
    print("Starting function")
    time.sleep(1)
    print("Function completed")

def using_logging():
    logging.info("Starting function")
    time.sleep(1)
    logging.info("Function completed")

print("Using print():")
using_print()

print("\nUsing logging:")
using_logging()

# Output:
# Using print():
# Starting function
# Function completed

# Using logging:
# 2024-09-28 12:30:00,123 - INFO - Starting function
# 2024-09-28 12:30:01,124 - INFO - Function completed
```

Trang trình bày 14: Tài nguyên bổ sung

Để biết thêm thông tin về ghi nhật ký Python, hãy xem xét khám phá các tài nguyên sau:

1. Tài liệu ghi nhật ký chính thức của Python: [https://docs.python.org/3/library/logging.html](https://docs.python.org/3/library/logging.html)
2. Ghi nhật ký Sách dạy nấu ăn: [https://docs.python.org/3/howto/logging-cookbook.html](https://docs.python.org/3/howto/logging-cookbook.html)
3. "Ghi nhật ký có cấu trúc bằng Python" của Yury Selivanov: [https://arxiv.org/abs/2110.07557](https://arxiv.org/abs/2110.07557)

Những tài nguyên này cung cấp những giải thích chuyên sâu và các kỹ thuật nâng cao để thành thạo việc ghi nhật ký bằng Python.
