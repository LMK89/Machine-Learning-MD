## Đơn giản hóa việc quản lý tài nguyên với Trình quản lý bối cảnh Python
Slide 1: Giới thiệu về Trình quản lý bối cảnh

Trình quản lý bối cảnh trong Python là công cụ mạnh mẽ giúp quản lý tài nguyên một cách hiệu quả và tự động. Chúng đảm bảo thiết lập và dọn dẹp tài nguyên đúng cách, giảm nguy cơ lỗi và rò rỉ tài nguyên. Hãy cùng khám phá cách hoạt động của trình quản lý bối cảnh và tại sao chúng lại cần thiết để viết mã rõ ràng, dễ bảo trì.

```python
# Basic structure of a context manager
with open('example.txt', 'w') as file:
    file.write('Hello, Context Managers!')

# The file is automatically closed after the 'with' block
```

Trang trình bày 2: Câu lệnh 'với'

Câu lệnh 'with' là nền tảng của quản lý ngữ cảnh trong Python. Nó cung cấp một cách rõ ràng và dễ đọc để làm việc với các tài nguyên cần được quản lý hợp lý, chẳng hạn như tệp, kết nối mạng hoặc con trỏ cơ sở dữ liệu.

```python
# Without context manager
file = open('example.txt', 'r')
content = file.read()
file.close()

# With context manager
with open('example.txt', 'r') as file:
    content = file.read()
# File is automatically closed
```

Trang trình bày 3: Trình quản lý bối cảnh tích hợp

Python cung cấp một số trình quản lý bối cảnh tích hợp sẵn cho các hoạt động phổ biến. Chúng bao gồm xử lý tệp, khóa luồng và quản lý thư mục tạm thời. Hãy xem một ví dụ sử dụng trình quản lý bối cảnh 'threading.Lock()'.

```python
import threading

lock = threading.Lock()

def increment_counter(counter):
    with lock:
        counter.value += 1

counter = threading.Value('i', 0)
threads = [threading.Thread(target=increment_counter, args=(counter,)) for _ in range(10)]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

print(f"Final counter value: {counter.value}")
```

Trang trình bày 4: Kết quả cho: Trình quản lý bối cảnh tích hợp

```
Final counter value: 10
```

Slide 5: Creating Custom Context Managers

While built-in context managers are useful, you can also create custom context managers to suit your specific needs. There are two ways to create custom context managers: using a class or using the 'contextlib.contextmanager' decorator.

```python
class CustomContextManager:
    def __enter__(self):
        print("Entering the context")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting the context")

with CustomContextManager() as cm:
    print("Inside the context")
```

Trang trình bày 6: Kết quả cho: Tạo Trình quản lý bối cảnh tùy chỉnh

```
Entering the context
Inside the context
Exiting the context
```

Slide 7: Using contextlib.contextmanager

The 'contextlib.contextmanager' decorator provides a more concise way to create context managers using generator functions. This approach can be more readable for simple context managers.

```python
from contextlib import contextmanager

@contextmanager
def custom_context():
    print("Entering the context")
    yield
    print("Exiting the context")

with custom_context():
    print("Inside the context")
```

Slide 8: Kết quả cho: Sử dụng contextlib.contextmanager

```
Entering the context
Inside the context
Exiting the context
```

Slide 9: Real-Life Example: Database Connection Management

Context managers are particularly useful for managing database connections. They ensure that connections are properly closed, even if an error occurs during execution.

```python
import sqlite3
from contextlib import contextmanager

@contextmanager
def db_connection(db_name):
    conn = sqlite3.connect(db_name)
    try:
        yield conn
    finally:
        conn.close()

with db_connection('example.db') as conn:
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)")
    cursor.execute("INSERT INTO users (name) VALUES (?)", ("Alice",))
    conn.commit()

# Connection is automatically closed after the 'with' block
```

Trang trình chiếu 10: Ví dụ thực tế: Quản lý tệp tạm thời

Trình quản lý bối cảnh có thể được sử dụng để quản lý các tệp tạm thời, đảm bảo chúng được tạo và xóa đúng cách khi không còn cần thiết.

```python
import tempfile
import os

with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
    temp_file.write("This is a temporary file.")
    temp_file_path = temp_file.name

print(f"Temporary file created at: {temp_file_path}")

# File content can be read outside the context
with open(temp_file_path, 'r') as file:
    content = file.read()
    print(f"File content: {content}")

# Clean up: remove the temporary file
os.unlink(temp_file_path)
print("Temporary file removed.")
```

Trang trình bày 11: Xử lý lỗi trong Trình quản lý bối cảnh

Người quản lý bối cảnh có thể xử lý khéo léo các trường hợp ngoại lệ xảy ra trong phạm vi của họ. Tính năng này đặc biệt hữu ích để đảm bảo dọn dẹp tài nguyên thích hợp trong trường hợp có lỗi.

```python
class DatabaseConnection:
    def __enter__(self):
        print("Connecting to the database")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type:
            print(f"An error occurred: {exc_value}")
        print("Closing the database connection")
        return True  # Suppress the exception

with DatabaseConnection() as db:
    print("Connected to the database")
    raise ValueError("Simulated error")

print("Execution continues after the context manager")
```

Trang trình bày 12: Kết quả cho: Xử lý lỗi trong Trình quản lý bối cảnh

```
Connecting to the database
Connected to the database
An error occurred: Simulated error
Closing the database connection
Execution continues after the context manager
```

Slide 13: Nested Context Managers

Context managers can be nested to manage multiple resources simultaneously. This is particularly useful when working with complex systems that require multiple setup and teardown steps.

```python
from contextlib import contextmanager

@contextmanager
def outer_context():
    print("Entering outer context")
    yield "outer"
    print("Exiting outer context")

@contextmanager
def inner_context():
    print("Entering inner context")
    yield "inner"
    print("Exiting inner context")

with outer_context() as outer:
    print(f"In {outer} context")
    with inner_context() as inner:
        print(f"In {inner} context")
        print("Performing nested operations")
```

Trang trình bày 14: Kết quả cho: Trình quản lý bối cảnh lồng nhau

```
Entering outer context
In outer context
Entering inner context
In inner context
Performing nested operations
Exiting inner context
Exiting outer context
```

Slide 15: Additional Resources

For more information on context managers and advanced Python programming techniques, consider exploring the following resources:

1.  Python's official documentation on context managers: [https://docs.python.org/3/reference/datamodel.html#context-managers](https://docs.python.org/3/reference/datamodel.html#context-managers)
2.  PEP 343 - The "with" Statement: [https://www.python.org/dev/peps/pep-0343/](https://www.python.org/dev/peps/pep-0343/)
3.  Real Python's comprehensive guide on context managers: [https://realpython.com/python-with-statement/](https://realpython.com/python-with-statement/)
4.  "Fluent Python" by Luciano Ramalho, which covers context managers in depth.
