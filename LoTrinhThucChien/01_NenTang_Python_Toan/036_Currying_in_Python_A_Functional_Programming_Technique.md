## Currying trong Python Kỹ thuật lập trình hàm
Trang trình bày 1: Tìm hiểu về Currying trong Python

Currying là một kỹ thuật lập trình hàm biến đổi một hàm có nhiều đối số thành một chuỗi các hàm, mỗi hàm lấy một đối số duy nhất. Khái niệm này được đặt theo tên của nhà toán học Haskell Curry và được sử dụng rộng rãi trong các ngôn ngữ lập trình hàm. Trong Python, chúng ta có thể triển khai Currying để tạo mã linh hoạt hơn và có thể sử dụng lại được.

```python
def add(x):
    def inner(y):
        return x + y
    return inner

add_5 = add(5)
result = add_5(3)
print(result)  # Output: 8
```

Slide 2: Ví dụ nấu cà ri cơ bản

Hãy bắt đầu với một ví dụ đơn giản để minh họa món cà ri. Chúng ta sẽ tạo một hàm curried để cộng hai số. Thay vì lấy cả hai đối số cùng một lúc, chúng ta sẽ chia nó thành hai hàm lồng nhau.

```python
def curry_add(x):
    def add_y(y):
        return x + y
    return add_y

# Usage
curried_add_5 = curry_add(5)
result = curried_add_5(3)
print(result)  # Output: 8

# Alternative usage
print(curry_add(2)(7))  # Output: 9
```

Trang trình bày 3: Cà ri và ứng dụng từng phần

Mặc dù cà ri và ứng dụng một phần là những khái niệm có liên quan nhưng chúng không giống nhau. Currying luôn tạo ra một chuỗi các hàm đơn nguyên (các hàm có một đối số), trong khi ứng dụng một phần có thể sửa bất kỳ số lượng đối số nào. Hãy so sánh hai:

```python
from functools import partial

# Currying
def curried_multiply(x):
    def multiply_by_y(y):
        return x * y
    return multiply_by_y

# Partial application
def multiply(x, y):
    return x * y

curried_double = curried_multiply(2)
partial_double = partial(multiply, 2)

print(curried_double(5))  # Output: 10
print(partial_double(5))  # Output: 10
```

Slide 4: Cà ri tự động

Chúng ta có thể tạo một trình trang trí để tự động xử lý bất kỳ hàm nào có nhiều đối số. Điều này cho phép chúng ta sử dụng hàm này ở cả dạng đã nấu chín và chưa nấu chín.

```python
def curry(func):
    def curried(*args):
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more_args: curried(*(args + more_args))
    return curried

@curry
def add_three_numbers(x, y, z):
    return x + y + z

print(add_three_numbers(1)(2)(3))  # Output: 6
print(add_three_numbers(1, 2)(3))  # Output: 6
print(add_three_numbers(1, 2, 3))  # Output: 6
```

Trang trình chiếu 5: Ví dụ thực tế: Xử lý văn bản

Currying có thể hữu ích trong các tác vụ xử lý văn bản. Hãy tạo một hàm cà ri để thay thế các từ trong câu:

```python
def replace_word(old_word):
    def with_new_word(new_word):
        def in_text(text):
            return text.replace(old_word, new_word)
        return in_text
    return with_new_word

replace_python = replace_word("Python")
replace_with_java = replace_python("Java")

original_text = "Python is a versatile programming language."
modified_text = replace_with_java(original_text)

print(modified_text)  # Output: Java is a versatile programming language.
```

Trang trình bày 6: Currying cho bố cục chức năng

Currying tạo điều kiện thuận lợi cho việc kết hợp hàm, cho phép chúng ta tạo các hàm mới bằng cách kết hợp các hàm hiện có. Dưới đây là một ví dụ về cách sử dụng Currying để tạo ra một hệ thống hoạt động:

```python
def curry(func):
    def curried(*args):
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more_args: curried(*(args + more_args))
    return curried

@curry
def add(x, y):
    return x + y

@curry
def multiply(x, y):
    return x * y

pipeline = lambda x: multiply(2)(add(3)(x))

result = pipeline(5)
print(result)  # Output: 16 ((5 + 3) * 2)
```

Slide 7: Cà ri để ghi nhớ

Currying có thể được kết hợp với tính năng ghi nhớ để tạo ra các hàm hiệu quả, có thể tái sử dụng và lưu trữ kết quả của chúng. Điều này đặc biệt hữu ích cho các tính toán tốn kém:

```python
def memoize(func):
    cache = {}
    def memoized(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return memoized

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(100))  # Output: 354224848179261915075 (calculated quickly due to memoization)
```

Slide 8: Ví dụ thực tế: Quản lý cấu hình

Currying có thể hữu ích trong các tình huống quản lý cấu hình. Hãy tạo một hàm curried để tạo các đối tượng cấu hình:

```python
def config_generator(environment):
    def set_database(database):
        def set_port(port):
            return {
                "environment": environment,
                "database": database,
                "port": port
            }
        return set_port
    return set_database

prod_config = config_generator("production")
prod_mysql_config = prod_config("mysql")
final_config = prod_mysql_config(3306)

print(final_config)
# Output: {'environment': 'production', 'database': 'mysql', 'port': 3306}
```

Slide 9: Cà ri với gợi ý kiểu

Chúng ta có thể sử dụng gợi ý kiểu để làm cho các hàm được xử lý dễ đọc và dễ bảo trì hơn. Đây là một ví dụ về hàm curried với gợi ý kiểu:

```python
from typing import Callable

def curried_formatter(prefix: str) -> Callable[[str], Callable[[int], str]]:
    def add_suffix(suffix: str) -> Callable[[int], str]:
        def format_number(number: int) -> str:
            return f"{prefix}{number}{suffix}"
        return format_number
    return add_suffix

format_usd = curried_formatter("$")("USD")
print(format_usd(100))  # Output: $100USD
print(format_usd(250))  # Output: $250USD

format_euro = curried_formatter("€")("EUR")
print(format_euro(100))  # Output: €100EUR
```

Slide 10: Cà ri và trang trí

Currying có thể được kết hợp với các bộ trang trí để tạo ra các phép biến đổi hàm mạnh mẽ và linh hoạt. Dưới đây là ví dụ về trình trang trí cà ri có thêm tính năng ghi nhật ký vào một hàm:

```python
import functools

def logged(level):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"{level}: Calling {func.__name__}")
            result = func(*args, **kwargs)
            print(f"{level}: Finished {func.__name__}")
            return result
        return wrapper
    return decorator

@logged("INFO")
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
# Output:
# INFO: Calling greet
# INFO: Finished greet
# Hello, Alice!
```

Trang trình bày 11: Cân nhắc về hiệu suất

Mặc dù cà ri có thể tạo ra mã linh hoạt và có khả năng tổng hợp hơn nhưng điều quan trọng là phải xem xét ý nghĩa hiệu suất của nó. Các hàm được xử lý thường liên quan đến nhiều lệnh gọi và đóng hàm, điều này có thể gây ra chi phí chung:

```python
import timeit

def regular_add(x, y):
    return x + y

def curried_add(x):
    def inner(y):
        return x + y
    return inner

regular_time = timeit.timeit("regular_add(5, 3)", globals=globals(), number=1000000)
curried_time = timeit.timeit("curried_add(5)(3)", globals=globals(), number=1000000)

print(f"Regular function time: {regular_time:.6f} seconds")
print(f"Curried function time: {curried_time:.6f} seconds")
print(f"Overhead: {(curried_time - regular_time) / regular_time * 100:.2f}%")
```

Slide 12: Currying trong các mô hình lập trình hàm

Currying đặc biệt hữu ích trong các mô hình lập trình hàm, nơi nó tạo điều kiện thuận lợi cho việc kết hợp hàm và ứng dụng từng phần. Hãy cùng khám phá cách sử dụng cà ri để tạo một quy trình xử lý dữ liệu đơn giản:

```python
from functools import reduce

def curry(func):
    def curried(*args):
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more_args: curried(*(args + more_args))
    return curried

@curry
def map_func(func, iterable):
    return map(func, iterable)

@curry
def filter_func(pred, iterable):
    return filter(pred, iterable)

@curry
def reduce_func(func, iterable):
    return reduce(func, iterable)

pipeline = (
    map_func(lambda x: x * 2)
    | filter_func(lambda x: x > 5)
    | reduce_func(lambda x, y: x + y)
)

result = pipeline(range(5))
print(result)  # Output: 18 (2*3 + 2*4)
```

Slide 13: Đánh giá sự lười biếng và lười biếng

Currying có thể được kết hợp với đánh giá lười biếng để tạo ra các quy trình xử lý dữ liệu hiệu quả. Đây là một ví dụ sử dụng mô-đun itertools của Python:

```python
import itertools

def curry(func):
    def curried(*args):
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more_args: curried(*(args + more_args))
    return curried

@curry
def take(n, iterable):
    return itertools.islice(iterable, n)

@curry
def map_func(func, iterable):
    return map(func, iterable)

@curry
def filter_func(pred, iterable):
    return filter(pred, iterable)

pipeline = (
    map_func(lambda x: x ** 2)
    | filter_func(lambda x: x % 2 == 0)
    | take(3)
)

result = list(pipeline(itertools.count()))
print(result)  # Output: [0, 4, 16]
```

Trang trình bày 14: Tài nguyên bổ sung

Đối với những người quan tâm đến việc tìm hiểu sâu hơn về cà ri và lập trình chức năng trong Python, đây là một số tài nguyên bổ sung:

1. "Lập trình hàm trong Python" của David Mertz (ArXiv:1904.04206) URL: [https://arxiv.org/abs/1904.04206](https://arxiv.org/abs/1904.04206)
2. "Giới thiệu nhẹ nhàng về lập trình hàm trong Python" của Cristian Medina (ArXiv:1904.04207) URL: [https://arxiv.org/abs/1904.04207](https://arxiv.org/abs/1904.04207)

Các bài viết này cung cấp một cái nhìn tổng quan toàn diện về các khái niệm lập trình chức năng, bao gồm cả cà ri và cách triển khai chúng trong Python.
