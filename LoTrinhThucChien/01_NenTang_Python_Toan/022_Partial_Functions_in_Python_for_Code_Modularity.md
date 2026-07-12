## Hàm từng phần trong Python dành cho tính mô đun mã
Trang trình bày 1: Giới thiệu về Hàm từng phần trong Python

Các hàm một phần trong Python cho phép chúng ta tạo các hàm mới bằng cách sửa một tập hợp con các đối số của hàm hiện có. Kỹ thuật này tăng cường tính mô đun hóa mã và khả năng sử dụng lại.

```python
from functools import partial

def power(base, exponent):
    return base ** exponent

# Creating a partial function
square = partial(power, exponent=2)

print(square(4))  # Output: 16
print(square(5))  # Output: 25
```

Slide 2: Hàm functools.partial

Hàm `functools.partial` là chìa khóa để tạo các hàm một phần trong Python. Nó nhận một hàm và một số đối số, trả về một hàm mới với các đối số được đặt trước đó.

```python
from functools import partial

def greet(greeting, name):
    return f"{greeting}, {name}!"

# Creating partial functions
say_hello = partial(greet, "Hello")
say_hi = partial(greet, "Hi")

print(say_hello("Alice"))  # Output: Hello, Alice!
print(say_hi("Bob"))       # Output: Hi, Bob!
```

Trang trình bày 3: Hàm từng phần và đối số mặc định

Các hàm một phần khác với các đối số mặc định. Trong khi các đối số mặc định được đặt ở định nghĩa hàm, các hàm một phần sẽ tạo các đối tượng hàm mới với các đối số được đặt trước.

```python
def multiply(a, b=2):
    return a * b

double = partial(multiply, b=2)

print(multiply(3))    # Output: 6 (using default argument)
print(double(3))      # Output: 6 (using partial function)
print(multiply(3, 4)) # Output: 12 (overriding default argument)
print(double(3, 4))   # Output: 12 (overriding partial function)
```

Trang trình bày 4: Hàm từng phần với đối số vị trí

Các hàm một phần cũng có thể được tạo bằng các đối số vị trí. Hàm mới sẽ có ít đối số bắt buộc hơn hàm ban đầu.

```python
from functools import partial

def power(base, exponent):
    return base ** exponent

# Creating a partial function with a positional argument
cube = partial(power, 3)

print(cube(2))  # Output: 8 (3^2)
print(cube(3))  # Output: 27 (3^3)
```

Trang trình bày 5: Các hàm một phần trong Trình xử lý sự kiện

Các hàm một phần đặc biệt hữu ích trong lập trình hướng sự kiện, cho phép chúng ta chuyển các đối số bổ sung cho các hàm gọi lại.

```python
import tkinter as tk
from functools import partial

def on_button_click(message, event):
    print(f"Button clicked! Message: {message}")

root = tk.Tk()
button = tk.Button(root, text="Click me!")
button.bind("<Button-1>", partial(on_button_click, "Hello, World!"))
button.pack()
root.mainloop()
```

Slide 6: Chức năng một phần cho cấu hình

Các hàm một phần có thể được sử dụng để tạo các phiên bản hàm được cấu hình sẵn, nâng cao tính mô đun mã.

```python
from functools import partial

def connect_to_database(host, port, user, password):
    # Simulating database connection
    return f"Connected to {host}:{port} as {user}"

# Pre-configured connection function
connect_to_prod = partial(connect_to_database,
                          host="prod.example.com",
                          port=5432,
                          user="admin")

print(connect_to_prod(password="secret"))
# Output: Connected to prod.example.com:5432 as admin
```

Slide 7: Hàm từng phần trong lập trình hàm

Các hàm một phần đóng một vai trò quan trọng trong các mô hình lập trình hàm, cho phép kết hợp hàm và cà ri.

```python
from functools import partial

def compose(f, g):
    return lambda x: f(g(x))

def add(a, b):
    return a + b

increment = partial(add, 1)
double = lambda x: x * 2

increment_and_double = compose(double, increment)

print(increment_and_double(3))  # Output: 8 ((3 + 1) * 2)
```

Slide 8: Chức năng một phần để liên kết tham số

Các hàm một phần có thể liên kết các tham số để tạo ra các phiên bản cụ thể hơn của các hàm chung.

```python
from functools import partial

def filter_by_attribute(items, attr, value):
    return [item for item in items if getattr(item, attr) == value]

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

people = [Person("Alice", 30), Person("Bob", 25), Person("Charlie", 30)]

filter_by_age = partial(filter_by_attribute, attr="age")
thirty_year_olds = filter_by_age(people, value=30)

for person in thirty_year_olds:
    print(person.name)  # Output: Alice, Charlie
```

Slide 9: Chức năng từng phần trong trang trí

Các hàm một phần có thể được sử dụng để tạo ra các trình trang trí linh hoạt chấp nhận các đối số.

```python
from functools import partial, wraps

def retry(max_attempts, exceptions):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions:
                    if attempt == max_attempts - 1:
                        raise
        return wrapper
    return decorator

# Creating a partial function for a specific retry decorator
retry_network_errors = partial(retry, exceptions=(ConnectionError, TimeoutError))

@retry_network_errors(max_attempts=3)
def fetch_data():
    # Simulating a network operation
    import random
    if random.random() < 0.5:
        raise ConnectionError("Network error occurred")
    return "Data fetched successfully"

print(fetch_data())  # May print: Data fetched successfully
```

Trang trình bày 10: Một phần chức năng ghi nhớ

Các hàm một phần có thể được sử dụng để triển khai tính năng ghi nhớ, một kỹ thuật để lưu vào bộ đệm các lệnh gọi hàm đắt tiền.

```python
from functools import partial, lru_cache

def memoize(func):
    return lru_cache(maxsize=None)(func)

@memoize
def expensive_computation(a, b):
    print(f"Computing {a} + {b}")
    return a + b

# Create partial functions for specific computations
compute_with_5 = partial(expensive_computation, 5)
compute_with_10 = partial(expensive_computation, 10)

print(compute_with_5(3))  # Output: Computing 5 + 3 \n 8
print(compute_with_5(3))  # Output: 8 (cached result)
print(compute_with_10(7))  # Output: Computing 10 + 7 \n 17
```

Slide 11: Chức năng từng phần trong kiểm thử

Các chức năng từng phần có thể đơn giản hóa việc thiết lập thử nghiệm bằng cách tạo các chức năng thử nghiệm được cấu hình sẵn.

```python
from functools import partial
import unittest

def validate_user(name, age, email):
    if len(name) < 2:
        raise ValueError("Name too short")
    if age < 18:
        raise ValueError("User too young")
    if "@" not in email:
        raise ValueError("Invalid email")
    return True

class TestUserValidation(unittest.TestCase):
    def setUp(self):
        self.validate = partial(validate_user, name="John", age=25)

    def test_valid_user(self):
        self.assertTrue(self.validate(email="john@example.com"))

    def test_invalid_email(self):
        with self.assertRaises(ValueError):
            self.validate(email="invalid-email")

if __name__ == "__main__":
    unittest.main()
```

Slide 12: Ví dụ thực tế: Xử lý ảnh

Các hàm từng phần có thể được sử dụng trong xử lý ảnh để tạo các hàm lọc có thể tái sử dụng.

```python
from functools import partial
from PIL import Image, ImageEnhance

def adjust_image(image, brightness=1.0, contrast=1.0, color=1.0):
    img = Image.open(image)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(color)
    return img

# Create partial functions for specific adjustments
increase_brightness = partial(adjust_image, brightness=1.5)
increase_contrast = partial(adjust_image, contrast=1.5)
grayscale = partial(adjust_image, color=0)

# Apply filters
original_image = "path/to/image.jpg"
bright_image = increase_brightness(original_image)
high_contrast_image = increase_contrast(original_image)
gray_image = grayscale(original_image)

bright_image.save("bright_image.jpg")
high_contrast_image.save("high_contrast_image.jpg")
gray_image.save("gray_image.jpg")
```

Trang trình bày 13: Ví dụ thực tế: Sắp xếp tùy chỉnh

Các hàm một phần có thể được sử dụng để tạo các hàm sắp xếp tùy chỉnh cho các cấu trúc dữ liệu phức tạp.

```python
from functools import partial

class Product:
    def __init__(self, name, price, rating):
        self.name = name
        self.price = price
        self.rating = rating

    def __repr__(self):
        return f"Product({self.name}, ${self.price}, {self.rating}★)"

def sort_products(products, key_func, reverse=False):
    return sorted(products, key=key_func, reverse=reverse)

# Create partial functions for different sorting criteria
sort_by_price = partial(sort_products, key_func=lambda p: p.price)
sort_by_rating = partial(sort_products, key_func=lambda p: p.rating, reverse=True)

products = [
    Product("Laptop", 1200, 4.5),
    Product("Phone", 800, 4.2),
    Product("Tablet", 500, 4.0),
    Product("Smartwatch", 300, 4.3)
]

print("Sorted by price (low to high):")
print(sort_by_price(products))

print("\nSorted by rating (high to low):")
print(sort_by_rating(products))
```

Trang trình bày 14: Tài nguyên bổ sung

Để khám phá thêm về các hàm từng phần và lập trình hàm trong Python:

1. "Lập trình hàm trong Python" của David Mertz (O'Reilly)
2. "Sách dạy nấu ăn Python" của David Beazley và Brian K. Jones (O'Reilly)
3. Tài liệu Python chính thức về functools: [https://docs.python.org/3/library/functools.html](https://docs.python.org/3/library/functools.html)
4. "Các hàm và thao tác bậc cao hơn trên các đối tượng có thể gọi được" (PEP 309): [https://www.python.org/dev/peps/pep-0309/](https://www.python.org/dev/peps/pep-0309/)
