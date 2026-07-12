## Làm chủ các mô-đun Python! Hướng dẫn toàn diện

Trang trình bày 1: Mô-đun Python là gì?

Các mô-đun Python là các tệp mã có thể tái sử dụng, chứa các hàm, lớp và biến. Chúng giúp tổ chức và cấu trúc mã, làm cho mã dễ bảo trì và hiệu quả hơn. Các mô-đun có thể được tích hợp sẵn, như 'os' và 'math' hoặc do nhà phát triển tạo tùy chỉnh.

```python
import math

# Using a function from the math module
radius = 5
area = math.pi * radius ** 2
print(f"The area of a circle with radius {radius} is {area:.2f}")

# Output:
# The area of a circle with radius 5 is 78.54
```

Trang trình bày 2: Nhập mô-đun

Các mô-đun có thể được nhập bằng câu lệnh 'import'. Có nhiều cách khác nhau để nhập mô-đun, mỗi cách có trường hợp sử dụng riêng.

```python
import random

# Importing specific functions from a module
from datetime import datetime, timedelta

# Importing all functions from a module (use with caution)
from math import *

# Using imported functions
print(random.randint(1, 10))
print(datetime.now())
print(sqrt(16))

# Output:
# 7
# 2024-09-16 14:30:45.123456
# 4.0
```

Trang trình bày 3: Tạo mô-đun tùy chỉnh

Các mô-đun tùy chỉnh cho phép bạn sắp xếp mã của mình thành các tệp riêng biệt để có khả năng bảo trì và sử dụng lại tốt hơn.

```python
def greet(name):
    return f"Hello, {name}!"

def add(a, b):
    return a + b

# File: main.py
import my_module

print(my_module.greet("Alice"))
print(my_module.add(3, 4))

# Output:
# Hello, Alice!
# 7
```

Slide 4: Sử dụng bí danh cho mô-đun

Bí danh có thể làm cho mã của bạn ngắn gọn và dễ đọc hơn, đặc biệt đối với các mô-đun có tên dài.

```python
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a plot
plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```

Trang trình bày 5: Nhập từ các thư mục khác nhau

Đôi khi, bạn cần nhập mô-đun từ các thư mục khác nhau. Hiểu đường dẫn tìm kiếm mô-đun của Python là rất quan trọng.

```python
import os

# Add a directory to Python's module search path
custom_module_path = os.path.abspath('../custom_modules')
sys.path.append(custom_module_path)

# Now you can import modules from the added directory
import my_custom_module

print(my_custom_module.custom_function())

# Output depends on the content of my_custom_module
```

Trang trình bày 6: Xử lý lỗi nhập

Lỗi nhập là phổ biến khi làm việc với các mô-đun. Hiểu cách khắc phục sự cố chúng là điều cần thiết.

```python
    import non_existent_module
except ImportError as e:
    print(f"Error importing module: {e}")

    # Suggest installing the module if it's a third-party package
    print("Try installing the module using:")
    print("pip install non_existent_module")

# Output:
# Error importing module: No module named 'non_existent_module'
# Try installing the module using:
# pip install non_existent_module
```

Trang trình bày 7: Sử dụng **name** == "**main**"

Thành ngữ `__name__ == "__main__"` cho phép bạn viết mã chỉ chạy khi tập lệnh được thực thi trực tiếp chứ không phải khi nó được nhập dưới dạng mô-đun.

```python
def main_function():
    print("This is the main function of my_module")

if __name__ == "__main__":
    print("This module is being run directly")
    main_function()
else:
    print("This module is being imported")

# When run directly:
# This module is being run directly
# This is the main function of my_module

# When imported:
# This module is being imported
```

Slide 8: Khám phá các mô-đun tích hợp

Python đi kèm với một bộ mô-đun tích hợp phong phú. Hãy cùng khám phá một số điều hữu ích.

```python
import sys
import random
import json

# Get current working directory
print(os.getcwd())

# Get Python version
print(sys.version)

# Generate a random number
print(random.randint(1, 100))

# Work with JSON data
data = {"name": "Alice", "age": 30}
json_string = json.dumps(data)
print(json_string)

# Output varies based on your system and random generation
```

Trang trình bày 9: Làm việc với Người quản lý gói

Trình quản lý gói như pip giúp dễ dàng cài đặt và quản lý các mô-đun của bên thứ ba.

```python
# First, install it using pip:
# pip install requests

import requests

response = requests.get("https://api.github.com")
print(f"GitHub API Status Code: {response.status_code}")

if response.status_code == 200:
    print("Successfully connected to GitHub API")
else:
    print("Failed to connect to GitHub API")

# Output:
# GitHub API Status Code: 200
# Successfully connected to GitHub API
```

Trang trình bày 10: Ví dụ thực tế: Quét web

Hãy sử dụng mô-đun 'requests' và 'beautifulsoup4' cho một tác vụ quét web đơn giản.

```python
# pip install requests beautifulsoup4

import requests
from bs4 import BeautifulSoup

url = "https://news.ycombinator.com"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract and print the titles of the top stories
for story in soup.find_all('span', class_='titleline')[:5]:
    print(story.get_text())

# Output will be the titles of the top 5 stories on Hacker News
```

Trang trình chiếu 11: Ví dụ thực tế: Phân tích dữ liệu

Sử dụng pandas và matplotlib để phân tích và trực quan hóa dữ liệu cơ bản.

```python
# pip install pandas matplotlib

import pandas as pd
import matplotlib.pyplot as plt

# Create a sample dataset
data = {
    'Year': [2010, 2011, 2012, 2013, 2014],
    'Sales': [100, 150, 200, 180, 210]
}

df = pd.DataFrame(data)

# Calculate year-over-year growth
df['Growth'] = df['Sales'].pct_change() * 100

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['Sales'], marker='o')
plt.title('Sales Over Time')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

print(df)

# Output:
#    Year  Sales     Growth
# 0  2010    100        NaN
# 1  2011    150  50.000000
# 2  2012    200  33.333333
# 3  2013    180 -10.000000
# 4  2014    210  16.666667
```

Trang trình bày 12: Thực tiễn tốt nhất về mô-đun

Việc tuân theo các phương pháp hay nhất khi làm việc với mô-đun có thể giúp mã dễ bảo trì và hiệu quả hơn.

```python
from math import sqrt, pi

# Bad practice: Using wildcard imports
# from math import *

def calculate_circle_area(radius):
    return pi * radius ** 2

def calculate_hypotenuse(a, b):
    return sqrt(a**2 + b**2)

print(f"Area of circle with radius 5: {calculate_circle_area(5):.2f}")
print(f"Hypotenuse of triangle with sides 3 and 4: {calculate_hypotenuse(3, 4):.2f}")

# Output:
# Area of circle with radius 5: 78.54
# Hypotenuse of triangle with sides 3 and 4: 5.00
```

Trang trình bày 13: Khám phá các khái niệm mô-đun nâng cao

Hãy cùng đi sâu vào một số khái niệm mô-đun nâng cao như nhập lười biếng và quản lý bối cảnh.

```python
from importlib import import_module

def lazy_import(module_name):
    return lambda: import_module(module_name)

# The module is only imported when needed
numpy = lazy_import('numpy')

# Context manager example
class FileManager:
    def __init__(self, filename):
        self.filename = filename
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, 'w')
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

# Using the context manager
with FileManager('example.txt') as file:
    file.write('Hello, World!')

print("File operations completed.")

# Output:
# File operations completed.
```

Trang trình bày 14: Tài nguyên bổ sung

Để khám phá thêm về các mô-đun Python và các chủ đề liên quan, hãy xem xét các tài nguyên sau:

1. Tài liệu chính thức của Python về các mô-đun: [https://docs.python.org/3/tutorial/modules.html](https://docs.python.org/3/tutorial/modules.html)
2. "Hướng dẫn sử dụng Python cho người đi nhờ xe" của Kenneth Reitz và Tanya Schlusser
3. "Trăn thông thạo" của Luciano Ramalho
4. Chỉ mục gói Python (PyPI): [https://pypi.org/](https://pypi.org/)
5. Hướng dẫn thực tế về Python: [https://realpython.com/](https://realpython.com/)

Hãy nhớ luôn xác minh độ tin cậy và mức độ liên quan của các tài nguyên bổ sung trước khi sử dụng chúng trong hành trình học tập của bạn.
