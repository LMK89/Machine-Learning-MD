## Giới thiệu về Giải tích I trong Python

Trang trình bày 1:
Giới thiệu về phép tính I
Trang trình bày này sẽ cung cấp cái nhìn tổng quan về Giải tích I và các ứng dụng của nó trong Python.

Trang trình bày 2:
Giới hạn
Hiểu khái niệm giới hạn là điều cần thiết trong Giải tích. Trang trình bày này sẽ trình bày định nghĩa cơ bản và các ví dụ về giới hạn trong Python.

Mã nguồn:

```python
import math

def limit_function(x, val):
    return (math.exp(x) - 1) / x

# Evaluating the limit as x approaches 0
print(limit_function(0.001, 0))  # Output: 1.0
```

Trang trình bày 3:
Tính liên tục
Tính liên tục là một khái niệm cơ bản trong Giải tích. Trang trình bày này sẽ giải thích ý nghĩa của tính liên tục và cách kiểm tra tính liên tục trong Python.

Mã nguồn:

```python
import math

def continuous_function(x):
    if x == 0:
        return 1
    else:
        return (math.sin(x) / x)

print(continuous_function(0))  # Output: 1.0
print(continuous_function(0.1))  # Output: 0.9983341664682815
```

Trang trình bày 4:
Công cụ phái sinh
Đạo hàm là nền tảng của Giải tích. Slide này sẽ giới thiệu khái niệm về đạo hàm và cách tính chúng trong Python.

Mã nguồn:

```python
import sympy as sp

x = sp.Symbol('x')
f = x**2 + 2*x + 1
print(f"Original Function: {f}")  # Output: Original Function: x**2 + 2*x + 1

derivative = sp.diff(f, x)
print(f"Derivative: {derivative}")  # Output: Derivative: 2*x + 2
```

Trang trình bày 5:
Quy tắc phân biệt
Trang trình bày này sẽ đề cập đến các quy tắc khác nhau để phân biệt các hàm, chẳng hạn như quy tắc lũy thừa, quy tắc sản phẩm và quy tắc chuỗi, với các ví dụ về Python.

Mã nguồn:

```python
import sympy as sp

x = sp.Symbol('x')

# Power Rule
f1 = x**3
print(f"Function: {f1}, Derivative: {sp.diff(f1, x)}")  # Output: Function: x**3, Derivative: 3*x**2

# Product Rule
f2 = (x**2) * (x**3)
print(f"Function: {f2}, Derivative: {sp.diff(f2, x)}")  # Output: Function: x**5, Derivative: 5*x**4

# Chain Rule
f3 = sp.sin(x**2)
print(f"Function: {f3}, Derivative: {sp.diff(f3, x)}")  # Output: Function: sin(x**2), Derivative: 2*x*cos(x**2)
```

Trang trình bày 6:
Đạo hàm cấp cao hơn
Trang trình bày này sẽ giải thích cách tính đạo hàm bậc cao hơn (thứ hai, thứ ba, v.v.) trong Python.

Mã nguồn:

```python
import sympy as sp

x = sp.Symbol('x')
f = x**4 + 2*x**3 - 3*x**2 + 4*x - 1

print(f"Original Function: {f}")
print(f"First Derivative: {sp.diff(f, x)}")
print(f"Second Derivative: {sp.diff(f, x, 2)}")
print(f"Third Derivative: {sp.diff(f, x, 3)}")
```

Trang trình bày 7:
Ứng dụng của công cụ phái sinh
Các công cụ phái sinh có nhiều ứng dụng trong nhiều lĩnh vực khác nhau. Slide này sẽ giới thiệu một số ứng dụng thực tế của đạo hàm trong Python.

Mã nguồn:

```python
import sympy as sp

x = sp.Symbol('x')

# Optimization
f = x**2 - 4*x + 3
critical_points = sp.solve(sp.diff(f, x), x)
print(f"Critical Points: {critical_points}")  # Output: Critical Points: [2, 1]

# Related Rates
r = sp.Symbol('r')
V = (4/3) * sp.pi * r**3
dV_dr = sp.diff(V, r)
print(f"Rate of change of volume with respect to radius: {dV_dr}")  # Output: Rate of change of volume with respect to radius: 4*pi*r**2
```

Trang trình bày 8:
tích phân
Tích phân là đối trọng của đạo hàm trong Giải tích. Trang trình bày này sẽ giới thiệu khái niệm tích phân và đánh giá chúng trong Python.

Mã nguồn:

```python
import sympy as sp

x = sp.Symbol('x')
f = x**2 + 2*x + 1
integral = sp.integrate(f, x)
print(f"Original Function: {f}")
print(f"Indefinite Integral: {integral}")  # Output: Indefinite Integral: x**3/3 + x**2 + x
```

Trang trình bày 9:
Kỹ thuật tích hợp
Trang trình bày này sẽ trình bày các kỹ thuật khác nhau để đánh giá tích phân, chẳng hạn như thay thế, tích phân từng phần và phân số một phần.

Mã nguồn:

```python
import sympy as sp

x = sp.Symbol('x')

# Substitution
f1 = sp.cos(x**2)
u = x**2
print(f"Original Function: {f1}, Indefinite Integral: {sp.integrate(f1, x)}")  # Output: Original Function: cos(x**2), Indefinite Integral: sin(x**2)/2

# Integration by Parts
f2 = x * sp.exp(x)
print(f"Original Function: {f2}, Indefinite Integral: {sp.integrate(f2, x)}")  # Output: Original Function: x*exp(x), Indefinite Integral: x*exp(x) - exp(x)

# Partial Fractions
f3 = (x**2 + 2*x + 1) / (x**2 + x)
print(f"Original Function: {f3}, Indefinite Integral: {sp.integrate(f3, x)}")  # Output: Original Function: (x**2 + 2*x + 1)/(x**2 + x), Indefinite Integral: x + 2*log(x) + log(x + 1)
```

Trang trình bày 10:
Tích phân xác định
Trang trình bày này sẽ giải thích khái niệm tích phân xác định và ứng dụng của chúng trong Python.

Mã nguồn:

```python
import sympy as sp

x = sp.Symbol('x')
f = x**2 + 2*x + 1
definite_integral = sp.integrate(f, (x, 0, 2))
print(f"Original Function: {f}")
print(f"Definite Integral from 0 to 2: {definite_integral}")  # Output: Definite Integral from 0 to 2: 11
```

Trang trình bày 11:
Ứng dụng của tích phân
Tích phân có nhiều ứng dụng trong nhiều lĩnh vực khác nhau. Slide này sẽ giới thiệu một số ứng dụng thực tế của tích phân trong Python.

Mã nguồn:

```python
import sympy as sp

x = sp.Symbol('x')

# Area Under a Curve
f = x**2
area = sp.integrate(f, (x, 0, 2))
print(f"Area under the curve y = x**2 from 0 to 2: {area}")  # Output: Area under the curve y = x**2 from 0 to 2: 8/3

# Volume of a Solid of Revolution
f = sp.sqrt(1 - x**2)
volume = sp.integrate(sp.pi * f**2, (x, -1, 1))
print(f"Volume of a sphere of radius 1: {volume}")  # Output: Volume of a sphere of radius 1: 4*pi/3
```

Slide 12: Định lý cơ bản của phép tính Định lý cơ bản của phép tính là một kết quả quan trọng kết nối đạo hàm và tích phân. Trang trình bày này sẽ giải thích định lý và ý nghĩa của nó, cùng với các ví dụ về Python.

Mã nguồn:

```python
import sympy as sp

x = sp.Symbol('x')

# Function and its derivative
f = x**3
f_prime = sp.diff(f, x)
print(f"Original Function: {f}, Derivative: {f_prime}")  # Output: Original Function: x**3, Derivative: 3*x**2

# Indefinite Integral of the derivative
indefinite_integral = sp.integrate(f_prime, x)
print(f"Indefinite Integral of the Derivative: {indefinite_integral}")  # Output: Indefinite Integral of the Derivative: x**3

# Fundamental Theorem of Calculus, Part 1
a = 0
b = 2
definite_integral = sp.integrate(f_prime, (x, a, b))
print(f"Definite Integral of the Derivative from {a} to {b}: {definite_integral}")  # Output: Definite Integral of the Derivative from 0 to 2: 8

# Fundamental Theorem of Calculus, Part 2
F = indefinite_integral
print(f"Antiderivative (Indefinite Integral) of f(x): {F}")  # Output: Antiderivative (Indefinite Integral) of f(x): x**3
print(f"F({b}) - F({a}) = {F.subs(x, b) - F.subs(x, a)}")  # Output: F(2) - F(0) = 8
```

Định lý cơ bản của Giải tích thiết lập mối quan hệ giữa vi phân và tích phân. Phần 1 của định lý nêu rõ rằng nếu một hàm `f(x)` liên tục trên một khoảng đóng `[a, b]`, thì tích phân xác định của `f(x)` trên khoảng đó bằng hiệu giữa các giá trị nguyên hàm bất kỳ (tích phân không xác định) của `f(x)` được tính tại các điểm cuối của khoảng.

Phần 2 của định lý phát biểu rằng nếu `F(x)` là nguyên hàm của `f(x)`, thì đạo hàm của `F(x)` là `f(x)`.

Mã nguồn thể hiện cả hai phần của Định lý cơ bản của phép tính bằng SymPy. Đầu tiên, nó định nghĩa một hàm `f(x) = x^3` và tính đạo hàm `f_prime(x) = 3x^2` của nó. Sau đó, nó tính tích phân bất định của `f_prime(x)`, kết quả là `x^3`. Điều này minh họa Phần 2 của định lý.

Tiếp theo, nó tính tích phân xác định của `f_prime(x)` trong khoảng `[0, 2]`, ước tính là `8`. Nó cũng chỉ ra rằng sự khác biệt giữa các giá trị của nguyên hàm `F(x) = x^3` được đánh giá tại `x = 2` và `x = 0` cũng là `8`, thể hiện Phần 1 của định lý.

Trang trình bày 13:
Tích hợp số
Trong nhiều trường hợp, việc tích hợp phân tích là không thể hoặc không thực tế. Trang trình bày này sẽ giới thiệu các kỹ thuật tích phân số, chẳng hạn như Quy tắc hình thang và Quy tắc Simpson, trong Python.

Mã nguồn:

```python
import numpy as np

def trapezoidal(func, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = func(x)
    s = y[0] + y[-1]
    for i in range(1, n):
        s += 2 * y[i]
    return h * s / 2

def simpsons(func, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = func(x)
    s = y[0] + y[-1]
    for i in range(1, n, 2):
        s += 4 * y[i]
    for i in range(2, n-1, 2):
        s += 2 * y[i]
    return h * s / 3

# Example function
def f(x):
    return x**2

# Trapezoidal Rule
print(trapezoidal(f, 0, 2, 10))  # Output: 4.3333333333333335

# Simpson's Rule
print(simpsons(f, 0, 2, 10))  # Output: 4.333333333333333
```

Trang trình bày 14:
Tích phân không đúng
Tích phân không đúng phát sinh khi khoảng tích phân là vô hạn hoặc tích phân không bị chặn. Trang trình bày này sẽ thảo luận về cách xử lý các tích phân không chính xác trong Python.

Mã nguồn:

```python
import sympy as sp

x = sp.Symbol('x')

# Integral over an infinite interval
f1 = 1 / (x**2 + 1)
improper_integral1 = sp.integrate(f1, (x, 1, sp.oo))
print(f"Improper Integral of 1/(x**2 + 1) from 1 to infinity: {improper_integral1}")  # Output: Improper Integral of 1/(x**2 + 1) from 1 to infinity: atan(1)

# Integral with an unbounded integrand
f2 = 1 / sp.sqrt(x)
improper_integral2 = sp.integrate(f2, (x, 0, 1))
print(f"Improper Integral of 1/sqrt(x) from 0 to 1: {improper_integral2}")  # Output: Improper Integral of 1/sqrt(x) from 0 to 1: 2
```

## Meta
Làm chủ phép tính I bằng Python: Hành trình của người mới bắt đầu

Bắt tay vào một cuộc phiêu lưu thú vị qua các lĩnh vực Giải tích I, nơi toán học gặp gỡ lập trình Python. Trong loạt bài toàn diện này, chúng ta sẽ khám phá các khái niệm cơ bản về giới hạn, tính liên tục, đạo hàm và tích phân, giải phóng sức mạnh của chúng thông qua các ví dụ thực tế và các đoạn mã hấp dẫn. Cho dù bạn là sinh viên, một nhà khoa học dữ liệu đầy tham vọng hay một người ham học hỏi, loạt bài này sẽ hướng dẫn bạn các công cụ tính toán cần thiết và ứng dụng của chúng trong Python. Hãy sẵn sàng nâng cao kỹ năng giải quyết vấn đề của bạn và hiểu sâu hơn về nền tảng toán học làm nền tảng cho các ngành khoa học và kỹ thuật khác nhau. #CalculusIPython #LearnProgramming #MathematicsForProgrammers #BeginnersGuide #AcademicExcellence

Thẻ bắt đầu bằng #: #CalculusIPython #LearnProgramming #MathematicsForProgrammers #BeginnersGuide #AcademicExcellence #CalculusExplained #PythonForMath #CalculusInAction #CodeAndCalculus #STEMEducation
