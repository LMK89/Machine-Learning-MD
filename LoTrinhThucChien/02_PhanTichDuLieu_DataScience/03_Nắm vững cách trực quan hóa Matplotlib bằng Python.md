## Làm chủ trực quan hóa Matplotlib bằng Python
Slide 1: Giới thiệu về Matplotlib

Matplotlib là một thư viện vẽ đồ thị mạnh mẽ dành cho Python, được sử dụng rộng rãi để tạo trực quan hóa tĩnh, hoạt hình và tương tác. Nó cung cấp giao diện giống MATLAB và có thể tạo ra các số liệu chất lượng xuất bản ở nhiều định dạng khác nhau.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('A Simple Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
```

Trang trình bày 2: Sơ đồ đường cơ bản

Biểu đồ đường là cơ bản trong trực quan hóa dữ liệu. Chúng hiển thị xu hướng trong một khoảng thời gian liên tục và rất phù hợp để hiển thị dữ liệu chuỗi thời gian.

```python
import matplotlib.pyplot as plt

years = [2015, 2016, 2017, 2018, 2019, 2020]
temperatures = [15.2, 15.5, 15.8, 16.1, 16.3, 16.5]

plt.figure(figsize=(10, 6))
plt.plot(years, temperatures, marker='o')
plt.title('Average Annual Temperatures')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()
```

Slide 3: Tùy chỉnh kiểu vẽ

Matplotlib cung cấp nhiều kiểu khác nhau để tùy chỉnh giao diện của ô của bạn. Bạn có thể thay đổi màu sắc, kiểu đường kẻ, điểm đánh dấu, v.v.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

plt.figure(figsize=(12, 6))
plt.plot(x, np.sin(x), 'r--', label='sin(x)')
plt.plot(x, np.cos(x), 'b-.', label='cos(x)')
plt.plot(x, -np.sin(x), 'g:', label='-sin(x)')
plt.legend()
plt.title('Trigonometric Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
```

Trang trình bày 4: Đồ thị phân tán

Biểu đồ phân tán rất hữu ích trong việc hiển thị mối quan hệ giữa hai biến. Chúng có thể tiết lộ các mẫu, mối tương quan hoặc cụm trong dữ liệu.

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.random.rand(50)
y = 2 * x + np.random.rand(50)

plt.figure(figsize=(10, 6))
plt.scatter(x, y, c='purple', alpha=0.6, s=100)
plt.title('Scatter Plot Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()
```

Trang trình bày 5: Biểu đồ thanh

Biểu đồ thanh là công cụ tuyệt vời để so sánh số lượng giữa các danh mục khác nhau. Chúng có thể dọc hoặc ngang.

```python
import matplotlib.pyplot as plt

fruits = ['Apples', 'Oranges', 'Bananas', 'Pears', 'Grapes']
quantities = [30, 25, 40, 20, 35]

plt.figure(figsize=(10, 6))
plt.bar(fruits, quantities, color=['red', 'orange', 'yellow', 'green', 'purple'])
plt.title('Fruit Quantities')
plt.xlabel('Fruit')
plt.ylabel('Quantity')
plt.ylim(0, 50)
for i, v in enumerate(quantities):
    plt.text(i, v + 1, str(v), ha='center')
plt.show()
```

Slide 6: Biểu đồ

Biểu đồ hiển thị sự phân bố của một tập dữ liệu. Chúng hữu ích để hiểu phân bố tần số cơ bản của một tập hợp dữ liệu liên tục.

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
data = np.random.normal(170, 10, 250)  # Generate 250 heights with mean 170 and std 10

plt.figure(figsize=(10, 6))
plt.hist(data, bins=20, edgecolor='black')
plt.title('Distribution of Heights')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```

Trang trình bày 7: Các ô phụ

Ô phụ cho phép bạn tạo nhiều ô trong một hình duy nhất, rất hữu ích để so sánh các tập dữ liệu khác nhau hoặc trực quan hóa các khía cạnh khác nhau của cùng một dữ liệu.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Different Plot Types in Subplots')

axs[0, 0].plot(x, np.sin(x))
axs[0, 0].set_title('Sine Wave')

axs[0, 1].scatter(np.random.rand(50), np.random.rand(50))
axs[0, 1].set_title('Scatter Plot')

axs[1, 0].bar(['A', 'B', 'C', 'D'], [3, 7, 2, 5])
axs[1, 0].set_title('Bar Chart')

axs[1, 1].hist(np.random.normal(0, 1, 1000), bins=30)
axs[1, 1].set_title('Histogram')

plt.tight_layout()
plt.show()
```

Trang trình bày 8: Sơ đồ 3D

Matplotlib có thể tạo các biểu đồ 3D, rất hữu ích để hiển thị dữ liệu hoặc bề mặt ba chiều.

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

surf = ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_title('3D Surface Plot')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
fig.colorbar(surf)
plt.show()
```

Trang trình bày 9: Biểu đồ hình tròn

Biểu đồ hình tròn được sử dụng để thể hiện thành phần của một tổng thể, được chia thành các phần. Chúng có hiệu quả trong việc hiển thị dữ liệu phần trăm hoặc tỷ lệ.

```python
import matplotlib.pyplot as plt

activities = ['Work', 'Sleep', 'Leisure', 'Eat', 'Commute']
hours = [8, 7, 5, 2, 2]
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']

plt.figure(figsize=(10, 8))
plt.pie(hours, labels=activities, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Daily Activities')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()
```

Trang trình bày 10: Bản đồ nhiệt

Bản đồ nhiệt rất hữu ích để trực quan hóa dữ liệu ma trận, hiển thị các mẫu, mối tương quan hoặc cường độ tương đối.

```python
import matplotlib.pyplot as plt
import numpy as np

data = np.random.rand(10, 10)
plt.figure(figsize=(10, 8))
heatmap = plt.imshow(data, cmap='YlOrRd')
plt.colorbar(heatmap)

plt.title('Heatmap Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Add value annotations
for i in range(10):
    for j in range(10):
        plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='black')

plt.show()
```

Trang trình chiếu 11: Hoạt hình

Matplotlib có thể tạo các biểu đồ hoạt hình, rất phù hợp để trực quan hóa dữ liệu thay đổi theo thời gian hoặc các lần lặp lại.

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))

def animate(i):
    line.set_ydata(np.sin(x + i/10))
    return line,

ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
plt.title('Animated Sine Wave')
plt.show()
```

Trang trình bày 12: Tùy chỉnh đánh dấu và nhãn

Việc tinh chỉnh các dấu trục và nhãn có thể cải thiện đáng kể khả năng đọc và hình thức của đồ thị của bạn.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)

ax.set_title('Customized Sine Wave Plot')
ax.set_xlabel('Angle (radians)')
ax.set_ylabel('Sine value')

# Customize x-ticks
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(['0', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$'])

# Customize y-ticks
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticklabels(['-1', '-0.5', '0', '0.5', '1'])

ax.grid(True)
plt.show()
```

Trang trình bày 13: Ví dụ thực tế: Trực quan hóa dữ liệu thời tiết

Hãy trực quan hóa dữ liệu nhiệt độ hàng tháng của một thành phố, trình bày cách xử lý dữ liệu chuỗi thời gian và tạo các biểu đồ giàu thông tin.

```python
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Generate sample weather data
np.random.seed(42)
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(365)]
temps = np.random.normal(15, 10, 365) + 10 * np.sin(np.arange(365) * 2 * np.pi / 365)

# Calculate monthly averages
monthly_temps = [temps[i:i+30].mean() for i in range(0, 360, 30)]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Weather Data Visualization', fontsize=16)

# Daily temperature plot
ax1.plot(dates, temps)
ax1.set_title('Daily Temperatures')
ax1.set_xlabel('Date')
ax1.set_ylabel('Temperature (°C)')
ax1.grid(True)

# Monthly average bar plot
ax2.bar(months, monthly_temps)
ax2.set_title('Monthly Average Temperatures')
ax2.set_xlabel('Month')
ax2.set_ylabel('Average Temperature (°C)')
ax2.set_ylim(0, max(monthly_temps) + 5)

for i, temp in enumerate(monthly_temps):
    ax2.text(i, temp + 0.5, f'{temp:.1f}°C', ha='center')

plt.tight_layout()
plt.show()
```

Slide 14: Ví dụ thực tế: Tháp dân số

Tháp dân số là biểu đồ thể hiện sự phân bổ độ tuổi và giới tính của dân số. Hãy tạo một cái bằng Matplotlib.

```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with real data for actual use)
ages = np.arange(0, 101, 10)
male_pop = [5, 7, 8, 10, 9, 8, 6, 4, 2, 1, 0.5]
female_pop = [5.2, 7.3, 8.4, 10.2, 9.4, 8.2, 6.3, 4.4, 2.3, 1.2, 0.6]

fig, ax = plt.subplots(figsize=(10, 8))

ax.barh(ages, male_pop, height=8, align='center', color='skyblue', label='Male')
ax.barh(ages, [-pop for pop in female_pop], height=8, align='center', color='pink', label='Female')

ax.set_xlabel('Population (%)')
ax.set_ylabel('Age Group')
ax.set_title('Population Pyramid Example')

ax.legend()

ax.set_xticks(np.arange(-10, 11, 2))
ax.set_xticklabels([str(abs(x)) for x in ax.get_xticks()])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.text(0, 105, 'Male', ha='right', va='bottom')
ax.text(0, 105, 'Female', ha='left', va='bottom')

plt.tight_layout()
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Để khám phá thêm về Matplotlib và các khả năng của nó, hãy xem xét các tài nguyên sau:

1. Tài liệu chính thức của Matplotlib: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
2. "Trực quan hóa bằng Matplotlib" của Jake VanderPlas (ArXiv:1412.3590): [https://arxiv.org/abs/1412.3590](https://arxiv.org/abs/1412.3590)
3. "Trực quan hóa khoa học: Python + Matplotlib" của Nicolas P. Rougier (ArXiv:1401.4127): [https://arxiv.org/abs/1401.4127](https://arxiv.org/abs/1401.4127)

Những tài nguyên này cung cấp các hướng dẫn, ví dụ chuyên sâu và các kỹ thuật nâng cao để thành thạo Matplotlib.
