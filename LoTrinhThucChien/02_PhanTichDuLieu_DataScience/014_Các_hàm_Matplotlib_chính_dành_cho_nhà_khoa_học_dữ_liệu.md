## Các hàm Matplotlib chính dành cho nhà khoa học dữ liệu
Slide 1: Giới thiệu về Matplotlib

Matplotlib là một thư viện trực tuyến hóa dữ liệu mạnh mẽ cho Python. Nó cung cấp một loạt chức năng để tạo ra nhiều loại biểu đồ và biểu đồ khác nhau. Bài trình bày này sẽ bao gồm các chức năng chính của Matplotlib mà mọi nhà khoa học dữ liệu nên tìm hiểu, cùng với các ví dụ thực tế và đoạn mã.

```python
import matplotlib.pyplot as plt
import numpy as np

# Create a simple line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('A Simple Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
```

Slide 2: Hàm cốt truyện ()

Hàmplot() được sử dụng để tạo đường biểu thức. Nó có thể hoạt động và có thể được sử dụng để phân tích các xu hướng theo thời gian hoặc liên kết giữa các liên kết biến. Trong ví dụ này, chúng tôi sẽ vẽ đồ thị sự phát triển của quần thể vi khu vực theo thời gian.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate data for bacterial growth
time = np.linspace(0, 24, 100)
population = 1000 * np.exp(0.2 * time)

plt.plot(time, population)
plt.title('Bacterial Growth Over Time')
plt.xlabel('Time (hours)')
plt.ylabel('Population')
plt.show()
```

Slide 3: Hàm phân tán()

Hàm phân tán() được sử dụng để tạo các biểu đồ phân tán, rất hữu ích trong việc hiển thị mối quan hệ giữa hai biến. Hãy sử dụng nó để khám phá mối tương quan giữa thời gian học và điểm thi.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate random data for study time and exam scores
study_time = np.random.randn(50) * 2 + 10
exam_scores = 5 * study_time + np.random.randn(50) * 10 + 60

plt.scatter(study_time, exam_scores)
plt.title('Study Time vs. Exam Scores')
plt.xlabel('Study Time (hours)')
plt.ylabel('Exam Score')
plt.show()
```

Slide 4: Hàm hist()

Hàm hist() tạo biểu đồ, rất hữu ích để trực quan hóa việc phân tích dữ liệu bổ sung. Vui lòng sử dụng nó để phân tích chiều cao phân bố trong dân số.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate random height data
heights = np.random.normal(170, 10, 1000)

plt.hist(heights, bins=30, edgecolor='black')
plt.title('Distribution of Heights in Population')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.show()
```

Slide 5: Hàm bar()

Hàm bar() tạo biểu đồ, lý tưởng để so sánh số lượng giữa các danh mục khác nhau. Vui lòng sử dụng nó để hiển thị mức độ phổ biến của các trình cài đặt ngôn ngữ khác nhau.

```python
import matplotlib.pyplot as plt

languages = ['Python', 'Java', 'JavaScript', 'C++', 'Ruby']
popularity = [68, 45, 63, 38, 22]

plt.bar(languages, popularity)
plt.title('Programming Language Popularity')
plt.xlabel('Programming Language')
plt.ylabel('Popularity Index')
plt.show()
```

Slide 6: Hàm pie()

Hàm pie() tạo biểu đồ hình tròn, rất hữu ích để hiển thị tỷ lệ tổng thể. Vui lòng sử dụng nó để hiển thị các giao thức phân tích bổ sung khác nhau trong một thành phố.

```python
import matplotlib.pyplot as plt

transport_modes = ['Car', 'Bus', 'Bicycle', 'Walking', 'Train']
percentages = [45, 20, 15, 12, 8]

plt.pie(percentages, labels=transport_modes, autopct='%1.1f%%')
plt.title('Transportation Mode Distribution')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()
```

Slide 7: Hàm subplots()

Hàm subplots() cho phép bạn tạo nhiều ô trong một hình. Điều này hữu ích khi so sánh các dữ liệu khác nhau hoặc trực quan hóa các khía cạnh khác của cùng một dữ liệu. Hãy tạo một hình với bốn ô khác nhau.

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Plot 1: Line plot
x = np.linspace(0, 10, 100)
axs[0, 0].plot(x, np.sin(x))
axs[0, 0].set_title('Sine Wave')

# Plot 2: Scatter plot
axs[0, 1].scatter(np.random.rand(50), np.random.rand(50))
axs[0, 1].set_title('Random Scatter')

# Plot 3: Bar plot
axs[1, 0].bar(['A', 'B', 'C', 'D'], [3, 7, 2, 5])
axs[1, 0].set_title('Bar Chart')

# Plot 4: Histogram
axs[1, 1].hist(np.random.normal(0, 1, 1000), bins=30)
axs[1, 1].set_title('Normal Distribution')

plt.tight_layout()
plt.show()
```

Slide 8: Hàm imshow()

Hàm imshow() được sử dụng để hiển thị hình ảnh hoặc mảng 2D dưới dạng hình ảnh được mã hóa màu. Nó đặc biệt hữu ích để trực tiếp hóa ma trận, bản đồ nhiệt hoặc hình ảnh thực tế. Hãy tạo một bản đồ nhiệt đơn giản bằng cách sử dụng ngẫu nhiên dữ liệu.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate a random 2D array
data = np.random.rand(10, 10)

plt.imshow(data, cmap='viridis')
plt.colorbar()
plt.title('Heatmap of Random Data')
plt.show()
```

Slide 9: Hàm title()

Hàm title() bổ sung thêm tiêu đề vào cốt truyện của bạn, cung cấp bối cảnh và sự rõ ràng. Đây là một chức năng đơn giản nhưng quan trọng để làm cho hình ảnh trực quan của bạn có nhiều thông tin hơn. Hãy tạo một cốt truyện có tiêu đề mô tả.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('Sine Wave Over One Complete Cycle', fontsize=16, fontweight='bold')
plt.xlabel('Angle (radians)')
plt.ylabel('Amplitude')
plt.show()
```

Slide 10: Hàm truyền thuyết()

Hàm legend() bổ sung chú thích vào biểu đồ của bạn, điều này rất cần thiết khi bạn có nhiều dữ liệu trong một biểu đồ. Nó giúp người xem hiểu được từng dòng hoặc chuỗi đại diện cho điều gì.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.title('Sine and Cosine Waves')
plt.xlabel('Angle (radians)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
```

Slide 11: Hàm xlabel() và ylabel()

Các hàm xlabel() và ylabel() được sử dụng để gắn nhãn cho trục x và y tương ứng của biểu đồ. Nhãn này cung cấp thông tin quan trọng về những gì các trục đại diện, làm cho biểu đồ của bạn dễ hiểu hơn.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate data for a quadratic function
x = np.linspace(-10, 10, 100)
y = x**2

plt.plot(x, y)
plt.title('Quadratic Function')
plt.xlabel('X-axis: Input Values', fontsize=12)
plt.ylabel('Y-axis: Output Values (x^2)', fontsize=12)
plt.grid(True)
plt.show()
```

Slide 12: Cốt truyện giao diện tùy chỉnh

Matplotlib cung cấp nhiều tùy chọn để điều chỉnh giao diện ô của bạn. Bạn có thể thay đổi màu sắc, kiểu kẻ, điểm đánh dấu, v.v. Vui lòng tạo một cách điều chỉnh kiểu tùy ý.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, color='blue', linestyle='--', linewidth=2, label='sin(x)')
plt.plot(x, y2, color='red', linestyle=':', linewidth=2, label='cos(x)')
plt.title('Customized Sin and Cos Waves', fontsize=16)
plt.xlabel('X-axis', fontsize=14)
plt.ylabel('Y-axis', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='-', alpha=0.7)
plt.show()
```

Slide 13: Lưu lô đất

Matplotlib cho phép bạn lưu các ô của mình ở nhiều định dạng tệp khác nhau. Hàm savefig() được sử dụng cho mục đích này. Vui lòng tạo một biểu đồ và lưu nó dưới dạng tệp PNG và PDF.

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('Angle (radians)')
plt.ylabel('Amplitude')

# Save as PNG
plt.savefig('sine_wave.png', dpi=300, bbox_inches='tight')

# Save as PDF
plt.savefig('sine_wave.pdf', bbox_inches='tight')

plt.show()
```

Trang trình bày 14: Ví dụ thực tế: Hiển thị dữ liệu trực quan

Hãy tạo một ví dụ thực tế, phức tạp hơn bằng cách trực quan hóa dữ liệu thời gian. Chúng tôi sẽ vẽ dữ liệu về nhiệt độ và lượng mưa cho một thành phố trong hơn một năm.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate mock weather data
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
temperature = [5, 7, 10, 15, 20, 25, 28, 27, 22, 15, 10, 6]
precipitation = [50, 40, 45, 60, 70, 80, 85, 90, 80, 70, 60, 55]

fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:red'
ax1.set_xlabel('Months')
ax1.set_ylabel('Temperature (°C)', color=color)
ax1.plot(months, temperature, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Precipitation (mm)', color=color)
ax2.bar(months, precipitation, alpha=0.3, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Temperature and Precipitation Over a Year', fontsize=16)
fig.tight_layout()
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người quan tâm đến việc tìm hiểu sâu hơn về Matplotlib và trực quan hóa dữ liệu trong Python, đây là một số tài nguyên có giá trị:

1. Tài liệu chính thức của Matplotlib: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
2. "Trực quan hóa bằng Matplotlib" của Nicolas P. Rougier (ArXiv): [https://arxiv.org/abs/1805.03383](https://arxiv.org/abs/1805.03383)
3. "Mười quy tắc đơn giản để có số liệu đẹp hơn" của Nicolas P. Rougier và cộng sự. (ArXiv): [https://arxiv.org/abs/1411.7396](https://arxiv.org/abs/1411.7396)

Các tài nguyên này cung cấp những giải pháp sâu sắc, kỹ thuật nâng cao và các phương pháp hay nhất để tạo hiệu ứng trực quan hóa bằng Matplotlib.
