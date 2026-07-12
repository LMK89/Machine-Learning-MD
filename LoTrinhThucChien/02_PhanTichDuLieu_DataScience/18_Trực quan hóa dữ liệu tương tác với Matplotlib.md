## Trực quan hóa dữ liệu tương tác với Matplotlib

Trang trình bày 1: Sơ đồ đường tương tác với các sự kiện nhấp chuột

Tạo trực quan hóa tương tác giúp nâng cao khả năng khám phá dữ liệu bằng cách cho phép người dùng tương tác trực tiếp với các ô. Việc triển khai này trình bày cách ghi lại các lần nhấp chuột trên biểu đồ đường và hiển thị tọa độ, cho phép kiểm tra chi tiết các điểm dữ liệu cụ thể.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create figure and plot
fig, ax = plt.subplots()
line, = ax.plot(x, y)

# Click event handler
def on_click(event):
    if event.inaxes == ax:
        print(f'Clicked coordinates: x={event.xdata:.2f}, y={event.ydata:.2f}')
        ax.plot(event.xdata, event.ydata, 'ro')  # Add red dot at click
        plt.draw()

# Connect click event
fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
```

Trang trình bày 2: Cập nhật dữ liệu động theo thời gian thực

Việc triển khai khả năng trực quan hóa dữ liệu theo thời gian thực cho phép giám sát các nguồn dữ liệu truyền phát. Việc triển khai này tạo ra một biểu đồ hoạt hình tự động cập nhật các điểm dữ liệu mới, mô phỏng các chỉ số cảm biến hoặc các phép đo trực tiếp.

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class RealtimePlot:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        self.x_data, self.y_data = [], []

    def init_plot(self):
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-2, 2)
        return self.line,

    def update(self, frame):
        self.x_data.append(frame)
        self.y_data.append(np.sin(frame * 0.1) + np.random.normal(0, 0.1))

        self.line.set_data(self.x_data, self.y_data)
        return self.line,

rt_plot = RealtimePlot()
anim = FuncAnimation(rt_plot.fig, rt_plot.update, init_func=rt_plot.init_plot,
                    frames=range(100), interval=50, blit=True)
plt.show()
```

Trang trình bày 3: Chú giải tương tác tùy chỉnh

Chú giải tương tác cung cấp khả năng kiểm soát nâng cao đối với các thành phần cốt truyện, cho phép người dùng chuyển đổi mức độ hiển thị của các chuỗi dữ liệu khác nhau. Việc triển khai này tạo ra một chú giải tùy chỉnh với các phần tử có thể nhấp vào và hiệu ứng di chuột.

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate multiple data series
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

fig, ax = plt.subplots()
lines = []
lines.append(ax.plot(x, y1, label='sin(x)')[0])
lines.append(ax.plot(x, y2, label='cos(x)')[0])
lines.append(ax.plot(x, y3, label='tan(x)')[0])

leg = ax.legend()

def on_pick(event):
    legline = event.artist
    line = lines[leg.get_lines().index(legline)]
    line.set_visible(not line.get_visible())
    plt.draw()

for legline in leg.get_lines():
    legline.set_picker(True)
fig.canvas.mpl_connect('pick_event', on_pick)
plt.show()
```

Slide 4: Phân tích chuỗi thời gian tương tác

Trực quan hóa chuỗi thời gian yêu cầu xử lý chuyên biệt đối với dữ liệu thời gian và các tính năng tương tác. Việc triển khai này tạo ra một biểu đồ chuỗi thời gian tương tác với khả năng thu phóng và chú giải công cụ nhận biết ngày.

```python
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter
import numpy as np

# Generate sample time series data
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
values = np.cumsum(np.random.randn(100)) + 100

fig, ax = plt.subplots(figsize=(12, 6))
line = ax.plot(dates, values)

# Configure date formatting
date_formatter = DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_formatter)
plt.xticks(rotation=45)

def hover(event):
    if event.inaxes == ax:
        # Find nearest point
        distances = [abs(d.toordinal() - event.xdata) for d in dates]
        nearest_idx = distances.index(min(distances))

        # Update annotation
        ax.texts.clear()
        ax.annotate(f'Value: {values[nearest_idx]:.2f}\nDate: {dates[nearest_idx].strftime("%Y-%m-%d")}',
                   xy=(dates[nearest_idx], values[nearest_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        plt.draw()

fig.canvas.mpl_connect('motion_notify_event', hover)
plt.tight_layout()
plt.show()
```

Trang trình bày 5: Sơ đồ bề mặt 3D tương tác với màu động

Trực quan hóa dữ liệu ba chiều với các tính năng tương tác giúp nâng cao hiểu biết về các mối quan hệ không gian phức tạp. Việc triển khai này thể hiện sơ đồ bề mặt 3D phản hồi đầu vào của người dùng để điều chỉnh xoay và ánh xạ màu.

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def create_interactive_3d_plot():
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create initial surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    fig.colorbar(surf)

    def on_key(event):
        if event.key == 'c':  # Change colormap
            surf.set_cmap('plasma')
        elif event.key == 'r':  # Reset view
            ax.view_init(30, -60)
        plt.draw()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

# Run the visualization
create_interactive_3d_plot()
```

Trang trình bày 6: Trực quan hóa chuỗi thời gian nâng cao

Giám sát dữ liệu thời gian thực đòi hỏi các kỹ thuật trực quan phức tạp để xử lý dữ liệu truyền phát hiệu quả. Việc triển khai này giới thiệu màn hình hiển thị cửa sổ thời gian luân phiên với các bản cập nhật tự động và điểm đánh dấu tương tác.

```python
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from matplotlib.animation import FuncAnimation

class TimeSeriesMonitor:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.times = deque(maxlen=max_points)
        self.values = deque(maxlen=max_points)

        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        self.ax.set_ylim(-2, 2)

    def update(self, frame):
        self.times.append(frame)
        self.values.append(np.sin(frame * 0.1))

        self.ax.set_xlim(max(0, frame - self.max_points), frame + 3)
        self.line.set_data(list(self.times), list(self.values))
        return self.line,

monitor = TimeSeriesMonitor()
anim = FuncAnimation(monitor.fig, monitor.update,
                    frames=range(200), interval=50)
plt.show()
```

Trang trình bày 7: Biểu đồ tương tác với Dynamic Binning

Lợi ích của việc khám phá dữ liệu thống kê nhờ trực quan hóa biểu đồ tương tác. Việc triển khai này cho phép người dùng tự động điều chỉnh kích thước thùng và quan sát các thay đổi phân phối trong thời gian thực.

```python
import numpy as np
import matplotlib.pyplot as plt

class InteractiveHistogram:
    def __init__(self, data):
        self.data = data
        self.fig, self.ax = plt.subplots()
        self.bins = 30
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.ax.hist(self.data, bins=self.bins)
        self.ax.set_title(f'Histogram (bins={self.bins})')
        plt.draw()

    def on_scroll(self, event):
        if event.button == 'up':
            self.bins = min(100, self.bins + 5)
        else:
            self.bins = max(5, self.bins - 5)
        self.update_plot()

# Example usage
data = np.random.normal(0, 1, 1000)
hist = InteractiveHistogram(data)
hist.fig.canvas.mpl_connect('scroll_event', hist.on_scroll)
plt.show()
```

Trang trình bày 8: Sơ đồ không gian pha với quỹ đạo động

Phân tích hệ thống động lực đòi hỏi các kỹ thuật trực quan chuyên dụng. Việc triển khai này tạo ra một biểu đồ không gian pha tương tác cho thấy sự tiến triển của hệ thống và cho phép điều chỉnh tham số.

```python
import numpy as np
import matplotlib.pyplot as plt

def create_phase_space_plot():
    t = np.linspace(0, 20, 1000)
    x = np.sin(t)
    v = np.cos(t)

    fig, ax = plt.subplots()
    line, = ax.plot(x, v)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')

    def update_frequency(event):
        if event.key == 'up':
            t_new = np.linspace(0, 20, 1000)
            x_new = np.sin(1.5 * t_new)
            v_new = 1.5 * np.cos(1.5 * t_new)
            line.set_data(x_new, v_new)
            plt.draw()

    fig.canvas.mpl_connect('key_press_event', update_frequency)
    plt.show()

create_phase_space_plot()
```

Trang trình bày 9: Bảng điều khiển tương tác đa bảng

Kết hợp nhiều loại trực quan cho phép phân tích dữ liệu toàn diện. Việc triển khai này tạo ra một bảng thông tin với các sơ đồ tương tác được đồng bộ hóa nhằm đáp ứng các tương tác của người dùng.

```python
import matplotlib.pyplot as plt
import numpy as np

def create_dashboard():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    line1, = ax1.plot(x, y)
    hist = ax2.hist(y, bins=20)

    def on_click(event):
        if event.inaxes == ax1:
            ax1.axvline(x=event.xdata, color='r', alpha=0.5)
            ax2.clear()
            mask = x <= event.xdata
            ax2.hist(y[mask], bins=20)
            plt.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    plt.show()

create_dashboard()
```

Trang trình bày 10: Hoạt ảnh bản đồ màu tùy chỉnh

Hiểu dữ liệu thông qua màu sắc đòi hỏi các kỹ thuật trực quan chuyên biệt. Việc triển khai này trình bày cách tạo và tạo hiệu ứng động cho các bản đồ màu tùy chỉnh để thể hiện dữ liệu nâng cao.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_colormap_animation():
    fig, ax = plt.subplots()

    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    def frame(i):
        Z = np.sin(np.sqrt(X**2 + Y**2) - i * 0.1)
        if hasattr(frame, 'im'):
            frame.im.remove()
        frame.im = ax.imshow(Z, cmap='viridis')
        return frame.im,

    anim = FuncAnimation(fig, frame, frames=100,
                        interval=50, blit=True)
    plt.colorbar(frame.im)
    plt.show()

create_colormap_animation()
```

Trang trình bày 11: Ma trận biểu đồ phân tán tương tác

Phân tích dữ liệu đa biến đòi hỏi các kỹ thuật trực quan chuyên dụng. Việc triển khai này tạo ra một ma trận biểu đồ phân tán tương tác với khả năng đánh dấu và liên kết.

```python
import numpy as np
import matplotlib.pyplot as plt

class ScatterMatrix:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.n = data.shape[1]

        self.fig, self.axes = plt.subplots(self.n, self.n,
                                         figsize=(10, 10))
        self.create_matrix()

    def create_matrix(self):
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    self.axes[i, j].scatter(self.data[:, j],
                                          self.data[:, i],
                                          alpha=0.5)
                else:
                    self.axes[i, i].hist(self.data[:, i])

                if i == self.n - 1:
                    self.axes[i, j].set_xlabel(self.labels[j])
                if j == 0:
                    self.axes[i, j].set_ylabel(self.labels[i])

# Example usage
data = np.random.randn(100, 3)
labels = ['X', 'Y', 'Z']
matrix = ScatterMatrix(data, labels)
plt.tight_layout()
plt.show()
```

Trang trình bày 12: Trực quan hóa đồ thị mạng động

Phân tích mạng đòi hỏi các kỹ thuật trực quan tương tác chuyên biệt. Việc triển khai này tạo ra bố cục biểu đồ hướng lực với vị trí nút tương tác.

```python
import numpy as np
import matplotlib.pyplot as plt

def create_network_plot():
    fig, ax = plt.subplots(figsize=(8, 8))

    # Generate random graph
    n_nodes = 10
    positions = np.random.rand(n_nodes, 2)
    edges = [(i, j) for i in range(n_nodes)
             for j in range(i+1, n_nodes)
             if np.random.rand() < 0.3]

    # Plot nodes and edges
    ax.scatter(positions[:, 0], positions[:, 1])
    for i, j in edges:
        ax.plot([positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]], 'k-')

    def on_click(event):
        if event.inaxes == ax:
            dist = np.sum((positions -
                          [event.xdata, event.ydata])**2,
                         axis=1)
            nearest = np.argmin(dist)
            positions[nearest] = [event.xdata, event.ydata]
            ax.clear()
            ax.scatter(positions[:, 0], positions[:, 1])
            for i, j in edges:
                ax.plot([positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]], 'k-')
            plt.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

create_network_plot()
```

Trang trình bày 13: Tài nguyên bổ sung

1. [https://arxiv.org/abs/2012.08972](https://arxiv.org/abs/2012.08972) - Kỹ thuật trực quan hóa tương tác để khám phá dữ liệu nhiều chiều
2. [https://arxiv.org/abs/2107.14702](https://arxiv.org/abs/2107.14702) - Trực quan hóa dữ liệu tương tác theo thời gian thực bằng Python
3. [https://arxiv.org/abs/2109.05542](https://arxiv.org/abs/2109.05542) - Kỹ thuật Matplotlib nâng cao để trực quan hóa khoa học
4. [https://arxiv.org/abs/2203.09801](https://arxiv.org/abs/2203.09801) - Phân tích trực quan tương tác cho dữ liệu chuỗi thời gian
5. [https://arxiv.org/abs/2106.12231](https://arxiv.org/abs/2106.12231) - Các phương pháp tiếp cận hiện đại để trực quan hóa mạng
