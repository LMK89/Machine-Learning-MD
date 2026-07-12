## Matplotlib Subplot Khảm Một giải pháp thay thế linh hoạt cho subplots()
Slide 1: Giới thiệu về Subplot Khảm

Phương thức plt.subplot\_mosaic() cách mạng hóa cách chúng ta tạo các bố cục ô con phức tạp trong Matplotlib. Không giống như plt.subplots() truyền thống, nó cho phép xác định các sắp xếp tùy chỉnh bằng cách sử dụng các chuỗi kiểu nghệ thuật ASCII đơn giản, cung cấp một cách trực quan và linh hoạt để thiết kế trực quan hóa nhiều ô.

```python
import matplotlib.pyplot as plt
import numpy as np

# Define the mosaic layout using ASCII art string
layout = """
AB
AC
"""

# Create the mosaic subplot layout
fig, axd = plt.subplot_mosaic(layout)

# Generate sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

# Plot in each subplot using dictionary-style access
axd['A'].plot(x, y1, 'r-', label='sin(x)')
axd['B'].plot(x, y2, 'b-', label='cos(x)')
axd['C'].plot(x, y3, 'g-', label='tan(x)')

# Add labels and titles
for ax_key in axd:
    axd[ax_key].set_title(f'Subplot {ax_key}')
    axd[ax_key].legend()

plt.tight_layout()
plt.show()
```

Trang trình bày 2: Bố cục khảm phức tạp

Subplot khảm hỗ trợ các bố cục phức tạp thông qua các danh sách chuỗi lồng nhau, cho phép tạo các sắp xếp lưới phức tạp với các kích cỡ khác nhau. Cách tiếp cận này loại bỏ nhu cầu thao tác GridSpec trong khi vẫn duy trì toàn quyền kiểm soát việc định vị và mở rộng ô phụ.

```python
import matplotlib.pyplot as plt
import numpy as np

# Define a complex mosaic layout
layout = [
    ['A B B'],
    ['C C D'],
    ['E F F']
]

fig, axd = plt.subplot_mosaic(layout, figsize=(10, 8))

# Generate random data for each subplot
np.random.seed(42)
data = {key: np.random.randn(100) for key in 'ABCDEF'}

# Create different plot types for each subplot
axd['A'].hist(data['A'], bins=20)
axd['B'].scatter(range(100), data['B'])
axd['C'].plot(data['C'])
axd['D'].boxplot(data['D'])
axd['E'].violinplot(data['E'])
axd['F'].hist2d(data['E'], data['F'], bins=20)

# Customize each subplot
for key in axd:
    axd[key].set_title(f'Plot {key}')

plt.tight_layout()
plt.show()
```

Trang trình bày 3: Tỷ lệ chiều cao động

Khảm ô phụ của Matplotlib cho phép kiểm soát tinh vi các kích thước ô phụ thông qua tỷ lệ chiều cao. Tính năng này cho phép tùy chỉnh bố cục chính xác trong khi vẫn duy trì hệ thống định nghĩa bố cục dựa trên chuỗi trực quan.

```python
import matplotlib.pyplot as plt
import numpy as np

# Define layout with custom height ratios
layout = """
AAA
BBC
BBC
"""

# Create figure with height ratios
fig, axd = plt.subplot_mosaic(
    layout,
    height_ratios=[1, 2, 2],
    figsize=(10, 8)
)

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(-0.1 * x)

# Create different visualizations
axd['A'].plot(x, y, 'r-', label='Damped Sine')
axd['B'].imshow(np.random.rand(20, 20))
axd['C'].hist2d(x, y, bins=30)

# Add titles and customize
for key in axd:
    axd[key].set_title(f'Region {key}')

plt.tight_layout()
plt.show()
```

Slide 4: Các mẫu tranh khảm nâng cao

Phương pháp khảm subplot hỗ trợ các mẫu phức tạp bao gồm bố cục lồng nhau và khoảng trống. Tính năng này cho phép tạo các trang tổng quan phức tạp và sắp xếp các số liệu có chất lượng xuất bản với độ phức tạp mã tối thiểu.

```python
import matplotlib.pyplot as plt
import numpy as np

# Define complex layout with empty spaces
layout = """
A.B.C
DDDDD
.EEE.
"""

fig, axd = plt.subplot_mosaic(
    layout,
    figsize=(12, 8),
    empty_sentinel="."  # Defines empty spaces
)

# Generate sample data
t = np.linspace(0, 10, 1000)
signals = {
    'A': np.sin(2*np.pi*t),
    'B': np.cos(2*np.pi*t),
    'C': np.tan(t),
    'D': np.exp(-0.1*t) * np.sin(2*np.pi*t),
    'E': np.random.randn(1000)
}

# Create different plot types
axd['A'].plot(t[:100], signals['A'][:100], 'r-')
axd['B'].scatter(t[::50], signals['B'][::50], alpha=0.5)
axd['C'].hist(signals['C'], bins=30)
axd['D'].specgram(signals['D'], NFFT=128)
axd['E'].hist2d(signals['D'], signals['E'], bins=50)

plt.tight_layout()
plt.show()
```

Trang trình bày 5: Ứng dụng thực tế: Bảng thông tin tài chính

Việc tạo bảng điều khiển tài chính bằng cách sử dụng khảm ô phụ thể hiện ứng dụng thực tế của nó trong việc trực quan hóa dữ liệu. Ví dụ này cho thấy cách sắp xếp nhiều số liệu tài chính theo bố cục mạch lạc và hấp dẫn về mặt trực quan.

```python
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Generate sample financial data
dates = [datetime.now() - timedelta(days=x) for x in range(100)]
stock_price = np.cumsum(np.random.randn(100)) + 100
volume = np.random.randint(1000, 5000, 100)
volatility = np.abs(np.diff(stock_price))
moving_avg = np.convolve(stock_price, np.ones(20)/20, mode='valid')

# Define dashboard layout
layout = """
AAAA
BBCC
DDEE
"""

fig, axd = plt.subplot_mosaic(layout, figsize=(12, 8))

# Create financial visualizations
axd['A'].plot(dates, stock_price, 'b-', label='Stock Price')
axd['A'].plot(dates[19:], moving_avg, 'r--', label='20-day MA')
axd['B'].bar(dates, volume, alpha=0.6, label='Trading Volume')
axd['C'].hist(stock_price, bins=30, orientation='horizontal')
axd['D'].plot(dates[1:], volatility, 'g-', label='Volatility')
axd['E'].boxplot([stock_price, volume])

# Customize appearance
for key in axd:
    axd[key].set_title(f'Panel {key}')
    axd[key].legend()

plt.tight_layout()
plt.show()
```

Slide 6: Trực quan hóa dữ liệu khoa học

Subplot khảm vượt trội trong trực quan hóa khoa học trong đó các mối quan hệ dữ liệu phức tạp cần được hiển thị đồng thời. Việc triển khai này trình bày cách tạo chế độ xem toàn diện về dữ liệu thử nghiệm bằng các kỹ thuật trực quan hóa khác nhau.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Define scientific visualization layout
layout = """
AAAB
CCCB
DDDE
"""

fig, axd = plt.subplot_mosaic(layout, figsize=(12, 10))

# Generate experimental data
x = np.linspace(-5, 5, 100)
exp_data = norm.pdf(x, loc=0, scale=1) + np.random.normal(0, 0.02, 100)
measurement_points = np.random.choice(x, 20)
measurement_values = norm.pdf(measurement_points, loc=0, scale=1)

# Create scientific plots
axd['A'].contourf(np.random.rand(20, 20))
axd['A'].set_title('2D Field Distribution')

axd['B'].scatter(measurement_values, measurement_points, c='red', alpha=0.6)
axd['B'].set_title('Scattered Measurements')

axd['C'].plot(x, exp_data, 'b-', label='Experimental')
axd['C'].plot(x, norm.pdf(x, 0, 1), 'r--', label='Theoretical')
axd['C'].set_title('Comparison with Theory')

spectrum = np.fft.fft(exp_data)
freq = np.fft.fftfreq(len(x))
axd['D'].plot(freq, np.abs(spectrum))
axd['D'].set_title('Frequency Spectrum')

axd['E'].hist2d(x, exp_data, bins=30)
axd['E'].set_title('Density Distribution')

plt.tight_layout()
plt.show()
```

Trang trình bày 7: Bảng điều khiển phân tích dữ liệu tương tác

Bố cục khảm cho phép tạo bảng thông tin tương tác nơi có thể phân tích đồng thời nhiều khía cạnh dữ liệu. Ví dụ này thể hiện giao diện phân tích dữ liệu toàn diện với hình ảnh trực quan được đồng bộ hóa.

```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Define analysis dashboard layout
layout = """
AB
CD
"""

fig, axd = plt.subplot_mosaic(layout, figsize=(12, 10))

# Generate multivariate data
n_points = 1000
x = np.random.normal(0, 1, n_points)
y = x * 0.5 + np.random.normal(0, 0.5, n_points)
z = x * 0.3 + y * 0.7 + np.random.normal(0, 0.3, n_points)

# Create synchronized visualizations
axd['A'].scatter(x, y, c=z, cmap='viridis', alpha=0.6)
axd['A'].set_title('X-Y Correlation with Z as color')

axd['B'].hist2d(y, z, bins=30, cmap='plasma')
axd['B'].set_title('Y-Z Density Distribution')

# Create violin plots for distributions
axd['C'].violinplot([x, y, z])
axd['C'].set_xticks([1, 2, 3])
axd['C'].set_xticklabels(['X', 'Y', 'Z'])
axd['C'].set_title('Distribution Comparison')

# Create correlation heatmap
corr_matrix = np.corrcoef([x, y, z])
im = axd['D'].imshow(corr_matrix, cmap='coolwarm')
axd['D'].set_title('Correlation Matrix')
plt.colorbar(im, ax=axd['D'])

# Add labels
for ax in axd.values():
    ax.grid(True)

plt.tight_layout()
plt.show()
```

Slide 8: Trực quan hóa chuỗi thời gian

Bố cục khảm đặc biệt hiệu quả để phân tích chuỗi thời gian, cho phép hiển thị đồng thời nhiều khía cạnh thời gian trong khi vẫn duy trì mối quan hệ rõ ràng giữa các thành phần khác nhau.

```python
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Generate time series data
dates = np.array([datetime.now() + timedelta(days=x) for x in range(100)])
signal = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
trend = np.cumsum(np.random.normal(0, 0.1, 100))

# Define time series layout
layout = """
AAA
BBC
DDD
"""

fig, axd = plt.subplot_mosaic(layout, figsize=(12, 8))

# Raw signal plot
axd['A'].plot(dates, signal, 'b-', label='Raw Signal')
axd['A'].set_title('Time Series Data')
axd['A'].legend()

# Rolling statistics
window = 10
rolling_mean = np.convolve(signal, np.ones(window)/window, mode='valid')
axd['B'].plot(dates[window-1:], rolling_mean, 'r-', label=f'{window}-point Moving Average')
axd['B'].set_title('Rolling Statistics')
axd['B'].legend()

# Histogram of values
axd['C'].hist(signal, bins=30, orientation='horizontal')
axd['C'].set_title('Value Distribution')

# Trend analysis
axd['D'].plot(dates, trend + signal, 'g-', label='Signal + Trend')
axd['D'].plot(dates, trend, 'r--', label='Trend')
axd['D'].set_title('Trend Analysis')
axd['D'].legend()

plt.tight_layout()
plt.show()
```

Trang trình bày 9: Đánh giá mô hình học máy

Bố cục khảm ô phụ cung cấp một khuôn khổ tuyệt vời để trực quan hóa các số liệu hiệu suất của mô hình học máy. Ví dụ này minh họa bảng thông tin đánh giá mô hình toàn diện với nhiều tiêu chí đánh giá.

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve

# Define evaluation dashboard layout
layout = """
ABC
DDD
"""

fig, axd = plt.subplot_mosaic(layout, figsize=(15, 8))

# Generate sample model results
np.random.seed(42)
y_true = np.random.randint(0, 2, 1000)
y_pred_proba = np.clip(y_true + np.random.normal(0, 0.3, 1000), 0, 1)
y_pred = (y_pred_proba > 0.5).astype(int)

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
axd['A'].plot(fpr, tpr, 'b-', label='ROC Curve')
axd['A'].plot([0, 1], [0, 1], 'r--', label='Random')
axd['A'].set_title('ROC Curve')
axd['A'].legend()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
axd['B'].plot(recall, precision, 'g-', label='PR Curve')
axd['B'].set_title('Precision-Recall Curve')
axd['B'].legend()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
im = axd['C'].imshow(cm, cmap='Blues')
axd['C'].set_title('Confusion Matrix')
plt.colorbar(im, ax=axd['C'])

# Prediction Distribution
axd['D'].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.5, label='Class 0')
axd['D'].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.5, label='Class 1')
axd['D'].set_title('Prediction Distribution')
axd['D'].legend()

plt.tight_layout()
plt.show()
```

Slide 10: Trực quan hóa dữ liệu không gian địa lý

Khảm subplot tạo điều kiện thuận lợi cho việc tạo trực quan hóa không gian địa lý phức tạp bằng cách cho phép hiển thị nhiều chế độ xem bản đồ và phân tích liên quan một cách mạch lạc.

```python
import matplotlib.pyplot as plt
import numpy as np

# Define geospatial layout
layout = """
AAB
AAC
DDD
"""

fig, axd = plt.subplot_mosaic(layout, figsize=(12, 10))

# Generate sample geospatial data
lat = np.random.uniform(30, 45, 100)
lon = np.random.uniform(-120, -70, 100)
intensity = np.random.uniform(0, 1, 100)

# Main map
scatter = axd['A'].scatter(lon, lat, c=intensity, cmap='viridis', alpha=0.6)
axd['A'].set_title('Geographic Distribution')
plt.colorbar(scatter, ax=axd['A'], label='Intensity')

# Latitude distribution
axd['B'].hist(lat, bins=20, orientation='horizontal')
axd['B'].set_title('Latitude Distribution')

# Longitude distribution
axd['C'].hist(lon, bins=20)
axd['C'].set_title('Longitude Distribution')

# Intensity surface plot
xi = np.linspace(lon.min(), lon.max(), 50)
yi = np.linspace(lat.min(), lat.max(), 50)
xi, yi = np.meshgrid(xi, yi)
from scipy.interpolate import griddata
zi = griddata((lon, lat), intensity, (xi, yi), method='cubic')
im = axd['D'].contourf(xi, yi, zi, levels=15, cmap='viridis')
axd['D'].set_title('Intensity Surface')
plt.colorbar(im, ax=axd['D'])

plt.tight_layout()
plt.show()
```

Trang trình bày 11: Khả năng chú thích tùy chỉnh

Hệ thống khảm cốt truyện phụ cung cấp các khả năng mạnh mẽ để thêm chú thích tùy chỉnh và kết nối các yếu tố cốt truyện khác nhau, nâng cao dòng tường thuật của các hình ảnh trực quan phức tạp.

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch

# Define annotation layout
layout = """
AB
CD
"""

fig, axd = plt.subplot_mosaic(layout, figsize=(10, 10))

# Generate sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create connected plots with annotations
axd['A'].plot(x, y1, 'r-')
axd['A'].set_title('Primary Signal')
point_a = (5, np.sin(5))
axd['A'].plot(*point_a, 'ko')

axd['B'].plot(x, y2, 'b-')
axd['B'].set_title('Secondary Signal')
point_b = (5, np.cos(5))
axd['B'].plot(*point_b, 'ko')

# Add connection between points
con = ConnectionPatch(
    xyA=point_a, xyB=point_b,
    coordsA="data", coordsB="data",
    axesA=axd['A'], axesB=axd['B'],
    color="gray", linestyle="--"
)
fig.add_artist(con)

# Add detailed views
axd['C'].plot(x[40:60], y1[40:60], 'r-')
axd['C'].set_title('Detail View 1')
axd['C'].fill_between(x[40:60], y1[40:60], alpha=0.3)

axd['D'].plot(x[40:60], y2[40:60], 'b-')
axd['D'].set_title('Detail View 2')
axd['D'].fill_between(x[40:60], y2[40:60], alpha=0.3)

# Add annotations
for ax in axd.values():
    ax.grid(True)
    ax.annotate('Peak',
                xy=(5, 0.8),
                xytext=(6, 0.9),
                arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.show()
```

Slide 12: Điều chỉnh bố cục động

Hệ thống khảm ô phụ cho phép điều chỉnh bố cục động dựa trên đặc điểm dữ liệu. Việc triển khai này trình bày cách tạo bố cục đáp ứng thích ứng với các cấu hình dữ liệu và tỷ lệ khung hình khác nhau.

```python
import matplotlib.pyplot as plt
import numpy as np

# Function to create adaptive layout
def create_adaptive_layout(data_shape):
    if data_shape[0] > data_shape[1]:
        return """
        AB
        AC
        AD
        """
    else:
        return """
        AAA
        BCD
        """

# Generate sample data
data = np.random.randn(100, 50)  # Change shape to test different layouts
layout = create_adaptive_layout(data.shape)

fig, axd = plt.subplot_mosaic(layout, figsize=(12, 8))

# Create visualizations that adapt to data shape
main_img = axd['A'].imshow(data, aspect='auto', cmap='viridis')
plt.colorbar(main_img, ax=axd['A'])
axd['A'].set_title('Main Data View')

# Add complementary visualizations
axd['B'].plot(np.mean(data, axis=1), 'r-', label='Row Means')
axd['B'].set_title('Row Statistics')
axd['B'].legend()

axd['C'].plot(np.mean(data, axis=0), 'b-', label='Column Means')
axd['C'].set_title('Column Statistics')
axd['C'].legend()

if 'D' in axd:
    axd['D'].hist2d(data.flatten(), np.roll(data.flatten(), 1), bins=50)
    axd['D'].set_title('Lag Plot')

plt.tight_layout()
plt.show()
```

Slide 13: Trực quan hóa thống kê nâng cao

Việc triển khai này cho thấy cách sử dụng khảm ô phụ để tạo bảng thông tin phân tích thống kê toàn diện với nhiều chế độ xem phối hợp của cùng một tập dữ liệu.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Define statistical dashboard layout
layout = """
ABC
DDD
"""

fig, axd = plt.subplot_mosaic(layout, figsize=(15, 8))

# Generate multivariate sample data
n_samples = 1000
data1 = np.random.normal(0, 1, n_samples)
data2 = data1 * 0.5 + np.random.normal(0, 0.5, n_samples)

# QQ Plot
stats.probplot(data1, dist="norm", plot=axd['A'])
axd['A'].set_title('Normal Q-Q Plot')

# Correlation Plot
axd['B'].scatter(data1, data2, alpha=0.5)
axd['B'].set_title('Correlation Plot')

# Joint Distribution
xmin, xmax = data1.min(), data1.max()
ymin, ymax = data2.min(), data2.max()
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([data1, data2])
kernel = stats.gaussian_kde(values)
z = np.reshape(kernel(positions).T, xx.shape)

im = axd['C'].imshow(z.T, extent=[xmin, xmax, ymin, ymax], origin='lower')
plt.colorbar(im, ax=axd['C'])
axd['C'].set_title('2D Kernel Density')

# Time Series View with Rolling Statistics
t = np.arange(n_samples)
axd['D'].plot(t, data1, 'b-', alpha=0.5, label='Series 1')
axd['D'].plot(t, data2, 'r-', alpha=0.5, label='Series 2')

# Add rolling mean
window = 50
roll_mean1 = np.convolve(data1, np.ones(window)/window, mode='valid')
roll_mean2 = np.convolve(data2, np.ones(window)/window, mode='valid')
axd['D'].plot(t[window-1:], roll_mean1, 'b-', linewidth=2, label='Rolling Mean 1')
axd['D'].plot(t[window-1:], roll_mean2, 'r-', linewidth=2, label='Rolling Mean 2')
axd['D'].legend()
axd['D'].set_title('Time Series View with Rolling Means')

plt.tight_layout()
plt.show()
```

Trang trình bày 14: Tài nguyên bổ sung

* Bài viết ArXiv về kỹ thuật hiển thị nâng cao: [https://arxiv.org/abs/2007.07799](https://arxiv.org/abs/2007.07799)
* Đánh giá phương pháp trực quan hóa thống kê: [https://arxiv.org/abs/1909.03083](https://arxiv.org/abs/1909.03083)
* Tài liệu khảm ô phụ Matplotlib: [https://matplotlib.org/stable/](https://matplotlib.org/stable/)
* Tìm kiếm được đề xuất trên Google:
    * "Ví dụ về khảm subplot của Matplotlib"
    * "Bố cục ô phụ nâng cao trong Python"
    * "Bố cục trực quan hóa tùy chỉnh với Matplotlib"
