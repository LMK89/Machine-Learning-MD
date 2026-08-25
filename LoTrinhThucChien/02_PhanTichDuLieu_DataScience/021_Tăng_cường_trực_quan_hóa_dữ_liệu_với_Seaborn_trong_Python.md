## Tăng cường trực tiếp dữ liệu hóa học với Seaborn trong Python
Slide 1: Giới thiệu về Seaborn

Seaborn là một thư viện Python mạnh mẽ để tạo trực tuyến dữ liệu thống kê. Được xây dựng dựa trên Matplotlib, nó cung cấp giao diện cao cấp để vẽ sơ đồ thống kê hấp dẫn và thông tin phong phú. Seaborn đặc biệt hữu ích cho việc khám phá và hiểu dữ liệu thông qua các loại cốt truyện khác nhau.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load a sample dataset
tips = sns.load_dataset("tips")

# Create a simple scatter plot
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.title("Relationship between Total Bill and Tip")
plt.show()
```

Slide 2: Thiết lập Seaborn

Trước khi đi sâu vào các tính năng của Seaborn, điều cần thiết là môi trường thiết lập của bạn. Seaborn có thể cài đặt bằng pip và nó thường được sử dụng cùng với Pandas để thao tác dữ liệu.

```python
# Install Seaborn (run this in your terminal or command prompt)
# pip install seaborn

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Set the default Seaborn style
sns.set_theme()

# Load a built-in dataset
df = sns.load_dataset("penguins")
print(df.head())
```

Slide 3: Tùy chỉnh thẩm mỹ của cốt truyện

Seaborn cung cấp nhiều chủ đề và bảng màu phù hợp khác nhau để nâng cao diện mạo cho ô tô của bạn. Bạn có thể dễ dàng tùy chỉnh tổng thể giao diện của hình ảnh trực tiếp của mình.

```python
# Set a specific style
sns.set_style("whitegrid")

# Create a plot with a custom color palette
sns.scatterplot(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species", palette="deep")
plt.title("Penguin Bill Dimensions by Species")
plt.show()

# Reset to default style
sns.set_style("darkgrid")
```

Slide 4: Các ô phân phối

Seaborn vượt trội trong công việc phân phối các bản phân phối. Hàm distplot (hiện được thay thế bằng displot) cho phép bạn tạo biểu đồ với hạt nhân mật khẩu.

```python
# Create a distribution plot
sns.displot(df, x="flipper_length_mm", kde=True, hue="species")
plt.title("Distribution of Flipper Lengths")
plt.show()
```

Slide 5: Categorical Plots

Categorical plots in Seaborn help visualize the distribution of a quantitative variable across different categories.

```python
# Create a box plot
sns.boxplot(data=df, x="species", y="body_mass_g")
plt.title("Body Mass Distribution by Penguin Species")
plt.show()

# Create a violin plot
sns.violinplot(data=df, x="species", y="body_mass_g")
plt.title("Body Mass Distribution (Violin Plot)")
plt.show()
```

Slide 6: Đồ thị phục hồi

Các biểu đồ hồi phục của Seaborn rất hữu ích trong công việc hình dung mối quan hệ giữa biến thể và điều chỉnh mô hình hồi phục.

```python
# Create a regression plot
sns.regplot(data=df, x="flipper_length_mm", y="body_mass_g")
plt.title("Relationship between Flipper Length and Body Mass")
plt.show()
```

Slide 7: Pair Plots

Pair plots are an excellent way to visualize relationships between multiple variables in a dataset.

```python
# Create a pair plot
sns.pairplot(df, hue="species")
plt.suptitle("Pair Plot of Penguin Measurements", y=1.02)
plt.show()
```

Trang trình bày 8: Bản đồ nhiệt

Bản đồ nhiệt độ hữu ích để trực quan hóa mối quan hệ giữa các biến trong dữ liệu.

```python
# Create a correlation matrix
corr_matrix = df.corr()

# Create a heatmap
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Penguin Measurements")
plt.show()
```

Trang trình bày 9: Lưới viền

Viền cạnh cho phép bạn tạo nhiều biểu đồ cho các dữ liệu tập hợp khác nhau.

```python
# Create a facet grid
g = sns.FacetGrid(df, col="species", height=4, aspect=1.2)
g.map(sns.scatterplot, "bill_length_mm", "bill_depth_mm")
g.add_legend()
plt.suptitle("Bill Dimensions by Species", y=1.05)
plt.show()
```

Slide 10: Real-Life Example: Environmental Data Analysis

Let's analyze air quality data using Seaborn to visualize pollution levels across different cities.

```python
# Create sample air quality data
air_quality = pd.DataFrame({
    'city': ['New York', 'London', 'Tokyo', 'Beijing', 'Mumbai'] * 12,
    'month': list(range(1, 13)) * 5,
    'aqi': [50, 45, 40, 80, 70, 55, 48, 42, 85, 75, 60, 52,
            40, 35, 30, 75, 65, 45, 38, 32, 78, 68, 50, 42,
            35, 30, 25, 70, 60, 40, 33, 27, 73, 63, 45, 37,
            85, 80, 75, 120, 110, 90, 83, 77, 125, 115, 95, 87,
            75, 70, 65, 110, 100, 80, 73, 67, 115, 105, 85, 77]
})

# Create a line plot to show AQI trends
sns.lineplot(data=air_quality, x='month', y='aqi', hue='city')
plt.title('Air Quality Index (AQI) Trends Across Cities')
plt.xlabel('Month')
plt.ylabel('AQI')
plt.show()
```

Slide 11: Ví dụ thực tế: Trực quan hóa dữ liệu khoa học

quan hóa dữ liệu khoa học là rất quan trọng để hiểu các phức tạp hiện tượng trực quan. Hãy sử dụng Seaborn để phân tích mối quan hệ giữa khối lượng của một hành tinh và chu kỳ đạo của nó.

```python
import numpy as np

# Generate sample exoplanet data
np.random.seed(42)
n_planets = 100
planet_data = pd.DataFrame({
    'mass': np.random.uniform(0.1, 10, n_planets),  # Earth masses
    'orbital_period': np.random.uniform(1, 1000, n_planets),  # Earth days
    'star_type': np.random.choice(['G', 'K', 'M'], n_planets)
})

# Create a scatter plot with logarithmic scales
sns.scatterplot(data=planet_data, x='mass', y='orbital_period', hue='star_type', alpha=0.7)
plt.xscale('log')
plt.yscale('log')
plt.title('Exoplanet Mass vs. Orbital Period')
plt.xlabel('Planet Mass (Earth masses)')
plt.ylabel('Orbital Period (Earth days)')
plt.show()
```

Trang trình bày 12: Nâng cao tùy chỉnh

Seaborn cho phép tùy chỉnh nâng cao các ô, bao gồm kết hợp nhiều loại ô và điều chỉnh các thông số khác nhau.

```python
# Create a complex plot combining multiple Seaborn features
g = sns.JointGrid(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species")
g.plot_joint(sns.scatterplot)
g.plot_marginals(sns.kdeplot)
g.add_legend()
plt.suptitle("Bill Length vs. Depth with Marginal Distributions", y=1.02)
plt.tight_layout()
plt.show()
```

Slide 13: Seaborn with Time Series Data

Seaborn can be used effectively with time series data, providing insights into trends and patterns over time.

```python
# Generate sample time series data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
ts_data = pd.DataFrame({
    'date': dates,
    'value': np.cumsum(np.random.randn(len(dates))) + 100
})

# Create a time series plot
sns.lineplot(data=ts_data, x='date', y='value')
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Slide 14: Kết hợp Seaborn với Matplotlib

Mặc dù Seaborn cung cấp cấp độ hiển thị sơ đồ vẽ chức năng cao nhưng bạn vẫn có thể sử dụng Matplotlib để kiểm soát chi tiết các hình ảnh trực quan của mình.

```python
# Create a Seaborn plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x="flipper_length_mm", y="body_mass_g", hue="species", ax=ax)

# Add Matplotlib customizations
ax.set_title("Penguin Flipper Length vs. Body Mass", fontsize=16)
ax.set_xlabel("Flipper Length (mm)", fontsize=12)
ax.set_ylabel("Body Mass (g)", fontsize=12)
ax.legend(title="Species", title_fontsize=12)
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Để khám phá thêm về Seaborn và kỹ thuật trực quan hóa dữ liệu, hãy xem xét các tài nguyên sau:

1. Tài liệu chính thức của Seaborn: [https://seaborn.pydata.org/](https://seaborn.pydata.org/)
2. "Trực quan hóa dữ liệu: Giới thiệu thực tế" của Kieran Healy
3. Bài viết ArXiv: "Trực hóa dữ liệu chiều cao bằng t-SNE" của L.J.P. van der Maaten và G.E. Hinton ([https://arxiv.org/abs/1307.1662](https://arxiv.org/abs/1307.1662))
4. Hướng dẫn quan sát dữ liệu trực tiếp của Kaggle
5. Buổi nói chuyện và hội thảo về PyData (có trên YouTube)

Tài nguyên này cung cấp các giải pháp chuyên sâu, nâng cao kỹ thuật và các ứng dụng trực quan hóa dữ liệu trong thế giới thực bằng Seaborn và các thư viện Python khác.
