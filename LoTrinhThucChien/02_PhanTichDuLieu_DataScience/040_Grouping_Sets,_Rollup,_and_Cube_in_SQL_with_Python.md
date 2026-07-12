## Nhóm các bộ, cuộn và khối trong SQL bằng Python
Trang trình bày 1: Giới thiệu về Grouping Sets, Rollup và Cube trong SQL

Nhóm nhóm, tập hợp và khối là các phần mở rộng SQL mạnh mẽ cho phép tạo ra nhiều tổ hợp nhóm linh hoạt và hiệu quả trong một truy vấn duy nhất. Những tính năng này đặc biệt hữu ích để tạo báo cáo tóm tắt và thực hiện phân tích dữ liệu đa chiều.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Create a sample dataset
data = {
    'Region': ['North', 'North', 'South', 'South', 'East', 'East', 'West', 'West'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'Sales': [100, 150, 200, 250, 300, 350, 400, 450]
}
df = pd.DataFrame(data)

# Display the dataset
print(df)
```

Slide 2: Cơ bản về nhóm nhóm

Nhóm nhóm cho phép bạn chỉ định nhiều mệnh đề nhóm trong một câu lệnh GROUP BY. Tính năng này kết hợp các mức tổng hợp khác nhau, tạo ra tập kết quả bao gồm tổng phụ và tổng cuối.

```python
import pandas as pd

# Sample data
data = {
    'Region': ['North', 'North', 'South', 'South'],
    'Product': ['A', 'B', 'A', 'B'],
    'Sales': [100, 150, 200, 250]
}
df = pd.DataFrame(data)

# Simulate GROUPING SETS
result = pd.concat([
    df.groupby('Region')['Sales'].sum().reset_index(),
    df.groupby('Product')['Sales'].sum().reset_index(),
    pd.DataFrame({'Sales': [df['Sales'].sum()]})
])

print(result)
```

Trang trình bày 3: Điều khoản ROLLUP

ROLLUP tạo ra một tập hợp kết quả với nhiều cấp độ tổng phụ, chuyển từ cấp độ chi tiết nhất đến cấp độ tổng cộng. Nó đặc biệt hữu ích cho việc tóm tắt dữ liệu theo thứ bậc.

```python
import pandas as pd

# Sample data
data = {
    'Year': [2022, 2022, 2023, 2023],
    'Quarter': [1, 2, 1, 2],
    'Sales': [1000, 1200, 1100, 1300]
}
df = pd.DataFrame(data)

# Simulate ROLLUP
result = pd.concat([
    df.groupby(['Year', 'Quarter'])['Sales'].sum().reset_index(),
    df.groupby('Year')['Sales'].sum().reset_index(),
    pd.DataFrame({'Sales': [df['Sales'].sum()]})
])

print(result)
```

Slide 4: Điều khoản CUBE

CUBE tạo ra một tập kết quả với tất cả các kết hợp có thể có của các thứ nguyên đã chỉ định. Nó cung cấp một bảng chéo hoàn chỉnh của tất cả các thứ nguyên trong truy vấn.

```python
import pandas as pd
import itertools

# Sample data
data = {
    'Region': ['North', 'North', 'South', 'South'],
    'Product': ['A', 'B', 'A', 'B'],
    'Sales': [100, 150, 200, 250]
}
df = pd.DataFrame(data)

# Simulate CUBE
dimensions = ['Region', 'Product']
combinations = list(itertools.chain.from_iterable(
    itertools.combinations(dimensions, r) for r in range(len(dimensions) + 1)
))

result = pd.concat([
    df.groupby(list(combo))['Sales'].sum().reset_index() for combo in combinations
])

print(result)
```

Slide 5: So sánh các bộ nhóm, ROLLUP và CUBE

Ba tính năng SQL này cung cấp các mức độ tổng hợp và tính linh hoạt khác nhau:

* Nhóm nhóm: Kết hợp kích thước tùy chỉnh
* ROLLUP: Tóm tắt theo thứ bậc
* CUBE: Tất cả các kết hợp có thể

```python
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {
    'Region': ['North', 'North', 'South', 'South'],
    'Product': ['A', 'B', 'A', 'B'],
    'Sales': [100, 150, 200, 250]
}
df = pd.DataFrame(data)

# Simulate different grouping operations
grouping_sets = pd.concat([
    df.groupby('Region')['Sales'].sum(),
    df.groupby('Product')['Sales'].sum(),
    pd.Series([df['Sales'].sum()], index=['Total'])
])

rollup = pd.concat([
    df.groupby(['Region', 'Product'])['Sales'].sum(),
    df.groupby('Region')['Sales'].sum(),
    pd.Series([df['Sales'].sum()], index=['Total'])
])

cube = pd.concat([
    df.groupby(['Region', 'Product'])['Sales'].sum(),
    df.groupby('Region')['Sales'].sum(),
    df.groupby('Product')['Sales'].sum(),
    pd.Series([df['Sales'].sum()], index=['Total'])
])

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
grouping_sets.plot(kind='bar', ax=ax1, title='Grouping Sets')
rollup.plot(kind='bar', ax=ax2, title='ROLLUP')
cube.plot(kind='bar', ax=ax3, title='CUBE')
plt.tight_layout()
plt.show()
```

Slide 6: Ví dụ thực tế: Phân tích doanh số

Hãy phân tích tập dữ liệu về doanh số bán sản phẩm ở các khu vực và khoảng thời gian khác nhau bằng cách sử dụng Nhóm nhóm, ROLLUP và CUBE.

```python
import pandas as pd

# Create a sample sales dataset
data = {
    'Year': [2022, 2022, 2022, 2022, 2023, 2023, 2023, 2023],
    'Region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West'],
    'Product': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'Sales': [100, 150, 200, 250, 300, 350, 400, 450]
}
df = pd.DataFrame(data)

# Display the dataset
print(df)

# Simulate GROUPING SETS
grouping_sets = pd.concat([
    df.groupby(['Year', 'Region'])['Sales'].sum().reset_index(),
    df.groupby(['Year', 'Product'])['Sales'].sum().reset_index(),
    df.groupby('Year')['Sales'].sum().reset_index(),
    pd.DataFrame({'Sales': [df['Sales'].sum()]})
])

print("\nGrouping Sets Result:")
print(grouping_sets)
```

Trang trình bày 7: Ví dụ ROLLUP: Tóm tắt bán hàng theo cấp bậc

Sử dụng ROLLUP để tạo bản tóm tắt phân cấp dữ liệu bán hàng.

```python
import pandas as pd

# Using the same dataset from the previous slide

# Simulate ROLLUP
rollup = pd.concat([
    df.groupby(['Year', 'Region', 'Product'])['Sales'].sum().reset_index(),
    df.groupby(['Year', 'Region'])['Sales'].sum().reset_index(),
    df.groupby('Year')['Sales'].sum().reset_index(),
    pd.DataFrame({'Sales': [df['Sales'].sum()]})
])

print("ROLLUP Result:")
print(rollup)

# Visualize the hierarchical structure
import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph()
G.add_edge("Total", "2022")
G.add_edge("Total", "2023")
G.add_edge("2022", "North 2022")
G.add_edge("2022", "South 2022")
G.add_edge("2023", "North 2023")
G.add_edge("2023", "South 2023")

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, arrows=True)
plt.title("ROLLUP Hierarchical Structure")
plt.axis('off')
plt.show()
```

Slide 8: CUBE Ví dụ: Phân tích đa chiều

Sử dụng CUBE để thực hiện phân tích đa chiều dữ liệu bán hàng.

```python
import pandas as pd
import itertools

# Using the same dataset from the previous slides

# Simulate CUBE
dimensions = ['Year', 'Region', 'Product']
combinations = list(itertools.chain.from_iterable(
    itertools.combinations(dimensions, r) for r in range(len(dimensions) + 1)
))

cube = pd.concat([
    df.groupby(list(combo))['Sales'].sum().reset_index() for combo in combinations
])

print("CUBE Result:")
print(cube)

# Visualize the cube structure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = [0, 1, 0, 1]
y = [0, 0, 1, 1]
z = [0, 0, 0, 0]

ax.scatter(x, y, z, c='r', s=100)
ax.plot([0, 1], [0, 0], [0, 0], 'b')
ax.plot([0, 0], [0, 1], [0, 0], 'b')
ax.plot([1, 1], [0, 1], [0, 0], 'b')
ax.plot([0, 1], [1, 1], [0, 0], 'b')

ax.set_xlabel('Year')
ax.set_ylabel('Region')
ax.set_zlabel('Product')
ax.set_title('CUBE Structure Visualization')

plt.show()
```

Trang trình bày 9: Cân nhắc về hiệu suất

Khi sử dụng Nhóm nhóm, ROLLUP và CUBE, hãy xem xét các khía cạnh hiệu suất sau:

1. Khối lượng dữ liệu: Các thao tác này có thể tạo ra các tập kết quả lớn, đặc biệt là CUBE.
2. Lập chỉ mục: Lập chỉ mục thích hợp trên các cột được nhóm có thể cải thiện đáng kể hiệu suất.
3. Chế độ xem cụ thể hóa: Đối với các nhóm được sử dụng thường xuyên, hãy cân nhắc sử dụng chế độ xem cụ thể hóa.

```python
import time
import pandas as pd
import numpy as np

# Generate a larger dataset
np.random.seed(0)
n = 1000000
data = pd.DataFrame({
    'Year': np.random.choice([2022, 2023], n),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], n),
    'Product': np.random.choice(['A', 'B', 'C', 'D'], n),
    'Sales': np.random.randint(100, 1000, n)
})

# Measure execution time for different operations
def measure_time(func):
    start = time.time()
    func()
    end = time.time()
    return end - start

grouping_sets_time = measure_time(lambda: data.groupby(['Year', 'Region'])['Sales'].sum())
rollup_time = measure_time(lambda: data.groupby(['Year', 'Region', 'Product'])['Sales'].sum())
cube_time = measure_time(lambda: data.groupby(['Year', 'Region', 'Product'])['Sales'].sum())

print(f"Grouping Sets time: {grouping_sets_time:.2f} seconds")
print(f"ROLLUP time: {rollup_time:.2f} seconds")
print(f"CUBE time: {cube_time:.2f} seconds")

# Plot execution times
import matplotlib.pyplot as plt

operations = ['Grouping Sets', 'ROLLUP', 'CUBE']
times = [grouping_sets_time, rollup_time, cube_time]

plt.figure(figsize=(10, 6))
plt.bar(operations, times)
plt.title('Execution Time Comparison')
plt.ylabel('Time (seconds)')
plt.show()
```

Trang trình chiếu 10: Ví dụ thực tế: Phân tích dữ liệu thời tiết

Phân tích dữ liệu nhiệt độ trên các vị trí và khoảng thời gian khác nhau bằng cách sử dụng Nhóm nhóm, ROLLUP và CUBE.

```python
import pandas as pd
import numpy as np

# Generate sample weather data
np.random.seed(0)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
locations = ['City A', 'City B', 'City C']
data = []

for date in dates:
    for location in locations:
        temp = np.random.normal(loc=20, scale=5)
        data.append([date, location, temp])

df = pd.DataFrame(data, columns=['Date', 'Location', 'Temperature'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Grouping Sets: Average temperature by year and location
grouping_sets = df.groupby(['Year', 'Location'])['Temperature'].mean().reset_index()
print("Grouping Sets Result:")
print(grouping_sets)

# ROLLUP: Hierarchical summary of temperatures
rollup = pd.concat([
    df.groupby(['Year', 'Month', 'Location'])['Temperature'].mean(),
    df.groupby(['Year', 'Month'])['Temperature'].mean(),
    df.groupby('Year')['Temperature'].mean(),
    pd.Series([df['Temperature'].mean()], index=['Overall'])
]).reset_index()
print("\nROLLUP Result:")
print(rollup)

# CUBE: Multi-dimensional analysis
cube = pd.concat([
    df.groupby(['Year', 'Month', 'Location'])['Temperature'].mean(),
    df.groupby(['Year', 'Month'])['Temperature'].mean(),
    df.groupby(['Year', 'Location'])['Temperature'].mean(),
    df.groupby(['Month', 'Location'])['Temperature'].mean(),
    df.groupby('Year')['Temperature'].mean(),
    df.groupby('Month')['Temperature'].mean(),
    df.groupby('Location')['Temperature'].mean(),
    pd.Series([df['Temperature'].mean()], index=['Overall'])
]).reset_index()
print("\nCUBE Result:")
print(cube)

# Visualize average temperatures by location
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
for location in locations:
    data = df[df['Location'] == location]
    plt.plot(data['Date'], data['Temperature'], label=location)

plt.title('Temperature Trends by Location')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
```

Slide 11: Ví dụ thực tế: Phân tích sản phẩm thương mại điện tử

Phân tích dữ liệu sản phẩm theo các danh mục và khoảng thời gian khác nhau bằng cách sử dụng Nhóm nhóm, ROLLUP và CUBE trong bối cảnh thương mại điện tử.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate sample e-commerce data
np.random.seed(0)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
categories = ['Electronics', 'Clothing', 'Home & Garden']
products = ['Product A', 'Product B', 'Product C']
data = []

for date in dates:
    for category in categories:
        for product in products:
            sales = np.random.randint(10, 100)
            data.append([date, category, product, sales])

df = pd.DataFrame(data, columns=['Date', 'Category', 'Product', 'Sales'])
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Grouping Sets: Total sales by category and product
grouping_sets = df.groupby(['Category', 'Product'])['Sales'].sum().reset_index()
print("Grouping Sets Result:")
print(grouping_sets.head(10))

# ROLLUP: Hierarchical summary of sales
rollup = pd.concat([
    df.groupby(['Year', 'Month', 'Category'])['Sales'].sum(),
    df.groupby(['Year', 'Month'])['Sales'].sum(),
    df.groupby('Year')['Sales'].sum(),
    pd.Series([df['Sales'].sum()], index=['Overall'])
]).reset_index()
print("\nROLLUP Result:")
print(rollup.head(10))

# CUBE: Multi-dimensional analysis
cube = pd.concat([
    df.groupby(['Year', 'Month', 'Category', 'Product'])['Sales'].sum(),
    df.groupby(['Year', 'Month', 'Category'])['Sales'].sum(),
    df.groupby(['Year', 'Month'])['Sales'].sum(),
    df.groupby(['Year', 'Category'])['Sales'].sum(),
    df.groupby('Year')['Sales'].sum(),
    pd.Series([df['Sales'].sum()], index=['Overall'])
]).reset_index()
print("\nCUBE Result:")
print(cube.head(10))

# Visualize sales trends
plt.figure(figsize=(12, 6))
for category in categories:
    category_data = df[df['Category'] == category].groupby('Date')['Sales'].sum()
    plt.plot(category_data.index, category_data.values, label=category)

plt.title('Sales Trends by Category')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.show()
```

Slide 12: Kỹ thuật nâng cao: Kết hợp Grouping Sets, ROLLUP và CUBE

Trong các tình huống phức tạp, bạn có thể kết hợp các tính năng này để tạo tập hợp kết quả có tính tùy chỉnh cao. Điều này đặc biệt hữu ích khi xử lý phân tích dữ liệu đa chiều.

```python
import pandas as pd
import numpy as np

# Using the e-commerce dataset from the previous slide

# Combined GROUPING SETS, ROLLUP, and CUBE
combined_analysis = pd.concat([
    # GROUPING SETS
    df.groupby(['Category', 'Product'])['Sales'].sum(),

    # ROLLUP
    df.groupby(['Year', 'Month', 'Category'])['Sales'].sum(),
    df.groupby(['Year', 'Month'])['Sales'].sum(),
    df.groupby('Year')['Sales'].sum(),

    # CUBE
    df.groupby(['Year', 'Category', 'Product'])['Sales'].sum(),
    df.groupby(['Year', 'Category'])['Sales'].sum(),
    df.groupby(['Category', 'Product'])['Sales'].sum(),

    # Overall total
    pd.Series([df['Sales'].sum()], index=['Overall'])
]).reset_index()

print("Combined Analysis Result:")
print(combined_analysis.head(15))

# Visualize the multi-level aggregation
plt.figure(figsize=(12, 8))
combined_analysis.groupby('Year')['Sales'].sum().plot(kind='bar', position=1, width=0.2, color='blue', label='Year')
combined_analysis.groupby('Category')['Sales'].sum().plot(kind='bar', position=0, width=0.2, color='green', label='Category')
combined_analysis.groupby('Product')['Sales'].sum().plot(kind='bar', position=2, width=0.2, color='red', label='Product')

plt.title('Multi-level Sales Aggregation')
plt.xlabel('Grouping Level')
plt.ylabel('Total Sales')
plt.legend()
plt.tight_layout()
plt.show()
```

Trang trình bày 13: Các phương pháp hay nhất và kỹ thuật tối ưu hóa

Khi làm việc với Nhóm nhóm, ROLLUP và CUBE, hãy xem xét các phương pháp hay nhất sau:

1. Sử dụng cách đánh chỉ mục phù hợp trên các cột được nhóm
2. Giới hạn số lượng kích thước để tránh tăng kích thước tập kết quả theo cấp số nhân
3. Cân nhắc sử dụng các chế độ xem cụ thể hóa cho các tập hợp được truy cập thường xuyên
4. Theo dõi hiệu suất truy vấn và tối ưu hóa khi cần

```python
import pandas as pd
import numpy as np
import time

# Generate a larger dataset for performance testing
np.random.seed(0)
n = 1000000
large_df = pd.DataFrame({
    'Year': np.random.choice([2022, 2023], n),
    'Month': np.random.randint(1, 13, n),
    'Category': np.random.choice(['A', 'B', 'C'], n),
    'Product': np.random.choice(['X', 'Y', 'Z'], n),
    'Sales': np.random.randint(100, 1000, n)
})

# Function to measure execution time
def measure_time(func):
    start = time.time()
    result = func()
    end = time.time()
    return result, end - start

# Test different grouping operations
grouping_sets, gs_time = measure_time(lambda: large_df.groupby(['Category', 'Product'])['Sales'].sum())
rollup, rollup_time = measure_time(lambda: pd.concat([
    large_df.groupby(['Year', 'Month', 'Category'])['Sales'].sum(),
    large_df.groupby(['Year', 'Month'])['Sales'].sum(),
    large_df.groupby('Year')['Sales'].sum()
]))
cube, cube_time = measure_time(lambda: pd.concat([
    large_df.groupby(['Year', 'Month', 'Category', 'Product'])['Sales'].sum(),
    large_df.groupby(['Year', 'Month', 'Category'])['Sales'].sum(),
    large_df.groupby(['Year', 'Month'])['Sales'].sum(),
    large_df.groupby('Year')['Sales'].sum()
]))

print(f"Grouping Sets execution time: {gs_time:.2f} seconds")
print(f"ROLLUP execution time: {rollup_time:.2f} seconds")
print(f"CUBE execution time: {cube_time:.2f} seconds")

# Visualize performance comparison
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(['Grouping Sets', 'ROLLUP', 'CUBE'], [gs_time, rollup_time, cube_time])
plt.title('Performance Comparison')
plt.ylabel('Execution Time (seconds)')
plt.show()
```

Slide 14: Kết luận và xu hướng tương lai

Nhóm nhóm, ROLLUP và CUBE là các tính năng SQL mạnh mẽ cho phép phân tích dữ liệu đa chiều hiệu quả. Khi khối lượng dữ liệu tiếp tục tăng và nhu cầu kinh doanh thông minh trở nên phức tạp hơn, những công cụ này sẽ đóng vai trò ngày càng quan trọng trong phân tích và báo cáo dữ liệu.

Xu hướng trong tương lai có thể bao gồm:

1. Tích hợp với quy trình máy học
2. Công cụ trực quan nâng cao cho dữ liệu đa chiều
3. Tối ưu hóa kho dữ liệu trên nền tảng đám mây
4. Tích hợp vào hệ thống phân tích thời gian thực

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulating future adoption trends
years = np.arange(2020, 2031)
grouping_sets_adoption = np.cumsum(np.random.normal(10, 2, len(years)))
rollup_adoption = np.cumsum(np.random.normal(8, 2, len(years)))
cube_adoption = np.cumsum(np.random.normal(6, 2, len(years)))

plt.figure(figsize=(12, 6))
plt.plot(years, grouping_sets_adoption, label='Grouping Sets')
plt.plot(years, rollup_adoption, label='ROLLUP')
plt.plot(years, cube_adoption, label='CUBE')

plt.title('Projected Adoption Trends')
plt.xlabel('Year')
plt.ylabel('Cumulative Adoption (arbitrary units)')
plt.legend()
plt.grid(True)
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Để khám phá thêm về Nhóm nhóm, ROLLUP và CUBE trong SQL:

1. Hiệu suất SQL được giải thích bởi Markus Winand Tham khảo: arXiv:1508.03474 \[cs.DB\]
2. Các kỹ thuật SQL nâng cao để phân tích dữ liệu Tham khảo: arXiv:1907.04346 \[cs.DB\]
3. Hoạt động OLAP hiệu quả trong SQL Tham khảo: arXiv:2003.01793 \[cs.DB\]

Các tài nguyên này cung cấp các cuộc thảo luận chuyên sâu về các kỹ thuật SQL nâng cao, bao gồm Nhóm nhóm, ROLLUP và CUBE, cùng với các chiến lược tối ưu hóa hiệu suất và các ứng dụng trong thế giới thực trong phân tích dữ liệu.
