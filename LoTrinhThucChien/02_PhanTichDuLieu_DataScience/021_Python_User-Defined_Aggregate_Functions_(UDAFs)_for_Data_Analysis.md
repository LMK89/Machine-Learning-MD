## Hàm tổng hợp do người dùng xác định (UDAF) Python để phân tích dữ liệu

Trang trình bày 1: Tìm hiểu về các hàm tổng hợp do người dùng xác định (UDAF) trong Python

Hàm tổng hợp do người dùng xác định (UDAF) trong Python cho phép các nhà phát triển tạo các hàm tùy chỉnh tóm tắt dữ liệu theo nhu cầu cụ thể. Các chức năng này vượt ra ngoài các tổng hợp tích hợp như tổng hoặc đếm, cho phép phân tích dữ liệu linh hoạt và phù hợp hơn.

```python
def custom_aggregate(data):
    return sum(data) / len(data) if data else 0

# Using the custom aggregate
numbers = [1, 2, 3, 4, 5]
result = custom_aggregate(numbers)
print(f"Custom aggregate result: {result}")
```

Trang trình bày 2: Tạo UDAF cơ bản: Trung bình có trọng số

Hãy tạo UDAF để tính giá trị trung bình có trọng số của tập dữ liệu. Hàm này có hai danh sách: một danh sách cho các giá trị và một danh sách khác cho trọng số tương ứng của chúng.

```python
    if len(values) != len(weights):
        raise ValueError("Values and weights must have the same length")
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)

# Example usage
values = [80, 90, 95]
weights = [0.3, 0.4, 0.3]
result = weighted_average(values, weights)
print(f"Weighted average: {result}")
```

Trang trình bày 3: Triển khai UDAF theo cấu trúc theo lớp

Đối với các UDAF phức tạp hơn, cấu trúc dựa trên lớp có thể có lợi. Cách tiếp cận này cho phép duy trì trạng thái giữa các lệnh gọi hàm và cung cấp một tổ chức rõ ràng hơn về logic của hàm tổng hợp.

```python
    def __init__(self):
        self.count = 0
        self.total = 0

    def step(self, value):
        self.count += 1
        self.total += value

    def finalize(self):
        return self.total / self.count if self.count > 0 else 0

# Usage
ra = RunningAverage()
for num in [1, 2, 3, 4, 5]:
    ra.step(num)
result = ra.finalize()
print(f"Running average: {result}")
```

Slide 4: UDAF để tính toán chế độ

Hãy tạo UDAF để tìm chế độ (giá trị thường xuyên nhất) trong tập dữ liệu. Ví dụ này minh họa việc xử lý logic phức tạp hơn trong hàm tổng hợp tùy chỉnh.

```python

def mode(data):
    if not data:
        return None
    counter = Counter(data)
    max_count = max(counter.values())
    modes = [k for k, v in counter.items() if v == max_count]
    return modes[0] if len(modes) == 1 else modes

# Example usage
numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
result = mode(numbers)
print(f"Mode: {result}")
```

Trang trình bày 5: UDAF để chuẩn hóa dữ liệu

UDAF này chuẩn hóa tập dữ liệu bằng cách chia tỷ lệ các giá trị thành phạm vi từ 0 đến 1. Nó hữu ích trong nhiều tình huống tiền xử lý dữ liệu khác nhau.

```python
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def step(self, value):
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    def finalize(self, data):
        range_val = self.max_val - self.min_val
        return [(x - self.min_val) / range_val for x in data] if range_val != 0 else [0] * len(data)

# Usage
normalizer = Normalizer()
data = [10, 20, 30, 40, 50]
for value in data:
    normalizer.step(value)
normalized_data = normalizer.finalize(data)
print(f"Normalized data: {normalized_data}")
```

Trang trình bày 6: UDAF cho đường trung bình động

Việc triển khai UDAF trung bình động có thể hữu ích để làm mịn dữ liệu chuỗi thời gian hoặc xác định xu hướng theo thời gian.

```python

class MovingAverage:
    def __init__(self, window_size):
        self.window = deque(maxlen=window_size)

    def step(self, value):
        self.window.append(value)

    def finalize(self):
        return sum(self.window) / len(self.window) if self.window else 0

# Usage
ma = MovingAverage(window_size=3)
data = [1, 3, 5, 2, 8, 4, 6]
moving_averages = []
for value in data:
    ma.step(value)
    moving_averages.append(ma.finalize())
print(f"Moving averages: {moving_averages}")
```

Slide 7: UDAF để tính toán phương sai

Việc tạo UDAF để tính toán phương sai thể hiện cách xử lý các tập hợp nhiều lượt trong đó chúng ta cần tính toán các kết quả trung gian.

```python

class VarianceCalculator:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.sum_sq = 0

    def step(self, value):
        self.count += 1
        self.sum += value
        self.sum_sq += value ** 2

    def finalize(self):
        if self.count < 2:
            return 0
        mean = self.sum / self.count
        return (self.sum_sq / self.count) - (mean ** 2)

# Usage
vc = VarianceCalculator()
data = [2, 4, 4, 4, 5, 5, 7, 9]
for value in data:
    vc.step(value)
variance = vc.finalize()
std_dev = math.sqrt(variance)
print(f"Variance: {variance}")
print(f"Standard Deviation: {std_dev}")
```

Trang trình bày 8: UDAF để tính phần trăm

UDAF này tính toán phần trăm được chỉ định của tập dữ liệu, rất hữu ích để hiểu phân phối dữ liệu và xác định các giá trị ngoại lệ.

```python

class PercentileCalculator:
    def __init__(self, percentile):
        self.percentile = percentile
        self.values = []

    def step(self, value):
        self.values.append(value)

    def finalize(self):
        if not self.values:
            return None
        sorted_values = sorted(self.values)
        index = (len(self.values) - 1) * self.percentile / 100
        lower_index = math.floor(index)
        upper_index = math.ceil(index)
        if lower_index == upper_index:
            return sorted_values[int(index)]
        lower_value = sorted_values[lower_index]
        upper_value = sorted_values[upper_index]
        return lower_value + (upper_value - lower_value) * (index - lower_index)

# Usage
pc = PercentileCalculator(percentile=75)
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for value in data:
    pc.step(value)
result = pc.finalize()
print(f"75th percentile: {result}")
```

Trang trình bày 9: UDAF để nối chuỗi

UDAF này trình bày cách áp dụng tổng hợp tùy chỉnh cho dữ liệu không phải là số, chẳng hạn như nối chuỗi bằng dấu phân cách tùy chỉnh.

```python
    def __init__(self, separator=', '):
        self.separator = separator
        self.strings = []

    def step(self, value):
        self.strings.append(str(value))

    def finalize(self):
        return self.separator.join(self.strings)

# Usage
sc = StringConcatenator(separator=' | ')
words = ['Python', 'is', 'awesome', 'for', 'data', 'analysis']
for word in words:
    sc.step(word)
result = sc.finalize()
print(f"Concatenated string: {result}")
```

Trang trình bày 10: Ví dụ thực tế: UDAF cho phân tích dữ liệu môi trường

Giả sử chúng ta đang phân tích dữ liệu nhiệt độ từ nhiều trạm thời tiết khác nhau. Chúng tôi muốn tạo UDAF để tính toán phạm vi nhiệt độ hàng ngày (chênh lệch giữa nhiệt độ tối đa và nhiệt độ tối thiểu) và gắn cờ những ngày có mức chênh lệch lớn.

```python
    def __init__(self, extreme_threshold):
        self.extreme_threshold = extreme_threshold
        self.daily_min = float('inf')
        self.daily_max = float('-inf')

    def step(self, temperature):
        self.daily_min = min(self.daily_min, temperature)
        self.daily_max = max(self.daily_max, temperature)

    def finalize(self):
        temp_range = self.daily_max - self.daily_min
        is_extreme = temp_range > self.extreme_threshold
        return {
            'min_temp': self.daily_min,
            'max_temp': self.daily_max,
            'temp_range': temp_range,
            'is_extreme': is_extreme
        }

# Usage
tra = TemperatureRangeAnalyzer(extreme_threshold=20)
daily_temperatures = [15, 18, 22, 25, 30, 28, 20]
for temp in daily_temperatures:
    tra.step(temp)
result = tra.finalize()
print(f"Temperature analysis: {result}")
```

Trang trình bày 11: Ví dụ thực tế: UDAF cho phân tích cảm xúc văn bản

UDAF này thực hiện phân tích cảm xúc đơn giản trên dữ liệu văn bản, đếm các từ tích cực và tiêu cực để xác định cảm xúc tổng thể.

```python
    def __init__(self):
        self.positive_words = set(['good', 'great', 'excellent', 'amazing', 'wonderful'])
        self.negative_words = set(['bad', 'poor', 'terrible', 'awful', 'horrible'])
        self.positive_count = 0
        self.negative_count = 0
        self.total_words = 0

    def step(self, text):
        words = text.lower().split()
        self.total_words += len(words)
        self.positive_count += sum(1 for word in words if word in self.positive_words)
        self.negative_count += sum(1 for word in words if word in self.negative_words)

    def finalize(self):
        if self.total_words == 0:
            return 'Neutral'
        sentiment_score = (self.positive_count - self.negative_count) / self.total_words
        if sentiment_score > 0.05:
            return 'Positive'
        elif sentiment_score < -0.05:
            return 'Negative'
        else:
            return 'Neutral'

# Usage
sa = SentimentAnalyzer()
reviews = [
    "This product is amazing and works great!",
    "I had a terrible experience with customer service.",
    "The quality is good but the price is a bit high."
]
for review in reviews:
    sa.step(review)
overall_sentiment = sa.finalize()
print(f"Overall sentiment: {overall_sentiment}")
```

Trang trình bày 12: Tối ưu hóa UDAF cho tập dữ liệu lớn

Khi làm việc với các tập dữ liệu lớn, điều quan trọng là phải tối ưu hóa UDAF để đạt hiệu quả và hiệu suất bộ nhớ. Dưới đây là ví dụ về UDAF tiết kiệm bộ nhớ để tính giá trị trung bình của một tập dữ liệu lớn.

```python

class MedianCalculator:
    def __init__(self):
        self.smaller = []  # max heap
        self.larger = []   # min heap

    def step(self, value):
        if len(self.smaller) == len(self.larger):
            heapq.heappush(self.larger, -heapq.heappushpop(self.smaller, -value))
        else:
            heapq.heappush(self.smaller, -heapq.heappushpop(self.larger, value))

    def finalize(self):
        if len(self.smaller) == len(self.larger):
            return (-self.smaller[0] + self.larger[0]) / 2
        else:
            return self.larger[0]

# Usage
mc = MedianCalculator()
large_dataset = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
for value in large_dataset:
    mc.step(value)
median = mc.finalize()
print(f"Median of the dataset: {median}")
```

Slide 13: Kết hợp nhiều UDAF

Trong các tình huống thực tế, bạn có thể cần kết hợp nhiều UDAF để thực hiện các phân tích phức tạp. Dưới đây là ví dụ kết hợp một số UDAF để phân tích tập dữ liệu đánh giá sản phẩm.

```python
    def __init__(self):
        self.total_reviews = 0
        self.total_rating = 0
        self.word_count = 0
        self.positive_words = set(['good', 'great', 'excellent', 'amazing'])
        self.negative_words = set(['bad', 'poor', 'terrible', 'awful'])
        self.sentiment_score = 0

    def step(self, review, rating):
        self.total_reviews += 1
        self.total_rating += rating
        words = review.lower().split()
        self.word_count += len(words)
        self.sentiment_score += sum(1 for word in words if word in self.positive_words)
        self.sentiment_score -= sum(1 for word in words if word in self.negative_words)

    def finalize(self):
        avg_rating = self.total_rating / self.total_reviews if self.total_reviews > 0 else 0
        avg_word_count = self.word_count / self.total_reviews if self.total_reviews > 0 else 0
        overall_sentiment = 'Positive' if self.sentiment_score > 0 else 'Negative' if self.sentiment_score < 0 else 'Neutral'
        return {
            'total_reviews': self.total_reviews,
            'average_rating': avg_rating,
            'average_word_count': avg_word_count,
            'overall_sentiment': overall_sentiment
        }

# Usage
ra = ReviewAnalyzer()
reviews = [
    ("This product is amazing!", 5),
    ("Not worth the money, terrible quality.", 1),
    ("Good product, but a bit overpriced.", 3)
]
for review, rating in reviews:
    ra.step(review, rating)
analysis_result = ra.finalize()
print(f"Review analysis: {analysis_result}")
```

Trang trình bày 14: Các phương pháp hay nhất để tạo UDAF

Khi tạo UDAF, hãy xem xét các phương pháp hay nhất sau:

1. Đảm bảo hiệu quả bộ nhớ, đặc biệt đối với các tập dữ liệu lớn.
2. Thực hiện xử lý lỗi rõ ràng và xác thực đầu vào.
3. Sử dụng tên mô tả cho hàm và biến.
4. Ghi lại UDAF của bạn một cách kỹ lưỡng, bao gồm cả đầu vào và đầu ra dự kiến.
5. Kiểm tra UDAF của bạn với nhiều trường hợp đặc biệt và bộ dữ liệu lớn.

Trang trình bày 15: Các phương pháp hay nhất để tạo UDAF

Đây là một ví dụ kết hợp các thực hành này:

```python
    """
    A UDAF that calculates robust statistics (median and IQR) for a dataset.
    """
    def __init__(self):
        self.data = []

    def step(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Input must be a number")
        self.data.append(value)

    def finalize(self):
        if not self.data:
            raise ValueError("Dataset is empty")
        sorted_data = sorted(self.data)
        n = len(sorted_data)
        median = sorted_data[n // 2] if n % 2 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        q1 = sorted_data[n // 4]
        q3 = sorted_data[3 * n // 4]
        iqr = q3 - q1
        return {"median": median, "IQR": iqr}

# Usage
calc = RobustStatCalculator()
dataset = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
for value in dataset:
    calc.step(value)
result = calc.finalize()
print(f"Robust statistics: {result}")
```

Trang trình bày 16: Tích hợp UDAF với Khung xử lý dữ liệu

UDAF có thể được tích hợp với các khung xử lý dữ liệu phổ biến như pandas hoặc PySpark để phân tích dữ liệu hiệu quả hơn. Đây là một ví dụ sử dụng gấu trúc:

```python

def custom_udaf(group):
    return pd.Series({
        'mean': group.mean(),
        'median': group.median(),
        'range': group.max() - group.min()
    })

# Sample data
data = {
    'category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'value': [10, 15, 20, 25, 30, 35]
}
df = pd.DataFrame(data)

# Apply UDAF
result = df.groupby('category')['value'].apply(custom_udaf)
print(result)
```

Trang trình bày 17: Tài nguyên bổ sung

Dành cho những người muốn tìm hiểu sâu hơn về UDAF và xử lý dữ liệu Python nâng cao:

1. "Python để phân tích dữ liệu" của Wes McKinney (O'Reilly Media)
2. "Python thông thạo" của Luciano Ramalho (O'Reilly Media)
3. "Những chú gấu trúc hiệu quả" của Matt Harrison (có sẵn trực tuyến)
4. Bài viết ArXiv: "Thuật toán tổng hợp hiệu quả cho dữ liệu xác suất" (arXiv:1703.02614)
5. PEP 450 - Thêm mô-đun thống kê vào Thư viện chuẩn (python.org/dev/peps/pep-0450/)

Những tài nguyên này cung cấp những giải thích sâu sắc và các kỹ thuật nâng cao để làm việc với dữ liệu trong Python.
