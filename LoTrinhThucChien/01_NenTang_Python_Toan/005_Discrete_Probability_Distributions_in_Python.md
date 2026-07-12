## Phân phối xác suất rời rạc trong Python
Trang trình bày 1: Giới thiệu về phân phối rời rạc

Phân phối rời rạc là phân phối xác suất mô tả các biến ngẫu nhiên với tập hợp hữu hạn hoặc vô hạn đếm được các giá trị có thể. Chúng là nền tảng trong lý thuyết thống kê và xác suất, được sử dụng để mô hình hóa các hiện tượng khác nhau trong thế giới thực trong đó các kết quả là khác biệt và riêng biệt.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate discrete data
x = np.arange(1, 7)
y = np.random.randint(1, 7, size=1000)

# Plot histogram
plt.hist(y, bins=x - 0.5, rwidth=0.8)
plt.xticks(x)
plt.xlabel('Outcome')
plt.ylabel('Frequency')
plt.title('Histogram of Discrete Data (Dice Rolls)')
plt.show()
```

Trang trình bày 2: Phân phối Bernoulli

Phân phối Bernoulli mô hình hóa một kết quả nhị phân duy nhất, chẳng hạn như thành công/thất bại hoặc có/không. Nó được đặt theo tên của Jacob Bernoulli và là phân bố xác suất rời rạc đơn giản nhất. Hàm khối lượng xác suất được xác định bởi một tham số p duy nhất, đại diện cho xác suất thành công.

```python
import numpy as np
import matplotlib.pyplot as plt

def bernoulli(p, size=1000):
    return np.random.random(size) < p

p = 0.7
results = bernoulli(p)

plt.bar(['Failure', 'Success'], [np.sum(results == 0), np.sum(results == 1)])
plt.title(f'Bernoulli Distribution (p={p})')
plt.ylabel('Count')
plt.show()
```

Trang trình bày 3: Phân phối Bernoulli - Ví dụ thực tế

Hãy xem xét một quy trình kiểm soát chất lượng trong một nhà máy sản xuất. Mỗi sản phẩm đều được kiểm tra và phân loại là bị lỗi hoặc không bị lỗi. Kịch bản này có thể được mô hình hóa bằng phân phối Bernoulli, trong đó thành công (1) đại diện cho một mặt hàng không bị lỗi và thất bại (0) đại diện cho một mặt hàng bị lỗi.

```python
def quality_control(defect_rate, num_items):
    return bernoulli(1 - defect_rate, num_items)

defect_rate = 0.05
num_items = 1000
inspection_results = quality_control(defect_rate, num_items)

print(f"Number of non-defective items: {np.sum(inspection_results)}")
print(f"Number of defective items: {num_items - np.sum(inspection_results)}")
```

Trang trình bày 4: Phân phối nhị thức

Phân phối nhị thức mô hình hóa số lần thành công trong một số thử nghiệm Bernoulli độc lập cố định. Nó được đặc trưng bởi hai tham số: n (số lần thử) và p (xác suất thành công của mỗi lần thử). Phân phối nhị thức được sử dụng rộng rãi trong nhiều lĩnh vực khác nhau, bao gồm sinh học, vật lý và khoa học xã hội.

```python
from scipy.stats import binom

n, p = 20, 0.3
x = np.arange(0, n+1)
pmf = binom.pmf(x, n, p)

plt.bar(x, pmf)
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.show()
```

Trang trình bày 5: Phân phối nhị thức - Ví dụ thực tế

Hãy tưởng tượng một trung tâm cuộc gọi nhận được trung bình 100 cuộc gọi mỗi giờ. Mỗi cuộc gọi có 20% khả năng yêu cầu báo cáo lên người giám sát. Chúng ta có thể sử dụng Phân phối nhị thức để lập mô hình số lượng cuộc gọi leo thang trong một giờ nhất định.

```python
n_calls = 100
p_escalation = 0.2

# Simulate one hour of calls
escalated_calls = np.random.binomial(n_calls, p_escalation)

print(f"Number of escalated calls in one hour: {escalated_calls}")

# Simulate multiple hours
hours = 1000
escalated_calls_per_hour = np.random.binomial(n_calls, p_escalation, hours)

plt.hist(escalated_calls_per_hour, bins=range(0, max(escalated_calls_per_hour)+2), align='left', rwidth=0.8)
plt.title('Distribution of Escalated Calls per Hour')
plt.xlabel('Number of Escalated Calls')
plt.ylabel('Frequency')
plt.show()
```

Trang trình bày 6: Phân phối hình học

Mô hình phân phối hình học số lượng phép thử Bernoulli cần thiết để đạt được thành công đầu tiên. Nó được đặc trưng bởi một tham số p duy nhất, là xác suất thành công của mỗi lần thử. Phân phối này không có bộ nhớ, nghĩa là xác suất thành công không phụ thuộc vào kết quả trước đó.

```python
from scipy.stats import geom

p = 0.3
x = np.arange(1, 15)
pmf = geom.pmf(x, p)

plt.bar(x, pmf)
plt.title(f'Geometric Distribution (p={p})')
plt.xlabel('Number of Trials until First Success')
plt.ylabel('Probability')
plt.show()
```

Slide 7: Phân bố hình học - Ứng dụng

Phân phối hình học có thể được sử dụng để mô hình hóa số lần thử cần thiết để đạt được kết quả mong muốn. Ví dụ: trong một trò chơi mà người chơi cần tung được số sáu vào một con súc sắc công bằng, số lần tung xúc xắc cho đến khi sáu con đầu tiên xuất hiện tuân theo sự phân bố Hình học.

```python
def roll_until_six():
    rolls = 0
    while True:
        rolls += 1
        if np.random.randint(1, 7) == 6:
            return rolls

# Simulate 1000 games
games = 1000
results = [roll_until_six() for _ in range(games)]

plt.hist(results, bins=range(1, max(results)+2), align='left', rwidth=0.8)
plt.title('Number of Rolls Until First Six')
plt.xlabel('Number of Rolls')
plt.ylabel('Frequency')
plt.show()

print(f"Average number of rolls: {np.mean(results):.2f}")
```

Trang trình bày 8: Phân phối Poisson

Phân phối Poisson mô hình hóa số lượng sự kiện xảy ra trong một khoảng thời gian hoặc không gian cố định, với điều kiện là những sự kiện này xảy ra với tốc độ trung bình đã biết và độc lập với nhau. Nó được đặc trưng bởi một tham số duy nhất λ (lambda), đại diện cho cả giá trị trung bình và phương sai của phân phối.

```python
from scipy.stats import poisson

lambda_param = 3
x = np.arange(0, 15)
pmf = poisson.pmf(x, lambda_param)

plt.bar(x, pmf)
plt.title(f'Poisson Distribution (λ={lambda_param})')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.show()
```

Slide 9: Phân phối Poisson - Ví dụ thực tế

Phân phối Poisson có thể mô hình hóa các hiện tượng khác nhau trong thế giới thực, chẳng hạn như số lượng khách hàng đến cửa hàng trong một giờ nhất định hoặc số lỗi chính tả trong một tài liệu có độ dài nhất định. Hãy mô phỏng số lượng trận động đất xảy ra ở khu vực có hoạt động địa chấn trong một khoảng thời gian.

```python
avg_earthquakes_per_year = 5
years = 100

earthquake_counts = np.random.poisson(avg_earthquakes_per_year, years)

plt.hist(earthquake_counts, bins=range(0, max(earthquake_counts)+2), align='left', rwidth=0.8)
plt.title('Annual Earthquake Counts over 100 Years')
plt.xlabel('Number of Earthquakes')
plt.ylabel('Frequency')
plt.show()

print(f"Average earthquakes per year: {np.mean(earthquake_counts):.2f}")
print(f"Maximum earthquakes in a year: {np.max(earthquake_counts)}")
```

Slide 10: Phân bố đồng đều (rời rạc)

Phân phối đồng nhất rời rạc ấn định xác suất bằng nhau cho một tập hợp kết quả hữu hạn. Nó được đặc trưng bởi hai tham số: a (giá trị tối thiểu) và b (giá trị tối đa). Phân phối này thường được sử dụng để mô hình hóa các tình huống trong đó mỗi kết quả đều có khả năng xảy ra như nhau, chẳng hạn như tung xúc xắc công bằng.

```python
from scipy.stats import randint

a, b = 1, 6  # Min and max values for a die
x = np.arange(a, b+1)
pmf = randint.pmf(x, a, b+1)

plt.bar(x, pmf)
plt.title(f'Discrete Uniform Distribution (a={a}, b={b})')
plt.xlabel('Outcome')
plt.ylabel('Probability')
plt.xticks(x)
plt.show()
```

Slide 11: Phân phối thống nhất - Ứng dụng

Phân phối Đồng nhất rời rạc có thể được sử dụng để mô hình hóa các tình huống khác nhau trong đó kết quả có khả năng xảy ra như nhau. Hãy mô phỏng một trò chơi đơn giản trong đó người chơi thắng nếu họ đoán đúng một số được chọn ngẫu nhiên trong khoảng từ 1 đến 10.

```python
def play_guessing_game(num_games):
    wins = 0
    for _ in range(num_games):
        secret_number = np.random.randint(1, 11)
        guess = np.random.randint(1, 11)  # Simulate a random guess
        if guess == secret_number:
            wins += 1
    return wins

num_games = 1000
wins = play_guessing_game(num_games)

print(f"Number of wins: {wins}")
print(f"Win rate: {wins/num_games:.2%}")

# Theoretical probability
print(f"Theoretical win probability: {1/10:.2%}")
```

Trang trình bày 12: So sánh các phân phối rời rạc

Các phân bố rời rạc khác nhau có thể được sử dụng để mô hình hóa các hiện tượng khác nhau. Dưới đây là so sánh trực quan về các hàm khối lượng xác suất cho các phân bố mà chúng ta đã thảo luận.

```python
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Bernoulli
p = 0.7
x = [0, 1]
axs[0, 0].bar(x, [1-p, p])
axs[0, 0].set_title('Bernoulli (p=0.7)')

# Binomial
n, p = 10, 0.3
x = np.arange(0, n+1)
axs[0, 1].bar(x, binom.pmf(x, n, p))
axs[0, 1].set_title(f'Binomial (n={n}, p={p})')

# Geometric
p = 0.3
x = np.arange(1, 15)
axs[1, 0].bar(x, geom.pmf(x, p))
axs[1, 0].set_title(f'Geometric (p={p})')

# Poisson
lambda_param = 3
x = np.arange(0, 15)
axs[1, 1].bar(x, poisson.pmf(x, lambda_param))
axs[1, 1].set_title(f'Poisson (λ={lambda_param})')

plt.tight_layout()
plt.show()
```

Trang trình bày 13: Chọn cách phân phối phù hợp

Việc lựa chọn phân phối rời rạc phù hợp phụ thuộc vào bản chất của vấn đề:

1. Bernoulli: Dành cho kết quả nhị phân (thành công/thất bại).
2. Nhị thức: Cho số lần thành công trong các lần thử cố định.
3. Hình học: Cho số lần thử cho đến thành công đầu tiên.
4. Poisson: Cho số lượng sự kiện trong một khoảng thời gian cố định.
5. Đồng nhất: Dành cho các kết quả có khả năng xảy ra như nhau.

Hãy xem xét quy trình cơ bản và các giả định khi chọn phân phối để lập mô hình dữ liệu của bạn.

```python
# Example: Deciding between Binomial and Poisson
n, p = 1000, 0.003
lambda_param = n * p

x = np.arange(0, 15)
binom_pmf = binom.pmf(x, n, p)
poisson_pmf = poisson.pmf(x, lambda_param)

plt.plot(x, binom_pmf, 'bo-', label='Binomial')
plt.plot(x, poisson_pmf, 'ro-', label='Poisson')
plt.title(f'Binomial vs Poisson (n={n}, p={p}, λ={lambda_param})')
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.legend()
plt.show()
```

Trang trình bày 14: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về các bản phân phối rời rạc và ứng dụng của chúng, đây là một số tài nguyên có giá trị:

1. "Khảo sát về phân phối xác suất rời rạc" của Aleksandar Nanevski (arXiv:2102.07850) URL: [https://arxiv.org/abs/2102.07850](https://arxiv.org/abs/2102.07850)
2. "Phân phối xác suất trong khoa học vật lý" của Michael Trott (arXiv:1611.08318) URL: [https://arxiv.org/abs/1611.08318](https://arxiv.org/abs/1611.08318)
3. "Phân phối thống kê" của Catherine Forbes et al. (Sách, Wiley)
4. Các khóa học trực tuyến về lý thuyết xác suất và thống kê từ các nền tảng như Coursera, edX hoặc MIT OpenCourseWare.
