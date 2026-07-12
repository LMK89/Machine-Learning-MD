## Quy trình phân tích dữ liệu tài chính tự động

Trang trình bày 1: Giới thiệu về Phân tích tài khoản tự động Bài trình bày này phác thảo quy trình làm việc toàn diện để phân tích tài khoản tài chính bằng trí tuệ nhân tạo và Python. Chúng ta sẽ bắt đầu bằng việc tải bảng cân đối kế toán lên chatbot AI, sau đó chuyển sang trích xuất, định dạng dữ liệu và cuối cùng là phân tích chuyên sâu bằng mã Python tùy chỉnh.

Trang trình bày 2: Tải bảng cân đối kế toán lên Bước đầu tiên trong quy trình của chúng tôi là tải bảng cân đối kế toán hoặc tài liệu tài chính tương đương lên một chatbot AI như ChatGPT. Điều này thường có thể được thực hiện bằng cách dán văn bản trực tiếp vào giao diện trò chuyện hoặc bằng cách mô tả chi tiết nội dung của tài liệu cho AI.

Trang trình bày 3: Tương tác với AI Chatbot Sau khi bảng cân đối kế toán được tải lên, chúng ta cần hướng dẫn AI trích xuất dữ liệu tài chính liên quan. Đây là một ví dụ nhắc sử dụng:

```
Please extract the following financial data from the balance sheet I've provided:
1. Total Assets
2. Total Liabilities
3. Total Equity
4. Current Assets
5. Current Liabilities
6. Long-term Debt
7. Cash and Cash Equivalents

For each item, provide the monetary value and the corresponding year. Format the data as a Python dictionary.
```

Slide 4: Khai thác dữ liệu AI AI sẽ xử lý bảng cân đối kế toán và trích xuất thông tin được yêu cầu. Sau đó, nó sẽ định dạng dữ liệu dưới dạng từ điển Python, có thể dễ dàng sử dụng trong phân tích tiếp theo. Đây là một ví dụ về kết quả đầu ra có thể trông như thế nào:

```python
financial_data = {
    "2023": {
        "Total Assets": 1000000,
        "Total Liabilities": 600000,
        "Total Equity": 400000,
        "Current Assets": 300000,
        "Current Liabilities": 200000,
        "Long-term Debt": 400000,
        "Cash and Cash Equivalents": 150000
    },
    "2022": {
        "Total Assets": 900000,
        "Total Liabilities": 550000,
        "Total Equity": 350000,
        "Current Assets": 250000,
        "Current Liabilities": 180000,
        "Long-term Debt": 370000,
        "Cash and Cash Equivalents": 120000
    }
}
```

Slide 5: Preparing for Python Analysis With the data extracted and formatted, we can now move on to analyzing it using Python. We'll use libraries such as pandas for data manipulation and matplotlib for visualization. First, let's import the necessary libraries and convert our data into a pandas DataFrame.

Slide 6: Creating a DataFrame Here's the Python code to create a DataFrame from our extracted data:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(financial_data, orient='index')

# Display the DataFrame
print(df)
```

Mã này sẽ tạo một DataFrame có cấu trúc mà chúng ta có thể sử dụng để phân tích thêm.

Trang trình bày 7: Phân tích tỷ lệ tài chính cơ bản Bây giờ chúng ta đã có dữ liệu trong DataFrame, chúng ta có thể tính toán một số tỷ lệ tài chính cơ bản. Chúng ta sẽ tập trung vào tỷ lệ thanh khoản, tỷ lệ khả năng thanh toán và tỷ suất sinh lời.

Trang trình bày 8: Tính các hệ số thanh khoản Hãy tính các hệ số khả năng thanh toán hiện hành và khả năng thanh toán nhanh:

```python
# Current Ratio
df['Current Ratio'] = df['Current Assets'] / df['Current Liabilities']

# Quick Ratio (assuming 50% of Current Assets are inventory)
df['Quick Ratio'] = (df['Current Assets'] * 0.5) / df['Current Liabilities']

print(df[['Current Ratio', 'Quick Ratio']])
```

Trang trình bày 9: Tính tỷ lệ khả năng thanh toán Bây giờ chúng ta sẽ tính Tỷ lệ nợ trên vốn chủ sở hữu và Tỷ lệ nợ trên tài sản:

```python
# Debt-to-Equity Ratio
df['Debt-to-Equity Ratio'] = df['Total Liabilities'] / df['Total Equity']

# Debt-to-Assets Ratio
df['Debt-to-Assets Ratio'] = df['Total Liabilities'] / df['Total Assets']

print(df[['Debt-to-Equity Ratio', 'Debt-to-Assets Ratio']])
```

Trang trình bày 10: Trực quan hóa các xu hướng tài chính Để hiểu rõ hơn về các xu hướng tài chính, chúng ta có thể tạo các hình ảnh trực quan bằng matplotlib. Dưới đây là ví dụ về cách tạo biểu đồ thanh so sánh Tổng tài sản, Tổng nợ phải trả và Tổng vốn chủ sở hữu qua các năm:

```python
# Create a bar chart
df[['Total Assets', 'Total Liabilities', 'Total Equity']].plot(kind='bar', figsize=(10, 6))
plt.title('Financial Overview')
plt.xlabel('Year')
plt.ylabel('Amount')
plt.legend(loc='upper left')
plt.show()
```

Slide 11: Advanced Analysis - DuPont Analysis For more advanced analysis, we can perform a DuPont analysis, which breaks down Return on Equity (ROE) into its component parts. The DuPont formula is:

ROE = (Net Income / Sales) \* (Sales / Total Assets) \* (Total Assets / Equity)

This formula requires additional data not present in our balance sheet, so we'll need to ask the AI for more information.

Slide 12: Requesting Additional Data To perform the DuPont analysis, we need to request additional information from the AI. Here's a prompt to use:

```
Based on the financial data you extracted earlier, please provide the following additional information for both 2022 and 2023:
1. Net Income
2. Sales

Format the data as a Python dictionary, similar to the previous output.
```

Trang trình bày 13: Thực hiện Phân tích DuPont Sau khi có dữ liệu bổ sung, chúng tôi có thể thực hiện phân tích DuPont:

```python
# Assuming we've received the additional data and added it to our DataFrame
df['Net Profit Margin'] = df['Net Income'] / df['Sales']
df['Asset Turnover'] = df['Sales'] / df['Total Assets']
df['Equity Multiplier'] = df['Total Assets'] / df['Total Equity']
df['ROE'] = df['Net Profit Margin'] * df['Asset Turnover'] * df['Equity Multiplier']

print(df[['Net Profit Margin', 'Asset Turnover', 'Equity Multiplier', 'ROE']])
```

Trang trình bày 14: Diễn giải kết quả Bước cuối cùng trong quy trình làm việc của chúng tôi là diễn giải kết quả phân tích của chúng tôi. Điều này liên quan đến việc kiểm tra các tỷ lệ và xu hướng được tính toán để đưa ra kết luận có ý nghĩa về tình hình tài chính, hiệu quả và lợi nhuận của công ty.

Trang trình bày 15: Kết luận và các bước tiếp theo Quy trình công việc này trình bày cách kết hợp AI và Python để hợp lý hóa và nâng cao phân tích tài chính. Những cải tiến trong tương lai có thể bao gồm tự động hóa quy trình nhập dữ liệu, kết hợp các mô hình tài chính tiên tiến hơn và tạo giao diện thân thiện với người dùng cho những người dùng không rành về kỹ thuật.

Slide 16: Tài liệu tham khảo bổ sung

1. "Phân tích báo cáo tài chính" của Martin Fridson và Fernando Alvarez
2. "Python cho tài chính" của Yves Hilpisch
3. Tài liệu ChatGPT của OpenAI: [https://openai.com/chatgpt](https://openai.com/chatgpt)
4. Tài liệu về gấu trúc: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
5. Tài liệu về matplotlib: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
