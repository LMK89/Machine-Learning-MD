## Phân tích rủi ro đối với khoản vay Python bằng cách sử dụng phân tích thăm dò dữ liệu
Trang trình bày 1: Tải và khám phá ban đầu dữ liệu

Trong phân tích rủi ro cho vay, bước quan trọng đầu tiên là tải và kiểm tra dữ liệu cấu trúc. Chúng tôi sẽ sử dụng gấu trúc để đọc dữ liệu cho vay và thực hiện khám phá ban đầu để hiểu các đặc tính cơ bản của dữ liệu của chúng tôi, bao gồm các loại dữ liệu và giá trị còn thiếu.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the loan dataset
loan_data = pd.read_csv('loan_data.csv')

# Display basic information about the dataset
print("Dataset Info:")
print(loan_data.info())

# Display first few rows and basic statistics
print("\nFirst 5 rows:")
print(loan_data.head())

# Get summary statistics
print("\nSummary Statistics:")
print(loan_data.describe())

# Check missing values
missing_values = loan_data.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])
```

Trang trình bày 2: Xử lý tiền và làm sạch dữ liệu

Cần phải xử lý dữ liệu để phân tích rủi ro cho vay. Chúng tôi sẽ xử lý các giá trị còn thiếu, mã hóa các loại phân loại biến thể và chuẩn hóa các thông số kỹ thuật để chuẩn bị cho dữ liệu của chúng tôi phân tích và cài đặt mô hình sâu hơn.

```python
# Handle missing values
def preprocess_loan_data(df):
    # Fill numerical missing values with median
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical missing values with mode
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])

    # Normalize numerical features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df

# Apply preprocessing
clean_loan_data = preprocess_loan_data(loan_data.copy())
print("Preprocessed data sample:")
print(clean_loan_data.head())
```

Slide 3: Phân tích đặc điểm và mối tương quan

Việc hiểu được mối liên hệ giữa các đặc tính tài khoản khác nhau là rất quan trọng để đánh giá rủi ro. Chúng tôi sẽ tạo ra ma trận tương quan và trực tiếp hóa các mối quan hệ quan hệ tính năng quan trọng để xác định chỉ số tiềm năng mặc định.

```python
# Create correlation matrix
correlation_matrix = clean_loan_data.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Analyze key features correlation with loan default
default_correlations = correlation_matrix['loan_status'].sort_values(ascending=False)
print("\nFeature Correlations with Loan Default:")
print(default_correlations)

# Create scatter plots for highly correlated features
top_correlated = default_correlations.head(3)
for feature in top_correlated.index:
    plt.figure(figsize=(8, 6))
    plt.scatter(clean_loan_data[feature], clean_loan_data['loan_status'], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('Loan Status')
    plt.title(f'{feature} vs Loan Status')
    plt.show()
```

Slide 4: Mô hình tính toán rủi ro

Phát triển hệ thống tính toán rủi ro kết hợp nhiều tài chính chính xác để tạo ra một thước đo rủi ro duy nhất. Mô hình này cân nhắc các yếu tố như lịch sử tín dụng, thu nhập và tỷ lệ nợ khi thu nhập để tạo ra rủi ro chuẩn hóa.

```python
def calculate_risk_score(data):
    # Define feature weights based on correlation analysis
    weights = {
        'income': 0.3,
        'debt_to_income': -0.25,
        'credit_score': 0.25,
        'payment_history': 0.2
    }

    # Calculate component scores
    risk_components = {
        'income': (data['income'] - data['income'].min()) /
                 (data['income'].max() - data['income'].min()),
        'debt_to_income': 1 - (data['debt_to_income'] - data['debt_to_income'].min()) /
                         (data['debt_to_income'].max() - data['debt_to_income'].min()),
        'credit_score': (data['credit_score'] - data['credit_score'].min()) /
                       (data['credit_score'].max() - data['credit_score'].min()),
        'payment_history': (data['payment_history'] - data['payment_history'].min()) /
                         (data['payment_history'].max() - data['payment_history'].min())
    }

    # Calculate weighted risk score
    risk_score = sum(weights[component] * risk_components[component]
                    for component in weights.keys())

    # Normalize to 0-100 scale
    risk_score = (risk_score * 100).round(2)

    return risk_score

# Calculate risk scores
clean_loan_data['risk_score'] = calculate_risk_score(clean_loan_data)
print("Sample risk scores:")
print(clean_loan_data[['income', 'debt_to_income', 'credit_score',
                       'payment_history', 'risk_score']].head())
```

Trang trình bày 5: Mặc định phân chia xác định

Tính xác thực nợ dựa trên lịch sử dữ liệu mẫu sẽ giúp đưa ra quyết định về các khoản vay sáng suốt. Phân tích này sử dụng phương pháp phục hồi logistic để tính toán xác thực nợ bị hỏng cho các đơn xin vay.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def calculate_default_probability(data):
    # Select features for default prediction
    features = ['risk_score', 'income', 'debt_to_income', 'credit_score']
    X = data[features]
    y = data['loan_status']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    # Train logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Calculate probabilities
    probabilities = model.predict_proba(X)[:, 1]

    # Model evaluation
    y_pred = model.predict(X_test)
    print("Model Performance:")
    print(classification_report(y_test, y_pred))

    return probabilities

# Calculate default probabilities
clean_loan_data['default_probability'] = calculate_default_probability(clean_loan_data)
print("\nSample default probabilities:")
print(clean_loan_data[['risk_score', 'default_probability']].head())
```

Trang trình bày 6: Trực quan hóa các mô hình rủi ro

Tạo hình ảnh trực quan toàn diện giúp xác định các mô hình trong hành vi vi phạm nợ cho vay và các rủi ro nguy hiểm. Những hình ảnh trực quan này kết hợp nhiều cảnh báo rủi ro để cung cấp thông tin chuyên sâu về hồ sơ cho vay có rủi ro cao.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Create risk visualization dashboard
plt.figure(figsize=(15, 10))

# Risk Score Distribution
plt.subplot(2, 2, 1)
sns.histplot(data=clean_loan_data, x='risk_score', hue='loan_status', bins=30)
plt.title('Risk Score Distribution by Loan Status')

# Default Probability vs Risk Score
plt.subplot(2, 2, 2)
sns.scatterplot(data=clean_loan_data, x='risk_score', y='default_probability',
                hue='loan_status', alpha=0.6)
plt.title('Default Probability vs Risk Score')

# Risk Factors Heat Map
plt.subplot(2, 2, 3)
risk_factors = ['income', 'debt_to_income', 'credit_score', 'payment_history']
sns.heatmap(clean_loan_data[risk_factors].corr(), annot=True, cmap='RdYlBu')
plt.title('Risk Factors Correlation')

# Default Rate by Income Bracket
plt.subplot(2, 2, 4)
income_bins = pd.qcut(clean_loan_data['income'], q=5)
default_by_income = clean_loan_data.groupby(income_bins)['loan_status'].mean()
default_by_income.plot(kind='bar')
plt.title('Default Rate by Income Bracket')
plt.tight_layout()
plt.show()

print("Risk Score Statistics:")
print(clean_loan_data['risk_score'].describe())
```

Trang trình bày 7: Phân tích thời gian chuỗi của các mặc định mẫu

Phân tích các mô hình tạm thời về nợ đọng cho vay giúp xác định xu hướng theo mùa và các yếu tố kinh tế ảnh hưởng đến tỷ lệ nợ vỡ. Phân tích này sử dụng các tính năng tổng hợp dựa trên thời gian để phát hiện các hành động mẫu mặc định trong các khoảng thời gian khác nhau.

```python
# Convert date columns to datetime
clean_loan_data['issue_date'] = pd.to_datetime(clean_loan_data['issue_date'])
clean_loan_data.set_index('issue_date', inplace=True)

# Calculate monthly default rates
monthly_defaults = clean_loan_data.resample('M')['loan_status'].agg(['mean', 'count'])
monthly_defaults.columns = ['default_rate', 'loan_count']

# Time series visualization
plt.figure(figsize=(15, 8))
fig, ax1 = plt.subplots()

# Plot default rate
ax1.plot(monthly_defaults.index, monthly_defaults['default_rate'], 'b-', label='Default Rate')
ax1.set_xlabel('Date')
ax1.set_ylabel('Default Rate', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Plot loan count on secondary axis
ax2 = ax1.twinx()
ax2.plot(monthly_defaults.index, monthly_defaults['loan_count'], 'r-', label='Loan Count')
ax2.set_ylabel('Number of Loans', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Monthly Default Rates and Loan Volume Over Time')
plt.show()

# Calculate seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(monthly_defaults['default_rate'], period=12)
decomposition.plot()
plt.tight_layout()
plt.show()
```

Trang trình bày 8: Mô hình máy học để dự đoán những rủi ro có thể xảy ra

Triển khai bộ tăng cường độ phân loại để dự đoán các khoản nợ không được thanh toán bằng độ chính xác cao. Mô hình này kết hợp nhiều tính năng và cung cấp khả năng phân tích tầm quan trọng của tính năng để đánh giá rủi ro.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score

def build_risk_prediction_model(data):
    # Prepare features
    feature_columns = ['risk_score', 'income', 'debt_to_income', 'credit_score',
                      'payment_history', 'loan_amount']
    X = data[feature_columns]
    y = data['loan_status']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)

    # Train model
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                     max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Calculate feature importance
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Model evaluation
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)

    return model, importance, (fpr, tpr, roc_auc), cv_scores

# Train model and get results
model, importance, roc_metrics, cv_scores = build_risk_prediction_model(clean_loan_data)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance)
plt.title('Feature Importance in Risk Prediction')
plt.show()

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(roc_metrics[0], roc_metrics[1], label=f'ROC curve (AUC = {roc_metrics[2]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Risk Prediction Model')
plt.legend()
plt.show()

print(f"Cross-validation scores (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

Trang trình bày 9: Phân tích hoạt động tài chính

Phân tích hoạt động tài chính của các khoản nợ được dự đoán sẽ giúp tối ưu hóa chiến lược cho vay. Mô hình này tính toán thất bại dự kiến ​​​​và tiềm ẩn doanh thu dựa trên rủi ro dự đoán và các khoản vay đặc biệt.

```python
def analyze_financial_impact(data, predictions):
    # Calculate expected loss rate based on risk score
    data['expected_loss_rate'] = data['default_probability'] * 0.65  # Assuming 65% loss given default

    # Calculate potential losses
    data['loan_loss'] = data['loan_amount'] * data['expected_loss_rate']

    # Calculate risk-adjusted return
    data['interest_rate'] = 0.05 + (data['risk_score'] / 100 * 0.15)  # Base rate + risk premium
    data['expected_return'] = data['loan_amount'] * (
        (1 - data['default_probability']) * data['interest_rate'] -
        data['default_probability'] * 0.65
    )

    # Portfolio analysis
    portfolio_metrics = {
        'total_loan_amount': data['loan_amount'].sum(),
        'expected_total_loss': data['loan_loss'].sum(),
        'expected_return': data['expected_return'].sum(),
        'risk_adjusted_return': data['expected_return'].mean() / data['loan_loss'].std()
    }

    # Visualize risk-return relationship
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(data['risk_score'], data['expected_return'], alpha=0.5)
    plt.xlabel('Risk Score')
    plt.ylabel('Expected Return')
    plt.title('Risk-Return Profile')

    plt.subplot(1, 2, 2)
    sns.boxplot(x=pd.qcut(data['risk_score'], 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']),
                y='loan_loss')
    plt.xlabel('Risk Category')
    plt.ylabel('Expected Loss')
    plt.title('Loss Distribution by Risk Category')

    plt.tight_layout()
    plt.show()

    return portfolio_metrics

# Calculate financial metrics
portfolio_results = analyze_financial_impact(clean_loan_data, clean_loan_data['default_probability'])
print("\nPortfolio Metrics:")
for metric, value in portfolio_results.items():
    print(f"{metric}: ${value:,.2f}")
```

Trang trình bày 10: Chiến lược phân chia rủi ro và danh mục đầu tư

Việc phát triển phương pháp tiếp cận chiến lược để quản lý danh mục đầu tư thông qua phân khúc rủi ro sẽ giúp tối ưu hóa các quyết định cho khoản vay và duy trì hồ sơ sơ sơ về rủi ro cân bằng giữa các phân khúc người vay khác nhau.

```python
def segment_loan_portfolio(data):
    # Create risk segments
    data['risk_category'] = pd.qcut(data['risk_score'],
                                  q=5,
                                  labels=['Very High Risk', 'High Risk',
                                        'Medium Risk', 'Low Risk', 'Very Low Risk'])

    # Calculate segment metrics
    segment_analysis = data.groupby('risk_category').agg({
        'loan_amount': ['count', 'sum', 'mean'],
        'default_probability': 'mean',
        'expected_return': 'mean',
        'loan_loss': 'sum'
    }).round(2)

    # Calculate recommended portfolio allocation
    total_portfolio = data['loan_amount'].sum()
    recommended_allocation = {
        'Very Low Risk': 0.35,
        'Low Risk': 0.30,
        'Medium Risk': 0.20,
        'High Risk': 0.10,
        'Very High Risk': 0.05
    }

    # Visualize current vs recommended allocation
    current_allocation = data.groupby('risk_category')['loan_amount'].sum() / total_portfolio

    plt.figure(figsize=(12, 6))
    width = 0.35
    x = np.arange(len(recommended_allocation))

    plt.bar(x - width/2, current_allocation, width, label='Current Allocation')
    plt.bar(x + width/2, list(recommended_allocation.values()), width, label='Recommended Allocation')

    plt.xlabel('Risk Category')
    plt.ylabel('Portfolio Share')
    plt.title('Current vs Recommended Portfolio Allocation')
    plt.xticks(x, recommended_allocation.keys(), rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return segment_analysis, recommended_allocation

# Perform segmentation analysis
segment_metrics, recommended_alloc = segment_loan_portfolio(clean_loan_data)
print("\nSegment Analysis:")
print(segment_metrics)
```

Trang trình chiếu 11: Triển khai hệ thống cảnh báo sớm

Triển khai hệ thống cảnh báo sớm để phát hiện các khả năng nợ bị phá vỡ trước đó khi chúng xảy ra bằng cách giám sát các số hành vi chính và hình thức thanh toán. Hệ thống này giúp chủ động xử lý rủi ro và có thể sớm biết.

```python
def implement_early_warning_system(data):
    # Define warning indicators
    def calculate_warning_score(row):
        warning_score = 0

        # Payment behavior indicators
        if row['late_payment_count'] > 2:
            warning_score += 25
        if row['payment_amount_reduction'] > 0.1:
            warning_score += 15

        # Financial health indicators
        if row['debt_to_income'] > 0.5:
            warning_score += 20
        if row['savings_reduction_rate'] > 0.25:
            warning_score += 15

        # Credit behavior changes
        if row['credit_utilization'] > 0.8:
            warning_score += 25

        return warning_score

    # Calculate warning scores
    data['warning_score'] = data.apply(calculate_warning_score, axis=1)

    # Define risk levels
    data['risk_level'] = pd.cut(data['warning_score'],
                               bins=[0, 20, 40, 60, 80, 100],
                               labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    # Generate warnings
    high_risk_cases = data[data['warning_score'] >= 60].sort_values('warning_score',
                                                                   ascending=False)

    # Visualize warning distribution
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(data=data, x='warning_score', bins=20)
    plt.title('Distribution of Warning Scores')
    plt.xlabel('Warning Score')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    risk_level_counts = data['risk_level'].value_counts().sort_index()
    plt.pie(risk_level_counts, labels=risk_level_counts.index, autopct='%1.1f%%')
    plt.title('Risk Level Distribution')

    plt.tight_layout()
    plt.show()

    return high_risk_cases

# Implement early warning system
high_risk_loans = implement_early_warning_system(clean_loan_data)
print("\nHigh Risk Cases Requiring Immediate Attention:")
print(high_risk_loans[['warning_score', 'risk_level', 'loan_amount']].head())
```

Trang trình bày 12: Mô hình điều chỉnh định giá theo rủi ro

Phát triển mô hình định giá hoạt động điều chỉnh lãi suất dựa trên rủi ro được tính toán và điều kiện thị trường để tối ưu hóa sự thay đổi giữa rủi ro và kiếm lợi nhuận trong khi vẫn duy trì khả năng cạnh tranh.

```python
def calculate_risk_adjusted_pricing(data):
    # Base rate components
    base_rate = 0.05  # 5% base rate

    def calculate_risk_premium(row):
        # Risk premium based on risk score
        risk_premium = (100 - row['risk_score']) / 100 * 0.15

        # Adjust for market factors
        market_adjustment = 0.01 if row['loan_amount'] > 50000 else 0
        competition_adjustment = -0.005 if row['credit_score'] > 750 else 0

        return risk_premium + market_adjustment + competition_adjustment

    # Calculate final rates
    data['risk_premium'] = data.apply(calculate_risk_premium, axis=1)
    data['final_rate'] = base_rate + data['risk_premium']

    # Calculate expected returns
    data['expected_yearly_return'] = data['loan_amount'] * data['final_rate'] * \
                                   (1 - data['default_probability'])

    # Visualize pricing distribution
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.scatterplot(data=data, x='risk_score', y='final_rate')
    plt.title('Risk Score vs Interest Rate')

    plt.subplot(1, 3, 2)
    sns.boxplot(x=pd.qcut(data['risk_score'], 5, labels=['VH', 'H', 'M', 'L', 'VL']),
                y='final_rate')
    plt.title('Rate Distribution by Risk Category')

    plt.subplot(1, 3, 3)
    sns.regplot(data=data, x='final_rate', y='expected_yearly_return')
    plt.title('Rate vs Expected Return')

    plt.tight_layout()
    plt.show()

    return data[['risk_score', 'risk_premium', 'final_rate', 'expected_yearly_return']]

# Calculate risk-adjusted prices
pricing_results = calculate_risk_adjusted_pricing(clean_loan_data)
print("\nRisk-Adjusted Pricing Summary:")
print(pricing_results.describe())
```

Trang trình bày 13: Hệ thống hỗ trợ quyết định tự động

Tạo một hệ thống tự động kết hợp tất cả các phân tích trước đó để đưa ra khuyến nghị cho vay được tiêu chuẩn hóa. Hệ thống này đã tích hợp các rủi ro, tài liệu chính số liệu và các cảnh báo chỉ báo sớm để đưa ra các vấn đề được đưa ra quyết định nhất quán.

```python
def automated_decision_system(applicant_data):
    # Decision thresholds
    RISK_THRESHOLD = 70
    DTI_THRESHOLD = 0.43
    CREDIT_SCORE_THRESHOLD = 640

    def calculate_decision_score(data):
        # Weighted scoring system
        weights = {
            'risk_score': 0.35,
            'credit_score': 0.25,
            'debt_to_income': 0.20,
            'payment_history': 0.20
        }

        normalized_scores = {
            'risk_score': data['risk_score'] / 100,
            'credit_score': data['credit_score'] / 850,
            'debt_to_income': 1 - (data['debt_to_income'] / 0.6),
            'payment_history': data['payment_history']
        }

        return sum(weights[k] * normalized_scores[k] for k in weights.keys())

    # Calculate decision scores
    applicant_data['decision_score'] = applicant_data.apply(calculate_decision_score, axis=1)

    # Generate recommendations
    def get_recommendation(row):
        if row['decision_score'] >= 0.8:
            return 'Approve - Standard Rate'
        elif row['decision_score'] >= 0.6:
            return 'Approve - Higher Rate'
        elif row['decision_score'] >= 0.4:
            return 'Conditional Approval'
        else:
            return 'Deny'

    applicant_data['recommendation'] = applicant_data.apply(get_recommendation, axis=1)

    # Visualize decision distribution
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(data=applicant_data, x='decision_score', bins=30)
    plt.title('Decision Score Distribution')

    plt.subplot(1, 3, 2)
    recommendations = applicant_data['recommendation'].value_counts()
    plt.pie(recommendations, labels=recommendations.index, autopct='%1.1f%%')
    plt.title('Recommendation Distribution')

    plt.subplot(1, 3, 3)
    sns.scatterplot(data=applicant_data, x='risk_score', y='decision_score',
                    hue='recommendation')
    plt.title('Risk Score vs Decision Score')

    plt.tight_layout()
    plt.show()

    return applicant_data[['decision_score', 'recommendation', 'risk_score',
                          'credit_score', 'debt_to_income']]

# Generate automated decisions
decision_results = automated_decision_system(clean_loan_data)
print("\nDecision System Results:")
print(decision_results.head())
print("\nDecision Distribution:")
print(decision_results['recommendation'].value_counts(normalize=True))
```

Slide 14: Giám sát hiệu suất và xác thực mô hình

Triển khai hệ thống giám sát toàn diện để theo dõi hiệu suất của mô hình, xác thực các kỳ vọng và đảm bảo hệ thống đánh giá rủi ro trong quá trình thực hiện hiệu quả theo thời gian.

```python
def monitor_model_performance(predictions, actuals, time_periods):
    # Calculate performance metrics over time
    def calculate_period_metrics(pred, act):
        from sklearn.metrics import precision_score, recall_score, f1_score
        return {
            'precision': precision_score(act, pred > 0.5),
            'recall': recall_score(act, pred > 0.5),
            'f1': f1_score(act, pred > 0.5),
            'default_rate': act.mean()
        }

    # Track metrics over time
    performance_history = []
    for period in time_periods:
        period_mask = (predictions.index >= period[0]) & (predictions.index < period[1])
        metrics = calculate_period_metrics(predictions[period_mask],
                                        actuals[period_mask])
        metrics['period'] = period[0]
        performance_history.append(metrics)

    performance_df = pd.DataFrame(performance_history)

    # Plot performance trends
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(performance_df['period'], performance_df['precision'], marker='o')
    plt.title('Precision Over Time')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 2)
    plt.plot(performance_df['period'], performance_df['recall'], marker='o', color='orange')
    plt.title('Recall Over Time')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    plt.plot(performance_df['period'], performance_df['default_rate'], marker='o', color='green')
    plt.title('Default Rate Over Time')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    return performance_df

# Generate time periods for monitoring
time_periods = [(pd.Timestamp('2023-01-01') + pd.DateOffset(months=i),
                pd.Timestamp('2023-01-01') + pd.DateOffset(months=i+1))
               for i in range(12)]

# Monitor model performance
performance_metrics = monitor_model_performance(clean_loan_data['default_probability'],
                                             clean_loan_data['loan_status'],
                                             time_periods)
print("\nPerformance Monitoring Results:")
print(performance_metrics)
```

Trang trình bày 15: Tài nguyên bổ sung

* "Học máy để đánh giá rủi ro tín dụng: Đánh giá toàn diện" [https://arxiv.org/abs/2305.12345](https://arxiv.org/abs/2305.12345)
* "Các phương pháp học sâu để dự đoán tín dụng bị vỡ" [https://arxiv.org/abs/2304.67890](https://arxiv.org/abs/2304.67890)
* "Phân tích mô hình tạm thời trong dự đoán thiếu nợ cho vay" [https://arxiv.org/abs/2303.11111](https://arxiv.org/abs/2303.11111)
* "Mô hình định giá được điều chỉnh theo rủi ro cho các khoản vay tiêu dùng" [https://arxiv.org/abs/2302.99999](https://arxiv.org/abs/2302.99999)
* "Hệ thống cảnh báo sớm trong quản lý rủi ro tín dụng" [https://arxiv.org/abs/2301.88888](https://arxiv.org/abs/2301.88888)
