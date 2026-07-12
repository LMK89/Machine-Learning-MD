## Hướng dẫn từng bước xây dựng phát hiện mô-đun với Rừng ngẫu nhiên và XGBoost trong Python
Trang trình bày 1: Giới thiệu về Mô hình phát hiện

Được phát triển để trở thành một ứng dụng quan trọng của máy học trong các ngành nghề nghiệp khác nhau. Bài trình bày này sẽ hướng dẫn bạn xây dựng mô-đun phát hiện bằng hai thuật toán mạnh mẽ: Rừng ngẫu nhiên và XGBoost. Chúng tôi sẽ sử dụng Python để phát triển các mô hình này, tập trung vào các bước thực tế, khả năng này dành cho người mới bắt đầu và người học ở cấp độ trung bình.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load a sample dataset (replace with your own data)
data = pd.read_csv('fraud_data.csv')
print(data.head())
```

Slide 2: Chuẩn bị dữ liệu

Trước khi xây dựng mô hình, chúng tôi cần chuẩn bị dữ liệu. Điều này liên quan đến việc tải dữ liệu, xử lý các giá trị bị thiếu, mã hóa các loại biến thể và chia dữ liệu cho người huấn luyện và kiểm tra tệp.

```python
# Handle missing values
data = data.fillna(data.mean())

# Encode categorical variables
data = pd.get_dummies(data, columns=['category', 'payment_method'])

# Split features and target
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
```

Slide 3: Mô hình rừng ngẫu nhiên

Rừng ngẫu nhiên là một phương pháp học tổng hợp, xây dựng nhiều cây quyết định và hợp lý nhất để có thể mong đợi độ chính xác và ổn định cao hơn. Hãy phát triển bộ phân loại Rừng ngẫu nhiên để tạo ra mô hình phát hiện cho chúng tôi.

```python
# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test)

# Evaluate the model
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_cm = confusion_matrix(y_test, rf_predictions)

print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Confusion Matrix:\n", rf_cm)
```

Trang trình bày 4: Model XGBoost

XGBoost (Tăng cường độ dốc eXtreme) là một phương pháp học tập tổng hợp mạnh mẽ khác sử dụng tính năng tăng cường độ dốc để tạo ra mô hình dự đoán mạnh mẽ. Hãy phát triển loại phân loại XGBoost để thực hiện nhiệm vụ phát hiện gian lận của chúng ta.

```python
# Initialize and train the XGBoost model
xgb_model = XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Make predictions
xgb_predictions = xgb_model.predict(X_test)

# Evaluate the model
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_cm = confusion_matrix(y_test, xgb_predictions)

print("XGBoost Accuracy:", xgb_accuracy)
print("XGBoost Confusion Matrix:\n", xgb_cm)
```

Trang trình bày 5: Tầm quan trọng của tính năng

Biết những tính năng nào đóng góp nhiều nhất để quyết định các mô hình của chúng tôi là rất quan trọng. Cả Rừng ngẫu nhiên và XGBoost đều cung cấp các phương pháp để tính toán tầm quan trọng của tính năng. Hãy hình dung điều này cho cả hai mô hình.

```python
import matplotlib.pyplot as plt

# Get feature importance for Random Forest
rf_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
rf_importance = rf_importance.sort_values('importance', ascending=False).head(10)

# Get feature importance for XGBoost
xgb_importance = pd.DataFrame({'feature': X.columns, 'importance': xgb_model.feature_importances_})
xgb_importance = xgb_importance.sort_values('importance', ascending=False).head(10)

# Plot feature importance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

rf_importance.plot(x='feature', y='importance', kind='bar', ax=ax1, title='Random Forest Feature Importance')
xgb_importance.plot(x='feature', y='importance', kind='bar', ax=ax2, title='XGBoost Feature Importance')

plt.tight_layout()
plt.show()
```

Slide 6: So sánh mô hình

Hiện tại, chúng tôi đã phát triển cả hai mô hình Rừng ngẫu nhiên và XGBoost, hãy so sánh hiệu suất của chúng bằng các số liệu khác nhau như độ chính xác, độ chính xác, khả năng thu hồi và điểm F1.

```python
from sklearn.metrics import classification_report

# Generate classification reports
rf_report = classification_report(y_test, rf_predictions)
xgb_report = classification_report(y_test, xgb_predictions)

print("Random Forest Classification Report:")
print(rf_report)
print("\nXGBoost Classification Report:")
print(xgb_report)

# Compare ROC curves
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(10, 6))
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {auc(rf_fpr, rf_tpr):.2f})')
plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {auc(xgb_fpr, xgb_tpr):.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
```

Slide 7: Điều chỉnh siêu thông số

Để cải thiện hiệu suất của mô hình, chúng tôi có thể điều chỉnh các siêu tham số của chúng. Chúng tôi sẽ sử dụng GridSearchCV để tìm ra sự hợp lý nhất tổng hợp tốt nhất cho Cả Rừng ngẫu nhiên và XGBoost.

```python
from sklearn.model_selection import GridSearchCV

# Random Forest hyperparameter tuning
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10]
}

rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=3, n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

print("Best Random Forest parameters:", rf_grid_search.best_params_)
print("Best Random Forest score:", rf_grid_search.best_score_)

# XGBoost hyperparameter tuning
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

xgb_grid_search = GridSearchCV(XGBClassifier(random_state=42), xgb_param_grid, cv=3, n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)

print("Best XGBoost parameters:", xgb_grid_search.best_params_)
print("Best XGBoost score:", xgb_grid_search.best_score_)
```

Slide 8: Diễn giải mô hình với SHAP

Giá trị SHAP (SHapley Additive exPlanations) giúp chúng tôi hiểu cách mỗi tính năng đóng góp được mong đợi của mô hình cho từng trường hợp riêng biệt. Vui lòng sử dụng SHAP để giải mã XGBoost của chúng tôi.

```python
import shap

# Create a SHAP explainer for the XGBoost model
explainer = shap.TreeExplainer(xgb_model)

# Calculate SHAP values for the test set
shap_values = explainer.shap_values(X_test)

# Visualize SHAP values for a single prediction
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# Plot summary of SHAP values
shap.summary_plot(shap_values, X_test)
```

Slide 9: Xử lý mất cân bằng dữ liệu

Bộ phát hiện dữ liệu nan thường bị mất cân bằng, số trường hợp không lừa đảo nhiều hơn số trường hợp lệ. Hãy cùng khám phá các kỹ thuật để xử lý tình trạng cân bằng này, đưa ra giới hạn như sử dụng lớp số và lấy mẫu trong quá trình.

```python
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Train Random Forest with class weights
rf_weighted = RandomForestClassifier(n_estimators=100, class_weight=class_weight_dict, random_state=42)
rf_weighted.fit(X_train, y_train)

# Apply SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train XGBoost on resampled data
xgb_resampled = XGBClassifier(n_estimators=100, random_state=42)
xgb_resampled.fit(X_train_resampled, y_train_resampled)

# Evaluate and compare the models
print("Weighted Random Forest Accuracy:", accuracy_score(y_test, rf_weighted.predict(X_test)))
print("XGBoost with SMOTE Accuracy:", accuracy_score(y_test, xgb_resampled.predict(X_test)))
```

Trang trình bày 10: Xác thực chéo

Để đảm bảo các mô hình của chúng tôi có khả năng hóa học tốt, chúng tôi sẽ sử dụng tính xác thực chéo k-Fold để đánh giá kết quả hiệu quả của chúng trên các tập dữ liệu khác nhau.

```python
from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)

# Perform 5-fold cross-validation for XGBoost
xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=5)

print("Random Forest CV Scores:", rf_cv_scores)
print("Random Forest Mean CV Score:", rf_cv_scores.mean())
print("\nXGBoost CV Scores:", xgb_cv_scores)
print("XGBoost Mean CV Score:", xgb_cv_scores.mean())

# Visualize cross-validation results
plt.figure(figsize=(10, 6))
plt.boxplot([rf_cv_scores, xgb_cv_scores], labels=['Random Forest', 'XGBoost'])
plt.title('Cross-Validation Scores')
plt.ylabel('Accuracy')
plt.show()
```

Trang trình bày 11: Ví dụ thực tế: Phát triển nan đơn hàng trong thương mại điện tử

Vui lòng áp dụng các mô hình phát hiện của chúng tôi vào các vấn đề về điện tử thương mại. Chúng tôi sẽ sử dụng các tính năng như đơn giá trị, lịch sử khách hàng, giao hàng địa chỉ và phương thức thanh toán để dự đoán các đơn hàng.

```python
# Sample e-commerce order data
order_data = pd.DataFrame({
    'order_value': [100, 500, 50, 1000, 200],
    'customer_age_days': [30, 365, 10, 180, 90],
    'shipping_billing_match': [1, 1, 0, 1, 1],
    'payment_method': ['credit_card', 'paypal', 'gift_card', 'credit_card', 'debit_card'],
    'is_fraud': [0, 0, 1, 0, 0]
})

# Prepare the data
order_data_encoded = pd.get_dummies(order_data, columns=['payment_method'])
X_order = order_data_encoded.drop('is_fraud', axis=1)
y_order = order_data_encoded['is_fraud']

# Train and evaluate the model
xgb_order = XGBClassifier(random_state=42)
xgb_order.fit(X_order, y_order)

# Make predictions on new data
new_order = pd.DataFrame({
    'order_value': [750],
    'customer_age_days': [5],
    'shipping_billing_match': [0],
    'payment_method_credit_card': [1],
    'payment_method_debit_card': [0],
    'payment_method_gift_card': [0],
    'payment_method_paypal': [0]
})

prediction = xgb_order.predict(new_order)
print("Fraud prediction for new order:", "Fraudulent" if prediction[0] == 1 else "Legitimate")
```

Trang trình chiếu 12: Ví dụ thực tế: Phát hiện email rác

Một ứng dụng phổ biến khác của công nghệ nano được phát hiện là email rác được xác định rõ. Chúng tôi sẽ sử dụng các tính năng như nội dung email, thông tin người gửi và siêu dữ liệu để phân loại email là thư rác hay không phải thư rác.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample email data
emails = [
    "Get rich quick! Limited time offer!",
    "Meeting agenda for tomorrow's conference",
    "Congratulations! You've won a free iPhone!",
    "Quarterly report attached for your review",
    "Urgent: Your account has been suspended"
]
labels = [1, 0, 1, 0, 1]  # 1 for spam, 0 for not spam

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_emails = vectorizer.fit_transform(emails)

# Train and evaluate the model
rf_spam = RandomForestClassifier(n_estimators=100, random_state=42)
rf_spam.fit(X_emails, labels)

# Make predictions on new emails
new_emails = [
    "Free trial membership for premium services!",
    "Project status update and next steps"
]
new_email_features = vectorizer.transform(new_emails)
predictions = rf_spam.predict(new_email_features)

for email, pred in zip(new_emails, predictions):
    print(f"Email: {email}")
    print(f"Prediction: {'Spam' if pred == 1 else 'Not Spam'}\n")
```

Slide 13: Triển khai và giám sát mô hình

Khi chúng tôi có mô hình phát hiện gian lận hoạt động tốt, điều quan trọng là phải phát triển mô hình đó là một hiệu quả và giám sát hiệu suất của mô hình đó theo thời gian. Đây là ví dụ về cách lưu và tải mô hình của chúng tôi cũng như thiết lập giám sát cơ bản.

```python
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import datetime
import pandas as pd

# Save the model
joblib.dump(xgb_model, 'fraud_detection_model.joblib')

# Load the model (simulating deployment)
loaded_model = joblib.load('fraud_detection_model.joblib')

# Function to make predictions and log results
def predict_and_log(model, X, y_true):
    y_pred = model.predict(X)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }

    # Log the results
    log_entry = pd.DataFrame([metrics], index=[datetime.datetime.now()])
    log_entry.to_csv('model_performance_log.csv', mode='a', header=False)

    return y_pred, metrics

# Simulate periodic monitoring
for i in range(3):  # Simulating 3 time periods
    print(f"Time period {i+1}")
    # In practice, you would use new data for each time period
    predictions, metrics = predict_and_log(loaded_model, X_test, y_test)
    print(f"Metrics: {metrics}")
    print()

# Read and display the log
log_df = pd.read_csv('model_performance_log.csv',
                     names=['timestamp', 'accuracy', 'precision', 'recall', 'f1_score'],
                     parse_dates=['timestamp'],
                     index_col='timestamp')
print("Model Performance Log:")
print(log_df)

# Plot performance metrics over time
log_df.plot(figsize=(10, 6))
plt.title('Model Performance Metrics Over Time')
plt.ylabel('Score')
plt.xlabel('Timestamp')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
```

Slide 14: Xử lý ý tưởng Drift

Khái niệm trôi dạt xảy ra khi danh sách đặc tính của các biến tiêu điểm thay đổi theo thời gian. Trong quá trình phát hiện, điều này có thể xảy ra khi chúng lừa đảo kỹ thuật điều chỉnh. Hãy thực hiện một phương pháp phát hiện sự trôi dạt đơn giản một cách đơn giản.

```python
import numpy as np
from scipy import stats

def detect_drift(baseline_predictions, new_predictions, threshold=0.05):
    # Perform Kolmogorov-Smirnov test
    ks_statistic, p_value = stats.ks_2samp(baseline_predictions, new_predictions)

    if p_value < threshold:
        print(f"Concept drift detected! p-value: {p_value}")
        return True
    else:
        print(f"No significant drift detected. p-value: {p_value}")
        return False

# Generate baseline predictions
baseline_predictions = xgb_model.predict_proba(X_test)[:, 1]

# Simulate new data (with potential drift)
np.random.seed(42)
drift_factor = np.random.normal(1, 0.2, size=X_test.shape[0])
X_test_drift = X_test * drift_factor

# Generate new predictions
new_predictions = xgb_model.predict_proba(X_test_drift)[:, 1]

# Detect drift
drift_detected = detect_drift(baseline_predictions, new_predictions)

if drift_detected:
    print("Consider retraining the model or investigating the cause of the drift.")
else:
    print("The model appears to be stable.")
```

Trang trình bày 15: Tài nguyên bổ sung

Để khám phá thêm về kỹ thuật phát hiện khoa học và thuật toán máy tính, hãy xem xét các tài nguyên sau:

1. "XGBoost: Hệ thống tăng cường cây có thể mở rộng" của Chen và Guestrin (2016) ArXiv: [https://arxiv.org/abs/1603.02754](https://arxiv.org/abs/1603.02754)
2. "Những khu rừng ngẫu nhiên" của Breiman (2001) ArXiv: [https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
3. "Khảo sát về kỹ thuật phát hiện thẻ tín dụng" của Zojaji et al. (2016) ArXiv: [https://arxiv.org/abs/1611.06439](https://arxiv.org/abs/1611.06439)
4. "Học từ dữ liệu không cân bằng" của He và Garcia (2009) IEEE: [https://ieeexplore.ieee.org/document/5128907](https://ieeexplore.ieee.org/document/5128907)

Các tài nguyên này cung cấp thông tin chuyên sâu về các thuật toán mà chúng tôi đã sử dụng cũng như các kỹ thuật bổ sung để phát hiện khoa học và xử lý cân bằng tập dữ liệu.
