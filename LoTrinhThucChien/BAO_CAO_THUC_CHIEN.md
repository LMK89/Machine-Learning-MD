# BÁO CÁO TỔNG KẾT DỰ ÁN: LỘ TRÌNH HỌC AI/ML THỰC CHIẾN

## 1. Chiến lược & Yêu cầu gốc
- **Học thực chiến, phù hợp người trái ngành:** Kiến thức cô đọng, dễ hiểu, học tới đâu xài tới đó.
- **Sắp xếp xen kẽ (Mix Chủ đề):** Đan xen Toán, Python, Xử lý Dữ liệu và Model AI.
- **Giao diện HTML trực quan:** Sơ đồ roadmap + checklist theo dõi tiến độ.

---

## 2. Roadmap 4 Phase (Đã tối ưu lại)

### Phase 1: Toán nền tảng cho AI & Python Data Stack
*Không học Python cơ bản (đã biết). Tập trung vào toán và thư viện Python cho data.*

| # | Chủ đề | Mục tiêu | Mix với |
|---|---|---|---|
| 1 | Linear Algebra: Vector, Ma trận, Tensor | Nắm shape, broadcast, dot product | NumPy |
| 2 | Linear Algebra: Eigendecomposition, SVD | Hiểu PCA, SVD trong ML | PCA (Phase 3) |
| 3 | Calculus: Đạo hàm, Gradient, Chain Rule | Hiểu Gradient Descent | ML (Phase 3) |
| 4 | Probability: Phân phối, Bayes, MLE | Hiểu loss function, Naive Bayes | ML (Phase 3) |
| 5 | Statistics: Mean, Var, Cov, Correlation | Hiểu data distribution | EDA (Phase 2) |
| 6 | NumPy căn bản | Array ops, broadcasting | — |
| 7 | Pandas căn bản (đọc/ghi data) | Chuẩn bị data cho sau này | Data (Phase 2) |

**Case study thực chiến:** Dùng NumPy tự code Linear Regression từ scratch → thấy toán áp dụng vào code ngay.

---

### Phase 2: Data Analysis & Visualization
*Làm chủ data pipeline trước khi học model.*

| # | Chủ đề | Mục tiêu | Mix với |
|---|---|---|---|
| 1 | Pandas nâng cao: groupby, merge, pivot | Xử lý data thực tế | SQL join |
| 2 | Data Cleaning: missing values, outliers | 80% thời gian làm data | — |
| 3 | SQL căn bản: SELECT, JOIN, GROUP BY | Truy vấn data từ DB | Pandas merge |
| 4 | Matplotlib: line, bar, scatter, histogram | Trực quan data | EDA |
| 5 | Seaborn: heatmap, pairplot, boxplot | Phát hiện pattern | Correlation (Phase 1) |
| 6 | EDA (Exploratory Data Analysis) | Tổng hợp: clean → viz → insight | — |

**Case study:** Phân tích bộ dữ liệu thật (House Prices, Titanic) → báo cáo EDA hoàn chỉnh.

---

### Phase 3: Machine Learning (Cân bằng)
*Cân bằng lại: giảm PCA (chỉ 1-2 bài), thêm đủ classification, clustering, evaluation.*

| # | Chủ đề | Mục tiêu | Mix với |
|---|---|---|---|
| 1 | Linear Regression + Metrics (MSE, R²) | Hiểu regression cơ bản | Gradient (Phase 1) |
| 2 | Gradient Descent từ scratch | Hiểu optimization | Đạo hàm (Phase 1) |
| 3 | Logistic Regression + Confusion Matrix | Classification cơ bản | — |
| 4 | KNN (K-Nearest Neighbors) | Non-parametric learning | Distance metrics |
| 5 | Naive Bayes + Probability | Generative model | Bayes (Phase 1) |
| 6 | Decision Trees | Interpretable model | — |
| 7 | Random Forest | Ensemble cơ bản | Decision Trees |
| 8 | XGBoost / LightGBM | Ensemble nâng cao | — |
| 9 | SVM + Kernel Trick | Margin-based classifier | Linear Algebra |
| 10 | K-Means Clustering | Unsupervised learning | — |
| 11 | PCA (1 bài, không 25!) | Dimension reduction | SVD (Phase 1) |
| 12 | Regularization: L1, L2, ElasticNet | Chống overfitting | — |
| 13 | Cross-validation + Hyperparameter Tuning | Đánh giá model | — |
| 14 | Feature Engineering & Selection | Cải thiện accuracy | — |

**Case study:** Xây pipeline hoàn chỉnh: clean → feature engineering → train → eval → tune → deploy concept.

---

### Phase 4: Deep Learning & AI (Modern Stack)
*Tập trung CNN, Transformer, RAG, LLM. Bỏ RNN (ít dùng), thêm GAN/RL nếu cần.*

| # | Chủ đề | Mục tiêu | Mix với |
|---|---|---|---|
| 1 | Neural Network cơ bản: Perceptron, Activation | Foundation DL | Gradient (Phase 1) |
| 2 | CNN: Conv, Pooling, Padding, Stride | Computer Vision | — |
| 3 | CNN Architectures: VGG, ResNet | Transfer Learning | — |
| 4 | Object Detection: YOLO, SSD (overview) | Ứng dụng CV | — |
| 5 | Word Embeddings: Word2Vec, GloVe | NLP cơ bản | — |
| 6 | Transformer: Self-Attention, Multi-Head | Nền tảng LLM | — |
| 7 | BERT: Encoder-only, Fine-tuning | NLP classification | — |
| 8 | GPT: Decoder-only, Autoregressive | Text generation | — |
| 9 | LLM Fine-tuning: LoRA, PEFT | Fine-tune model | — |
| 10 | Prompt Engineering: Zero-shot, Chain-of-Thought | Dùng LLM hiệu quả | — |
| 11 | RAG: Retrieval-Augmented Generation | LLM + Knowledge base | — |
| 12 | Vector Database: FAISS, ChromaDB | Lưu embeddings | RAG |
| 13 | Evaluation: ROUGE, BLEU, Perplexity | Đánh giá LLM | — |
| 14 | Deployment: FastAPI, Docker, MLflow | Đưa model lên production | — |

**Case study:** Xây chatbot RAG với FastAPI + FAISS + LLM local (Ollama).

---

## 3. Lưu ý học tập
- **Không dịch trước:** Học tới đâu, đọc tài liệu gốc tiếng Anh tới đó. Chỉ dịch chú thích cá nhân khi cần.
- **Học theo project:** Mỗi phase làm 1 case study nhỏ (ghi trong roadmap).
- **Kho tài liệu gốc (2000+ file):** Nằm ở thư mục `Machine-Learning-MD/`. Khi cần học chủ đề nào → search trong đó.
- **Checklist HTML:** Mở `index.html` → tick ô khi học xong → track progress.

*(Cập nhật bởi Khang - 2026)*
