## Nâng cao RAG với Sơ đồ tri thức trong Python

Trang trình bày 1: Giới thiệu về Sơ đồ tri thức

Sơ đồ tri thức là sự biểu diễn có cấu trúc của các thực thể trong thế giới thực và các mối quan hệ của chúng. Chúng cung cấp một cách mạnh mẽ để lưu trữ, sắp xếp và truy vấn thông tin phức tạp, khiến chúng trở nên lý tưởng để nâng cao các mô hình Thế hệ tăng cường truy xuất (RAG).

Mã số:

```python
import networkx as nx

# Create a knowledge graph
kg = nx.DiGraph()

# Add nodes (entities)
kg.add_node("Python", type="Programming Language")
kg.add_node("Java", type="Programming Language")
kg.add_node("C++", type="Programming Language")

# Add edges (relationships)
kg.add_edge("Python", "Java", relation="similar_to")
kg.add_edge("Python", "C++", relation="similar_to")
```

Trang trình bày 2: Thế hệ tăng cường truy xuất (RAG) là gì?

Thế hệ tăng cường truy xuất (RAG) là một phương pháp xử lý ngôn ngữ tự nhiên kết hợp các mô hình truy xuất và tạo. Đầu tiên, nó truy xuất thông tin liên quan từ nguồn tri thức (ví dụ: biểu đồ tri thức) và sau đó tạo phản hồi dựa trên thông tin được truy xuất.

Mã số:

```python
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="nq-open", passages=True)
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

question = "What is the capital of France?"
inputs = tokenizer(question, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, num_beams=2, early_stopping=True)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

Trang trình bày 3: Tích hợp Sơ đồ tri thức với RAG

Để nâng cao mô hình RAG bằng biểu đồ tri thức, chúng ta cần tạo một trình truy xuất tri thức có thể truy xuất thông tin liên quan từ biểu đồ tri thức dựa trên truy vấn đầu vào. Bộ thu hồi này sau đó có thể được tích hợp vào đường ống RAG.

Mã số:

```python
from rdflib import Graph

# Load the knowledge graph
kg = Graph().parse("path/to/knowledge_graph.ttl", format="turtle")

def retrieve_from_kg(query):
    # Define SPARQL query based on the input query
    sparql_query = """
        PREFIX : <http://example.org/>
        SELECT ?subject ?predicate ?object
        WHERE {
            ?subject ?predicate ?object .
            FILTER (
                regex(?subject, "%s", "i")
                || regex(?predicate, "%s", "i")
                || regex(?object, "%s", "i")
            )
        }
    """ % (query, query, query)

    # Execute the SPARQL query
    results = kg.query(sparql_query)

    # Return the retrieved triples
    return [(str(row.subject), str(row.predicate), str(row.object)) for row in results]
```

Slide 4: Xây dựng Sơ đồ tri thức

Trước khi tích hợp biểu đồ tri thức với RAG, bạn cần xây dựng biểu đồ tri thức. Quá trình này bao gồm việc trích xuất các thực thể, quan hệ và thông tin liên quan khác từ nhiều nguồn dữ liệu khác nhau và cấu trúc chúng theo định dạng biểu đồ.

Mã số:

```python
from rdflib import Graph, Literal, Namespace, URIRef

# Define namespaces
kg_ns = Namespace("http://example.org/kg#")

# Create a new knowledge graph
kg = Graph()

# Add triples to the knowledge graph
kg.add((URIRef(kg_ns["Paris"]), URIRef(kg_ns["capitalOf"]), URIRef(kg_ns["France"])))
kg.add((URIRef(kg_ns["Python"]), URIRef(kg_ns["programmingLanguage"]), Literal("Python")))
kg.add((URIRef(kg_ns["Java"]), URIRef(kg_ns["programmingLanguage"]), Literal("Java")))

# Serialize the knowledge graph to a file
kg.serialize("path/to/knowledge_graph.ttl", format="turtle")
```

Trang trình bày 5: Nhúng sơ đồ tri thức

Nhúng biểu đồ tri thức là quá trình biểu diễn các thực thể và quan hệ trong biểu đồ tri thức dưới dạng biểu diễn vectơ dày đặc. Điều này có thể cải thiện hiệu suất truy xuất kiến ​​thức và tăng cường tích hợp biểu đồ tri thức với các mô hình RAG.

Mã số:

```python
import numpy as np
from ampligraph.datasets import load_from_rdf
from ampligraph.latent_features import TransE

# Load the knowledge graph
kg = load_from_rdf("path/to/knowledge_graph.ttl", "turtle")

# Define the TransE model
model = TransE(batches_count=64, seed=0, epochs=200, k=100, eta=20)

# Train the model
X = np.array([kg.train_idx_obt[:]])
model.fit(X)

# Get embeddings for entities and relations
entity_embeddings = model.ent_embeddings
relation_embeddings = model.rel_embeddings
```

Trang trình bày 6: Truy vấn sơ đồ tri thức với SPARQL

SPARQL (Giao thức SPARQL và Ngôn ngữ truy vấn RDF) là ngôn ngữ truy vấn tiêu chuẩn để truy vấn và thao tác dữ liệu được lưu trữ ở định dạng RDF, thường được sử dụng để biểu diễn biểu đồ tri thức. SPARQL cho phép bạn truy xuất thông tin cụ thể từ biểu đồ tri thức dựa trên các truy vấn của bạn.

Mã số:

```python
from rdflib import Graph

# Load the knowledge graph
kg = Graph().parse("path/to/knowledge_graph.ttl", format="turtle")

# Define a SPARQL query
query = """
    PREFIX : <http://example.org/>
    SELECT ?capital ?country
    WHERE {
        ?capital :capitalOf ?country .
    }
"""

# Execute the SPARQL query
results = kg.query(query)

# Print the results
for row in results:
    print(f"{row.capital.value} is the capital of {row.country.value}")
```

Trang trình bày 7: Trực quan hóa sơ đồ tri thức

Trực quan hóa biểu đồ tri thức có thể cung cấp cái nhìn sâu sắc về cấu trúc, mối quan hệ và mẫu trong dữ liệu. Điều này có thể hữu ích để hiểu, khám phá và gỡ lỗi biểu đồ tri thức.

Mã số:

```python
import networkx as nx
import matplotlib.pyplot as plt

# Load the knowledge graph
kg = nx.read_gml("path/to/knowledge_graph.gml")

# Draw the knowledge graph
pos = nx.spring_layout(kg)
nx.draw(kg, pos, with_labels=True, node_color="skyblue", edge_color="gray")
plt.axis("off")
plt.show()
```

Slide 8: Lý luận về Sơ đồ tri thức

Lý luận về biểu đồ tri thức liên quan đến việc suy ra kiến ​​thức mới từ kiến ​​thức hiện có trong biểu đồ. Điều này có thể đạt được thông qua các kỹ thuật khác nhau, chẳng hạn như lý luận dựa trên quy tắc, học quan hệ thống kê hoặc các phương pháp dựa trên học sâu.

Mã số:

```python
from ampligraph.latent_features import ComplEx
from ampligraph.utils import create_bf

# Load the knowledge graph
kg = load_from_rdf("path/to/knowledge_graph.ttl", "turtle")

# Define the ComplEx model
model = ComplEx(batches_count=64, seed=0, epochs=200, k=100, eta=20)

# Train the model
X = np.array([kg.train_idx_obt[:]])
model.fit(X)

# Create a new batch of triples for inference
new_triples = create_bf(model, 100)

# Infer new knowledge from the knowledge graph
model.predict(new_triples)
```

Trang trình bày 9: Kết hợp sơ đồ tri thức

Hợp nhất biểu đồ tri thức bao gồm việc kết hợp nhiều biểu đồ tri thức thành một biểu đồ thống nhất duy nhất. Điều này có thể hữu ích khi làm việc với các nguồn dữ liệu đa dạng hoặc tích hợp các cơ sở kiến ​​thức bổ sung.

Mã số:

```python
from rdflib import Graph

# Load the first knowledge graph
kg1 = Graph().parse("path/to/kg1.ttl", format="turtle")

# Load the second knowledge graph
kg2 = Graph().parse("path/to/kg2.ttl", format="turtle")

# Create a new graph to hold the fused knowledge graph
fused_kg = Graph()

# Add triples from kg1 to the fused graph
for s, p, o in kg1.triples((None, None, None)):
    fused_kg.add((s, p, o))

# Add triples from kg2 to the fused graph
for s, p, o in kg2.triples((None, None, None)):
    fused_kg.add((s, p, o))

# Optionally, perform deduplication or conflict resolution
# ...

# Serialize the fused knowledge graph to a file
fused_kg.serialize("path/to/fused_kg.ttl", format="turtle")
```

Slide 10: Căn chỉnh Sơ đồ tri thức

Căn chỉnh biểu đồ tri thức là quá trình tìm kiếm sự tương ứng giữa các thực thể hoặc quan hệ trong các biểu đồ tri thức khác nhau. Điều này đặc biệt hữu ích khi tích hợp hoặc kết hợp nhiều biểu đồ tri thức vì nó giúp xác định và giải quyết các xung đột hoặc dư thừa tiềm ẩn.

Mã số:

```python
from ampligraph.evaluation import hits_at_n_train
from ampligraph.latent_features import TransE

# Load the first knowledge graph
kg1 = load_from_rdf("path/to/kg1.ttl", "turtle")

# Load the second knowledge graph
kg2 = load_from_rdf("path/to/kg2.ttl", "turtle")

# Define the TransE model
model = TransE(batches_count=64, seed=0, epochs=200, k=100, eta=20)

# Train the model on kg1
X1 = np.array([kg1.train_idx_obt[:]])
model.fit(X1)

# Evaluate the model on kg2 to find entity alignments
hits = hits_at_n_train(model, kg2, np.array([kg2.train_idx_obt[:]]), hits=[1, 3, 10])
print(hits)
```

Trang trình bày 11: Hoàn thiện Sơ đồ tri thức

Hoàn thành biểu đồ tri thức là nhiệm vụ suy ra các sự kiện hoặc mối quan hệ còn thiếu trong biểu đồ tri thức. Điều này có thể đạt được thông qua các kỹ thuật khác nhau, chẳng hạn như khai thác quy tắc, hệ số tensor hoặc các phương pháp dựa trên mạng thần kinh.

Mã số:

```python
import numpy as np
from ampligraph.latent_features import ComplEx
from ampligraph.utils import create_bf

# Load the knowledge graph
kg = load_from_rdf("path/to/knowledge_graph.ttl", "turtle")

# Define the ComplEx model
model = ComplEx(batches_count=64, seed=0, epochs=200, k=100, eta=20)

# Train the model
X = np.array([kg.train_idx_obt[:]])
model.fit(X)

# Create a batch of triples with missing components
new_triples = create_bf(model, 100, form="SP?")

# Predict the missing components (objects)
predictions = model.predict(new_triples)
```

Slide 12: Cập nhật Sơ đồ tri thức

Sơ đồ tri thức rất linh hoạt và có thể yêu cầu cập nhật khi có thông tin mới. Cập nhật biểu đồ tri thức bao gồm việc thêm, sửa đổi hoặc xóa các thực thể, quan hệ hoặc sự kiện trong biểu đồ.

Mã số:

```python
from rdflib import Graph, Literal, Namespace, URIRef

# Load the existing knowledge graph
kg = Graph().parse("path/to/knowledge_graph.ttl", format="turtle")

# Define namespaces
kg_ns = Namespace("http://example.org/kg#")

# Add a new triple to the knowledge graph
kg.add((URIRef(kg_ns["Python"]), URIRef(kg_ns["version"]), Literal("3.9")))

# Remove an existing triple from the knowledge graph
kg.remove((URIRef(kg_ns["Java"]), URIRef(kg_ns["programmingLanguage"]), Literal("Java")))

# Modify an existing triple in the knowledge graph
kg.remove((URIRef(kg_ns["Paris"]), URIRef(kg_ns["capitalOf"]), URIRef(kg_ns["France"])))
kg.add((URIRef(kg_ns["Paris"]), URIRef(kg_ns["capitalOf"]), URIRef(kg_ns["France"]), Literal("2024")))

# Serialize the updated knowledge graph to a file
kg.serialize("path/to/updated_kg.ttl", format="turtle")
```

Trang trình bày 13: Đánh giá sơ đồ tri thức

Đánh giá chất lượng và hiệu suất của biểu đồ tri thức là điều cần thiết để đảm bảo độ tin cậy và hiệu quả của nó. Một số số liệu và kỹ thuật có thể được sử dụng cho mục đích này, chẳng hạn như dự đoán liên kết, phân loại ba lần và phân giải thực thể.

Mã số:

```python
from ampligraph.evaluation import hits_at_n_train
from ampligraph.latent_features import TransE

# Load the knowledge graph
kg = load_from_rdf("path/to/knowledge_graph.ttl", "turtle")

# Define the TransE model
model = TransE(batches_count=64, seed=0, epochs=200, k=100, eta=20)

# Train the model
X = np.array([kg.train_idx_obt[:]])
model.fit(X)

# Evaluate the model using link prediction
hits = hits_at_n_train(model, kg, np.array([kg.train_idx_obt[:]]), hits=[1, 3, 10])
print(hits)
```

Trang trình bày 14: Tài nguyên bổ sung

Để tìm hiểu và khám phá thêm về biểu đồ tri thức cũng như sự tích hợp của chúng với các mô hình RAG, dưới đây là một số tài nguyên bổ sung:

* Khảo sát ứng dụng và xây dựng sơ đồ tri thức (ArXiv): [https://arxiv.org/abs/2205.04888](https://arxiv.org/abs/2205.04888)
* Kỹ thuật, ứng dụng và điểm chuẩn nhúng sơ đồ tri thức: Khảo sát (ArXiv): [https://arxiv.org/abs/2002.00819](https://arxiv.org/abs/2002.00819)
* Thế hệ tăng cường truy xuất cho các nhiệm vụ NLP chuyên sâu tri thức (Giấy): [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)

Các tài nguyên này cung cấp thông tin chuyên sâu, tài liệu nghiên cứu và khảo sát về các khía cạnh khác nhau của biểu đồ tri thức và ứng dụng của chúng trong các nhiệm vụ xử lý ngôn ngữ tự nhiên, bao gồm cả Thế hệ tăng cường truy xuất (RAG).
