import os
import json

nodes = []
edges = []
node_id = 1

# Phase definition for styling
phases = {
    "01_NenTang_Python_Toan": {"id": 1, "label": "Phần 1: Nền Tảng\nPython & Toán", "group": "phase1", "level": 1},
    "02_PhanTichDuLieu_DataScience": {"id": 2, "label": "Phần 2: Xử Lý Dữ Liệu\n(Data Science)", "group": "phase2", "level": 2},
    "03_HocMay_MachineLearning": {"id": 3, "label": "Phần 3: Học Máy\n(Machine Learning)", "group": "phase3", "level": 3},
    "04_HocSau_AI_ThucChien": {"id": 4, "label": "Phần 4: Trí Tuệ Nhân Tạo\n(Deep Learning & LLM)", "group": "phase4", "level": 4}
}

# Add phase root nodes
phase_roots = {}
for p_key, p_val in phases.items():
    nodes.append({
        "id": p_val["id"],
        "label": p_val["label"],
        "group": p_val["group"],
        "shape": "hexagon",
        "level": p_val["level"],
        "font": {"size": 20, "bold": True}
    })
    phase_roots[p_key] = p_val["id"]

# Flow edges between phases
edges.append({"from": 1, "to": 2, "arrows": "to", "label": "Cung cấp nền tảng số học & code", "width": 3})
edges.append({"from": 2, "to": 3, "arrows": "to", "label": "Cung cấp dữ liệu sạch", "width": 3})
edges.append({"from": 3, "to": 4, "arrows": "to", "label": "Cung cấp thuật toán tối ưu", "width": 3})

current_id = 10
for phase_dir in sorted(phases.keys()):
    path = os.path.join("LoTrinhThucChien", phase_dir)
    if os.path.isdir(path):
        files = sorted([f for f in os.listdir(path) if f.endswith(".md")])
        root_id = phase_roots[phase_dir]
        level = phases[phase_dir]["level"]
        for f in files:
            name = f[:-3] # remove .md
            # Shorten label for node
            short_name = name[3:30] + "..." if len(name) > 30 else name[3:]
            nodes.append({
                "id": current_id,
                "label": short_name,
                "title": name, # Tooltip on hover
                "group": phases[phase_dir]["group"],
                "shape": "dot",
                "size": 10,
                "level": level
            })
            # Connect file to its phase root
            edges.append({
                "from": root_id,
                "to": current_id,
                "color": {"opacity": 0.3}
            })
            current_id += 1

html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lộ Trình Học AI/ML Thực Chiến (80 Bài Cốt Lõi)</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #121212;
            color: #ffffff;
            margin: 0;
            padding: 20px;
        }}
        h1 {{ text-align: center; color: #00e676; margin-bottom: 5px; }}
        p.subtitle {{ text-align: center; font-size: 1.1em; color: #b0bec5; margin-top: 0; }}
        #mynetwork {{
            width: 100%;
            height: 800px;
            border: 1px solid #333;
            border-radius: 12px;
            background-color: #1e1e1e;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.8);
        }}
    </style>
</head>
<body>

    <h1>🧠 Lộ Trình Học AI & Machine Learning Thực Chiến</h1>
    <p class="subtitle">Sơ đồ tương tác phân cấp (Hierarchical) 80 bài học cốt lõi</p>

    <div id="mynetwork"></div>

    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});

        var container = document.getElementById('mynetwork');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            layout: {{
                hierarchical: {{
                    direction: 'UD',
                    sortMethod: 'directed',
                    nodeSpacing: 150,
                    levelSeparation: 200
                }}
            }},
            groups: {{
                phase1: {{color:{{background:'#ff9800', border:'#e65100'}}, font:{{color:'white'}}}},
                phase2: {{color:{{background:'#2196f3', border:'#0d47a1'}}, font:{{color:'white'}}}},
                phase3: {{color:{{background:'#4caf50', border:'#1b5e20'}}, font:{{color:'white'}}}},
                phase4: {{color:{{background:'#e91e63', border:'#880e4f'}}, font:{{color:'white'}}}}
            }},
            edges: {{
                smooth: {{ type: 'cubicBezier', forceDirection: 'vertical', roundness: 0.4 }},
                color: {{ color: '#666', highlight: '#fff' }},
                font: {{ color: '#aaa', size: 12, background: '#1e1e1e' }}
            }},
            nodes: {{
                font: {{ size: 14, face: 'Segoe UI' }},
                shadow: true,
                borderWidth: 2
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true
            }},
            physics: false // turn off physics for strict hierarchical layout
        }};
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>
"""

with open("LoTrinhThucChien/index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Generated complex hierarchical index.html")
