import os
import json

nodes = []
edges = []
current_id = 10

phases = {
    "01_NenTang_Python_Toan": {"id": 1, "label": "Phần 1: Nền Tảng Python & Toán", "group": "phase1", "level": 1},
    "02_PhanTichDuLieu_DataScience": {"id": 2, "label": "Phần 2: Xử Lý Dữ Liệu (Data Science)", "group": "phase2", "level": 2},
    "03_HocMay_MachineLearning": {"id": 3, "label": "Phần 3: Học Máy (Machine Learning)", "group": "phase3", "level": 3},
    "04_HocSau_AI_ThucChien": {"id": 4, "label": "Phần 4: AI & Deep Learning", "group": "phase4", "level": 4}
}

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

edges.append({"from": 1, "to": 2, "arrows": "to", "label": "Cung cấp nền tảng", "width": 3})
edges.append({"from": 2, "to": 3, "arrows": "to", "label": "Cung cấp dữ liệu", "width": 3})
edges.append({"from": 3, "to": 4, "arrows": "to", "label": "Cung cấp mô hình", "width": 3})

for phase_dir in sorted(phases.keys()):
    path = os.path.join("LoTrinhThucChien", phase_dir)
    if os.path.isdir(path):
        files = sorted([f for f in os.listdir(path) if f.endswith(".md")])
        root_id = phase_roots[phase_dir]
        level = phases[phase_dir]["level"]
        for f in files:
            name = f[:-3]
            short_name = name[:30] + "..." if len(name) > 30 else name
            nodes.append({
                "id": current_id,
                "label": short_name,
                "title": name,
                "group": phases[phase_dir]["group"],
                "shape": "dot",
                "size": 10,
                "level": level
            })
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
    <title>Lộ Trình Học AI/ML Thực Chiến (200 Bài Cốt Lõi)</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ font-family: sans-serif; background-color: #121212; color: #ffffff; padding: 20px; margin: 0; }}
        h1 {{ text-align: center; color: #00e676; }}
        #mynetwork {{ width: 100%; height: 800px; border: 1px solid #333; background-color: #1e1e1e; border-radius: 12px; }}
    </style>
</head>
<body>
    <h1>🧠 Lộ Trình Học AI & Machine Learning Thực Chiến</h1>
    <div id="mynetwork"></div>
    <script type="text/javascript">
        var nodes = new vis.DataSet({json.dumps(nodes)});
        var edges = new vis.DataSet({json.dumps(edges)});
        var container = document.getElementById('mynetwork');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
            layout: {{ hierarchical: {{ direction: 'UD', sortMethod: 'directed', nodeSpacing: 100, levelSeparation: 250 }} }},
            groups: {{
                phase1: {{color:{{background:'#ff9800', border:'#e65100'}}, font:{{color:'white'}}}},
                phase2: {{color:{{background:'#2196f3', border:'#0d47a1'}}, font:{{color:'white'}}}},
                phase3: {{color:{{background:'#4caf50', border:'#1b5e20'}}, font:{{color:'white'}}}},
                phase4: {{color:{{background:'#e91e63', border:'#880e4f'}}, font:{{color:'white'}}}}
            }},
            edges: {{ smooth: {{ type: 'cubicBezier' }}, color: '#666' }},
            nodes: {{ font: {{ size: 12 }}, shadow: true, borderWidth: 1 }},
            physics: false
        }};
        var network = new vis.Network(container, data, options);
    </script>
</body>
</html>
"""

with open("LoTrinhThucChien/index.html", "w", encoding="utf-8") as f:
    f.write(html_content)
