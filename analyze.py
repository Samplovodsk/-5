"""
Кластеризация клиентских данных.
Алгоритмы: K-Means, Иерархическая кластеризация.
Метрики: метод локтя, силуэтный коэффициент.
Результат: report.html со встроенными графиками.
"""
import csv, json, os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

CSV_PATH = os.path.join(os.path.dirname(__file__), "PP13_ISP23V_clustering.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "report.html")

def load_data(path):
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    headers = list(rows[0].keys())
    data = []
    for row in rows:
        try:
            data.append([float(row[h]) for h in headers])
        except ValueError:
            pass
    return headers, np.array(data)

headers, X_raw = load_data(CSV_PATH)
n_samples = len(X_raw)

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

corr = np.corrcoef(X_raw.T)

K_RANGE = range(2, 9)
inertias, silhouettes = [], []
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    inertias.append(round(km.inertia_, 2))
    silhouettes.append(round(silhouette_score(X, labels), 4))

best_k = list(K_RANGE)[silhouettes.index(max(silhouettes))]

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_labels = km_final.fit_predict(X)

agg = AgglomerativeClustering(n_clusters=best_k)
agg_labels = agg.fit_predict(X)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

cluster_stats = []
for c in range(best_k):
    mask = km_labels == c
    group = X_raw[mask]
    stat = {"cluster": c, "count": int(mask.sum())}
    for i, h in enumerate(headers):
        stat[h + "_mean"] = round(float(group[:, i].mean()), 2)
        stat[h + "_min"]  = round(float(group[:, i].min()), 2)
        stat[h + "_max"]  = round(float(group[:, i].max()), 2)
    cluster_stats.append(stat)

COLORS = ["#4d80c9", "#f1892d", "#5ae150", "#e93841", "#ffd19d", "#7d3ebf", "#eb6c82", "#1e8a4c"]

scatter_km  = [{"x": round(float(X_2d[i,0]),3), "y": round(float(X_2d[i,1]),3), "c": int(km_labels[i])}  for i in range(n_samples)]
scatter_agg = [{"x": round(float(X_2d[i,0]),3), "y": round(float(X_2d[i,1]),3), "c": int(agg_labels[i])} for i in range(n_samples)]

js_data = {
    "headers": headers,
    "n_samples": n_samples,
    "best_k": best_k,
    "k_range": list(K_RANGE),
    "inertias": inertias,
    "silhouettes": silhouettes,
    "km_scatter": scatter_km,
    "agg_scatter": scatter_agg,
    "cluster_stats": cluster_stats,
    "corr": [[round(float(v),3) for v in row] for row in corr],
    "colors": COLORS,
    "pca_var": [round(float(v)*100, 1) for v in pca.explained_variance_ratio_],
    "sil_best": max(silhouettes),
}

subtitle = "K-Means + Иерархическая кластеризация | Оптимальное k = " + str(best_k) + " | Силуэт = " + str(round(max(silhouettes), 4))
js_json  = json.dumps(js_data, ensure_ascii=False)

# HTML собирается конкатенацией — нет конфликта с JS {}
out = []
out.append("""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Кластеризация клиентов</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',sans-serif;background:#0f1117;color:#e2e8f0;padding:32px 16px}
h1{color:#e2e8f0;font-size:1.5rem;text-align:center;margin-bottom:8px;font-weight:600}
.sub{color:#8892a4;text-align:center;font-size:.88rem;margin-bottom:32px}
h2{color:#e2e8f0;font-size:.9rem;margin-bottom:16px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px;max-width:1200px;margin:0 auto 24px}
.card{background:#1a1d27;border:1px solid #2d3348;border-radius:8px;padding:22px}
.kpi-row{display:flex;gap:16px;flex-wrap:wrap;max-width:1200px;margin:0 auto 24px}
.kpi{flex:1;min-width:140px;background:#1a1d27;border:1px solid #2d3348;border-radius:8px;padding:18px;text-align:center}
.kpi-val{font-size:1.8rem;font-weight:700;color:#e2e8f0}
.kpi-lbl{font-size:.72rem;color:#8892a4;margin-top:4px;text-transform:uppercase;letter-spacing:.04em}
table{width:100%;border-collapse:collapse;font-size:.82rem}
thead th{color:#8892a4;padding:6px 8px;border-bottom:1px solid #2d3348;text-align:left;font-weight:500;text-transform:uppercase;font-size:.72rem;letter-spacing:.04em}
td{padding:7px 8px;border-bottom:1px solid #1e2235}
tr:last-child td{border-bottom:none}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:middle}
canvas{max-height:300px}
.full{max-width:1200px;margin:0 auto 24px}
</style>
</head>
<body>
<h1>Кластеризация клиентов</h1>
""")
out.append('<p class="sub">' + subtitle + '</p>\n')
out.append("""
<div class="kpi-row" id="kpi"></div>
<div class="grid">
  <div class="card"><h2>Метод локтя (инерция)</h2><canvas id="elbow"></canvas></div>
  <div class="card"><h2>Силуэтный коэффициент</h2><canvas id="sil"></canvas></div>
</div>
<div class="grid">
  <div class="card"><h2>K-Means — 2D (PCA)</h2><canvas id="sc-km"></canvas></div>
  <div class="card"><h2>Иерархическая — 2D (PCA)</h2><canvas id="sc-agg"></canvas></div>
</div>
<div class="grid">
  <div class="card"><h2>Корреляционная матрица</h2><canvas id="corr"></canvas></div>
  <div class="card"><h2>Средние по кластерам</h2><canvas id="bar"></canvas></div>
</div>
<div class="full card"><h2>Характеристики кластеров (K-Means)</h2><div id="tbl"></div></div>
<script>
const D = """)
out.append(js_json)
out.append(""";
const C = D.colors;
const al = (h,a) => h + Math.round(a*255).toString(16).padStart(2,'0');

document.getElementById('kpi').innerHTML = [
  ['Наблюдений',D.n_samples],['Признаков',D.headers.length],
  ['Кластеров',D.best_k],['Силуэт',D.sil_best.toFixed(4)],
  ['PCA',D.pca_var[0]+'%+'+D.pca_var[1]+'%']
].map(([l,v])=>'<div class="kpi"><div class="kpi-val">'+v+'</div><div class="kpi-lbl">'+l+'</div></div>').join('');

new Chart(document.getElementById('elbow'),{
  type:'line',
  data:{labels:D.k_range,datasets:[{label:'Инерция',data:D.inertias,borderColor:C[0],backgroundColor:al(C[0],0.15),tension:.3,fill:true}]},
  options:{plugins:{legend:{labels:{color:'#aeb5bd'}}},scales:{x:{ticks:{color:'#aeb5bd'},grid:{color:'#2a2050'}},y:{ticks:{color:'#aeb5bd'},grid:{color:'#2a2050'}}}}
});

new Chart(document.getElementById('sil'),{
  type:'bar',
  data:{labels:D.k_range,datasets:[{label:'Силуэт',data:D.silhouettes,backgroundColor:D.k_range.map(k=>k===D.best_k?C[1]:al(C[0],0.7)),borderRadius:4}]},
  options:{plugins:{legend:{labels:{color:'#aeb5bd'}}},scales:{x:{ticks:{color:'#aeb5bd'},grid:{color:'#2a2050'}},y:{min:0,max:1,ticks:{color:'#aeb5bd'},grid:{color:'#2a2050'}}}}
});

function scatter(id,pts){
  const ds=[];
  for(let c=0;c<D.best_k;c++){
    ds.push({label:'Кластер '+c,data:pts.filter(p=>p.c===c).map(p=>({x:p.x,y:p.y})),backgroundColor:al(C[c],0.75),pointRadius:4});
  }
  new Chart(document.getElementById(id),{
    type:'scatter',data:{datasets:ds},
    options:{plugins:{legend:{labels:{color:'#aeb5bd'}}},scales:{x:{ticks:{color:'#aeb5bd'},grid:{color:'#2a2050'},title:{display:true,text:'PC1',color:'#aeb5bd'}},y:{ticks:{color:'#aeb5bd'},grid:{color:'#2a2050'},title:{display:true,text:'PC2',color:'#aeb5bd'}}}}
  });
}
scatter('sc-km',D.km_scatter);
scatter('sc-agg',D.agg_scatter);

(function(){
  const n=D.headers.length, ds=[];
  for(let i=0;i<n;i++) for(let j=0;j<n;j++){
    const v=D.corr[i][j], a=Math.abs(v);
    ds.push({label:'',data:[{x:j,y:i,r:12,v}],backgroundColor:[v>=0?'rgba(77,128,201,'+a+')':'rgba(233,56,65,'+a+')'],borderColor:'transparent'});
  }
  new Chart(document.getElementById('corr'),{
    type:'bubble',data:{datasets:ds},
    options:{plugins:{legend:{display:false},tooltip:{callbacks:{label:ctx=>D.headers[Math.round(ctx.raw.x)]+'/'+D.headers[Math.round(ctx.raw.y)]+': '+ctx.raw.v}}},
      scales:{x:{min:-.5,max:n-.5,ticks:{callback:v=>D.headers[v]||'',color:'#aeb5bd'},grid:{color:'#2a2050'}},y:{min:-.5,max:n-.5,ticks:{callback:v=>D.headers[v]||'',color:'#aeb5bd'},grid:{color:'#2a2050'}}}}
  });
})();

(function(){
  const ds=D.headers.map((h,i)=>({label:h,data:D.cluster_stats.map(s=>s[h+'_mean']),backgroundColor:al(C[i%C.length],0.8),borderRadius:4}));
  new Chart(document.getElementById('bar'),{
    type:'bar',data:{labels:D.cluster_stats.map(s=>'Кластер '+s.cluster),datasets:ds},
    options:{plugins:{legend:{labels:{color:'#aeb5bd'}}},scales:{x:{ticks:{color:'#aeb5bd'},grid:{color:'#2a2050'}},y:{ticks:{color:'#aeb5bd'},grid:{color:'#2a2050'}}}}
  });
})();

(function(){
  const cols=['cluster','count',...D.headers.map(h=>h+'_mean')];
  const lbls=['Кластер','Кол-во',...D.headers.map(h=>h+' (среднее)')];
  let h='<table><thead><tr>'+lbls.map(l=>'<th>'+l+'</th>').join('')+'</tr></thead><tbody>';
  D.cluster_stats.forEach(s=>{
    h+='<tr>';
    cols.forEach(c=>{
      if(c==='cluster') h+='<td><span class="dot" style="background:'+C[s.cluster]+'"></span>Кластер '+s.cluster+'</td>';
      else h+='<td>'+s[c]+'</td>';
    });
    h+='</tr>';
  });
  document.getElementById('tbl').innerHTML=h+'</tbody></table>';
})();
</script></body></html>""")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("".join(out))

print("Done. Samples:", n_samples, "| best k:", best_k, "| silhouette:", round(max(silhouettes), 4))
print("Report:", OUT_PATH)
