import json, re
h = open("template.html").read()
D = json.dumps(json.load(open("metrics.json")))
h = re.sub(r"const D=\{.*?\}(?=,\$=i=>)", lambda m: "const D=" + D, h, flags=re.S)
css = """
:root{--chip:#f3f4f1}
html,body{background:#fff}body{padding-top:calc(20px + env(safe-area-inset-top,0px));padding-bottom:calc(48px + env(safe-area-inset-bottom,0px))}
header{border-bottom:2px solid var(--ink)}.k{background:var(--chip);padding:10px 12px 12px}.k b{font-variant-numeric:tabular-nums}
.shock{display:flex;flex-wrap:wrap;align-items:center;gap:8px 16px;margin:0 0 22px;padding:12px 14px;border:2px solid var(--ink);background:var(--chip)}
.shock label{font:700 22px 'Barlow Condensed',sans-serif;text-transform:uppercase;letter-spacing:.04em}.shock input{flex:1 1 220px;accent-color:var(--rust);height:28px}
.shock .num{font-size:22px;min-width:3.5ch}tr:hover td{background:var(--chip)}.b{background:var(--chip)}
.sc{max-width:100%}.post circle{transition:stroke-width .12s}.post:hover circle{stroke-width:5}
"""
h = h.replace("</style>", css + "</style>")
h = h.replace('<div class="grid">', '<div class="shock"><label for="k">Weather shock</label><input id="k" type="range" min="0.5" max="4" step="0.1" value="1"><b class="num" id="kv">1.0&times;</b><span class="mute">Scales closure risk on every route. Cover, status and map update live; the dispatch table below stays at today\'s forecast.</span></div>\n<div class="grid">', 1)
js = """
const base=P.map(p=>p.p.slice()),cur={id:null},o=sel;sel=id=>{cur.id=id;o(id)};
function shock(k){P.forEach((p,i)=>{p.p=base[i].map(v=>Math.min(1,v*k));const e=1.3*p.p.slice(0,p.L).reduce((a,b)=>a+b,0),r=D.series.filter(s=>s.pid==p.id);
r.forEach(s=>{s.slack=+(s.cover-p.L-e).toFixed(1);s.st=s.slack<1?'CRITICAL':s.slack<5?'WATCH':'ADEQUATE'});p.slack=Math.min(...r.map(s=>s.slack));p.st=p.slack<1?'CRITICAL':p.slack<5?'WATCH':'ADEQUATE'});
document.querySelectorAll('.k b')[3].textContent=D.series.filter(s=>s.st=='CRITICAL').length+' of 24';map();sel(cur.id||P[0].id)}
$('k').oninput=e=>{const k=+e.target.value;$('kv').innerHTML=k.toFixed(1)+'&times;';shock(k)};
"""
h = h.replace("</script></body>", js + "</script></body>")
h = h.replace("Backtest: 122 unseen days", "Backtest: 122 unseen days (held out)")
open("anumaan.html", "w").write(h)
