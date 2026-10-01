"""Anumaan: simulate 3 yrs of forward-post logistics, train demand + route-closure models,
backtest against a static reorder policy, write the page. Run: python train.py"""
import json, re, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, brier_score_loss

rng = np.random.default_rng(11)
T, TEST, CAP = 1096, 122, 12000
NT, SPLIT, PAD = T + 14, T - TEST, 20
DATES = pd.date_range("2023-10-01", periods=NT); NOW = pd.Timestamp("2026-10-01")
doy, dow = DATES.dayofyear.values, DATES.dayofweek.values
COMM = ["Rations", "POL", "Medical"]
XY = {"BD":[70,225],"AD-1":[290,330],"AD-2":[290,120],"F-01":[480,410],"F-02":[570,320],"F-03":[680,380],"F-04":[745,285],"F-05":[480,50],"F-06":[580,135],"F-07":[690,70],"F-08":[745,175]}
POSTS = [("F-01",3050,220,"AD-1",62,3),("F-02",3400,160,"AD-1",88,4),("F-03",3850,120,"AD-1",121,5),("F-04",4250,90,"AD-1",150,6),
         ("F-05",3200,260,"AD-2",70,3),("F-06",3700,140,"AD-2",109,5),("F-07",4100,100,"AD-2",134,5),("F-08",4600,70,"AD-2",168,6)]
LT = {p[0]: p[5] for p in POSTS}

# ---- simulate weather, closures, demand -------------------------------------------------
season = 0.5 + 0.5 * np.sin(2*np.pi*(doy-150)/365)
storm = np.zeros(NT)
for t in range(1, NT):
    storm[t] = 0.7*storm[t-1] + (rng.random() < 0.05+0.12*season[t]) * rng.gamma(3, 2.2)
W, Y, CL, OPS = {}, {}, {}, {}
for pid, alt, men, dep, km, L in POSTS:
    temp = 15-(alt-1000)*0.0065 + 11*np.sin(2*np.pi*(doy-110)/365) + rng.normal(0, 2.5, NT)
    pr = storm*np.exp(rng.normal(0, .35, NT))*(0.6+alt/5000)
    W[pid] = dict(fp=np.maximum(0, pr+rng.normal(0, 1.8, NT)), ft=temp+rng.normal(0, 1.5, NT))
    z = -6.6 + 0.22*pr + 0.0004*(alt-3000) + 0.012*km
    c = np.zeros(NT)
    for t in range(NT):
        c[t] = rng.random() < 1/(1+np.exp(-(z[t] + (1.4 if t and c[t-1] else 0))))
    CL[pid] = c
    ops = np.repeat(rng.integers(0, 3, NT//20+1), 20)[:NT]; OPS[pid] = ops
    wk = 1+0.04*(dow >= 5)
    mean = {"Rations": men*1.9*(1+4e-5*(alt-3000)),
            "POL": men*(1.45+3.5e-4*(alt-3000)) + men*0.06*np.maximum(0, 6-temp),
            "Medical": np.full(NT, men*0.11)}
    for cm, k in zip(COMM, (.08, .12, .2)):
        Y[pid, cm] = mean[cm]*(1+k*ops)*wk*np.exp(rng.normal(0, .05, NT))
SER = list(Y)

# ---- demand model: log-target gradient boosting, lags >= 14d (no leakage) -----------------
fr = []
for pid, alt, men, *_ in POSTS:
    for ci, cm in enumerate(COMM):
        y = pd.Series(Y[pid, cm]); y[T:] = np.nan
        fr.append(pd.DataFrame(dict(pid=pid, c=cm, t=np.arange(NT), y=y, comm=ci, alt=alt, men=men, dow=dow,
            sin=np.sin(2*np.pi*doy/365), cos=np.cos(2*np.pi*doy/365), temp_fc=W[pid]["ft"], ops=OPS[pid],
            lag14=y.shift(14), lag21=y.shift(21), lag28=y.shift(28),
            r7=y.shift(14).rolling(7).mean(), r28=y.shift(14).rolling(28).mean())))
df = pd.concat(fr).dropna(subset=["lag28", "r28"]).reset_index(drop=True)
F = ["comm","alt","men","dow","sin","cos","temp_fc","ops","lag14","lag21","lag28","r7","r28"]
tr, te = df[df.t < SPLIT], df[(df.t >= SPLIT) & (df.t < T)]
kw = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=24, categorical_features=["comm"], random_state=0)
mu_m = HGR(**kw).fit(tr[F], np.log(tr.y))
q_m = HGR(loss="quantile", quantile=0.9, **kw).fit(tr[F], np.log(tr.y))
df["mu"], df["q90"] = np.exp(mu_m.predict(df[F])), np.exp(q_m.predict(df[F]))
te = df[(df.t >= SPLIT) & (df.t < T)]
wape = lambda p: float(np.abs(te.y-p).sum()/te.y.sum())
met = dict(wape=wape(te.mu), wape_lag14=wape(te.lag14), wape_r28=wape(te.r28), bias=float((te.mu-te.y).sum()/te.y.sum()),
           cov90=float((te.y <= te.q90).mean()), test_days=TEST, rows=int((df.t < T).sum()), series=len(SER),
           comm={cm: float(np.abs(g.y-g.mu).sum()/g.y.sum()) for cm, g in te.groupby("c")})
nm = dict(comm="Commodity class", r7="7-day avg (lagged)", alt="Post altitude", r28="28-day avg (lagged)", lag14="Demand 14 days ago",
          temp_fc="Temperature", lag21="Demand 21 days ago", ops="Planned ops tempo", men="Troop strength", lag28="Demand 28 days ago", dow="Day of week")
pi = permutation_importance(mu_m, te[F], np.log(te.y), n_repeats=3, random_state=0, scoring="neg_mean_absolute_error")
imp = sorted(((nm.get(f, "Season"), float(v)) for f, v in zip(F, pi.importances_mean)), key=lambda x: -x[1])[:8]
imp = [[n, round(v/imp[0][1], 3)] for n, v in imp]

# ---- route-closure classifier on forecast weather --------------------------------------------
cr = []
for pid, alt, men, dep, km, L in POSTS:
    fp = pd.Series(W[pid]["fp"])
    cr.append(pd.DataFrame(dict(pid=pid, t=np.arange(NT), y=CL[pid], fp=fp, fp1=fp.shift(1), fp2=fp.shift(2), ft=W[pid]["ft"],
                                alt=alt, km=km, sin=np.sin(2*np.pi*doy/365), cos=np.cos(2*np.pi*doy/365))))
cd = pd.concat(cr).dropna().reset_index(drop=True); CF = ["fp","fp1","fp2","ft","alt","km","sin","cos"]
clf = HGC(max_iter=250, learning_rate=0.05, random_state=0).fit(cd[cd.t < SPLIT][CF], cd[cd.t < SPLIT].y)
cd["p"] = clf.predict_proba(cd[CF])[:, 1]; ct = cd[(cd.t >= SPLIT) & (cd.t < T)]
met.update(auc=float(roc_auc_score(ct.y, ct.p)), brier=float(brier_score_loss(ct.y, ct.p)), base=float(ct.y.mean()))
pad = lambda a: np.concatenate([a, np.full(PAD, a[-1])])
PC = {pid: pad(g.sort_values("t").p.values) for pid, g in cd.groupby("pid")}
CLo = {pid: np.concatenate([CL[pid], np.zeros(PAD)]) for pid in LT}
Q = {k: pad(g.sort_values("t").q90.values) for k, g in df.groupby(["pid", "c"])}
OFF = {k: int(df[(df.pid == k[0]) & (df.c == k[1])].t.min()) for k in SER}
q90 = lambda k, a, b: Q[k][a-OFF[k]:b-OFF[k]].sum()
mu = {k: g.sort_values("t").mu.values for k, g in df.groupby(["pid", "c"])}
m28 = {k: pd.Series(Y[k]).rolling(28).mean().values for k in SER}

# ---- backtest: static reorder vs forecast-driven, closure-aware dispatch -------------------------
def run(policy):
    stock = {k: m28[k][SPLIT-1]*8 for k in SER}; arr = {k: {} for k in SER}; dem = unm = so = 0; cov = []
    for d in range(SPLIT, T):
        cand = []
        for k in SER:
            stock[k] += arr[k].pop(d, 0); y = Y[k][d]; s = min(stock[k], y)
            stock[k] -= s; dem += y; unm += y-s; so += y > s+1e-9; cov.append(stock[k]/m28[k][d])
        for k in SER:
            L, inf = LT[k[0]], sum(arr[k].values())
            if policy == "static":
                trig, up = m28[k][d]*(L+3), m28[k][d]*(L+7)
            else:
                H = int(np.ceil(L+1.3*PC[k[0]][d+1:d+1+L].sum()))+2
                trig, up = q90(k, d+1, d+1+H), q90(k, d+1, d+H+6)
            if stock[k]+inf < trig: cand.append(((stock[k]+inf)/trig, k, up-stock[k]-inf))
        left = CAP
        for _, k, q in sorted(cand):
            q = min(q, left)
            if q <= 0: continue
            left -= q; L = LT[k[0]]; a = d+1+L+int(CLo[k[0]][d+1:d+1+L].sum()); arr[k][a] = arr[k].get(a, 0)+q
    return dict(fill=1-unm/dem, so=int(so), cover=float(np.mean(cov))), stock, arr
sim_s, stock0, arr0 = run("static"); sim_a, _, _ = run("anumaan")

# ---- today's picture from status-quo stock, then the dispatch plan -------------------------------
posts, series = [], []
for pid, alt, men, dep, km, L in POSTS:
    p7 = PC[pid][T:T+7]; ed = 1.3*PC[pid][T:T+L].sum(); sl = []
    for cm in COMM:
        k = (pid, cm); s = stock0[k]; inf = sum(arr0[k].values()); dq = Q[k][T-OFF[k]:T-OFF[k]+7].mean()
        cover = s/dq; slack = cover-L-ed; H = int(np.ceil(L+ed))+7
        need = max(0, q90(k, T, T+H)-s-inf); sl.append(slack)
        series.append(dict(pid=pid, comm=cm, stock=round(s), cover=round(cover, 1), slack=round(slack, 1), ed=round(ed, 2), L=L,
            st="CRITICAL" if slack < 1 else "WATCH" if slack < 5 else "ADEQUATE", need=round(need, -1),
            fc7=round(mu[k][T-OFF[k]:T-OFF[k]+7].sum())))
    ms = min(sl)
    posts.append(dict(id=pid, alt=alt, men=men, dep=dep, km=km, L=L, xy=XY[pid], p=[round(float(v), 2) for v in p7], slack=round(ms, 1),
                      st="CRITICAL" if ms < 1 else "WATCH" if ms < 5 else "ADEQUATE"))
hist = {}
for pid, *_ in POSTS:
    g = lambda col, a, b: [round(float(sum(df[(df.pid == pid) & (df.t == t)][col].sum() for _ in [0]))) for t in range(a, b)]
    hist[pid] = dict(h=[round(sum(Y[pid, c][t] for c in COMM)) for t in range(T-28, T)], f=g("mu", T, NT), q=g("q90", T, NT))
road, heli, disp = [CAP]*3, [1500]*3, []
for s in sorted((s for s in series if s["need"] > 0), key=lambda s: s["slack"]):
    L, rp, q = s["L"], float(PC[s["pid"]][T:T+s["L"]].mean()), s["need"]
    use_heli = s["slack"] < 0 and rp > .12 and q <= 1500; pool = heli if use_heli else road
    day = next((i for i in range(3) if pool[i] >= q), None)
    if day is not None: pool[day] -= q
    lead = 1 if use_heli else L+round(s["ed"])
    disp.append(dict(pid=s["pid"], comm=s["comm"], qty=q, slack=s["slack"], L=L, rp=round(rp, 2), ed=s["ed"],
        day=None if day is None else str((NOW+pd.Timedelta(days=day)).date()), mode="Helicopter" if use_heli else "Road convoy",
        eta=None if day is None else str((NOW+pd.Timedelta(days=day+lead)).date())))
D = dict(met=met, imp=imp, sim=dict(static=sim_s, anumaan=sim_a), posts=posts, series=series, hist=hist, disp=disp, cap=CAP, XY=XY,
         need=float(sum(s["need"] for s in series)))
json.dump(D, open("metrics.json", "w"), indent=1, default=float)
print(json.dumps(dict(met=met, sim=D["sim"], crit=sum(s["st"] == "CRITICAL" for s in series), planned=sum(d["day"] is not None for d in disp), n=len(disp)), indent=1, default=float))
