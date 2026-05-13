"""Veri Akışı sayfası — SSE tabanlı gerçek zamanlı izleme."""
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Veri Akışı", page_icon="🌊", layout="wide")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from theme import apply_theme, section  # noqa: E402
from data_loader import get_layer_stats  # noqa: E402

apply_theme()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown("### ⚡ Canlı İzleme")
    st.markdown(
        '<p style="color:#10B981;font-size:0.8rem">● SSE — sayfa yenilemesiz</p>',
        unsafe_allow_html=True,
    )
    if st.button("🔄 Katman istatistiklerini yenile", use_container_width=True):
        get_layer_stats.clear()
        st.rerun()

st.markdown("# 🌊 Veri Akışı")
st.markdown(
    '<p style="color:#94A3B8">Kafka\'dan ML\'e: 3 katmanlı Delta Lake mimarisi.</p>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Canlı SSE bileşeni ────────────────────────────────────────────────────────
section("🟢 Canlı İzleme", "WebSocket tabanlı anlık veri — sayfa yenilenmez")

_LIVE_HTML = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:transparent;color:#E2E8F0;font-size:13px}
.row{display:flex;gap:10px;margin-bottom:12px}
.card{flex:1;background:rgba(15,23,42,0.6);border-radius:12px;
      border:1px solid rgba(99,102,241,0.18);padding:14px 16px}
/* Kafka card */
.kafka-inner{display:flex;align-items:center;gap:14px}
.k-dot{width:11px;height:11px;border-radius:50%;flex-shrink:0;background:#64748B}
.k-info{flex:1}.k-label{font-weight:700;font-size:14px}
.k-sub{color:#64748B;font-size:11px;margin-top:2px}
.k-right{text-align:right}
.k-count{font-size:22px;font-weight:700;color:#F1F5F9}
.k-rate{color:#64748B;font-size:11px}
/* Layer cards */
.lcard{display:flex;align-items:center;gap:10px;
       background:rgba(15,23,42,0.5);border-radius:10px;
       border-left:3px solid;padding:10px 12px;flex:1}
.licon{font-size:20px}
.lname{font-weight:600;font-size:13px;color:#F1F5F9}
.lstatus{font-size:11px;margin-top:2px}
.lmeta{margin-left:auto;text-align:right}
.lmeta-lbl{color:#64748B;font-size:10px}
.lmeta-val{font-size:12px;font-weight:600;color:#E2E8F0}
.lrows{font-size:10px;color:#64748B}
/* Animation */
@keyframes pf{
  0%{left:-8px;opacity:0}15%{opacity:1}85%{opacity:1}
  100%{left:calc(100% + 8px);opacity:0}}
@keyframes glow{
  0%,100%{box-shadow:0 0 4px 2px var(--g,rgba(255,255,255,.2))}
  50%{box-shadow:0 0 18px 7px var(--g,rgba(255,255,255,.2))}}
.af{display:flex;align-items:center;background:rgba(15,23,42,0.5);
    border-radius:14px;border:1px solid rgba(99,102,241,.15);padding:18px 10px}
.afbox{display:flex;flex-direction:column;align-items:center;justify-content:center;
       width:96px;height:84px;border-radius:12px;font-weight:700;font-size:12px;
       flex-shrink:0;position:relative;z-index:2;transition:opacity .4s}
.afbox.w{animation:glow 1.4s ease-in-out infinite}
.pipe{position:relative;flex:1;height:5px;border-radius:3px;overflow:visible;margin:0 -1px}
.dot{position:absolute;top:50%;transform:translateY(-50%);
     width:9px;height:9px;border-radius:50%;animation:pf linear infinite}
/* Time series */
canvas{display:block;border-radius:10px;width:100%}
.legend{display:flex;gap:14px;margin-bottom:6px}
.li{display:flex;align-items:center;gap:5px;font-size:11px;color:#94A3B8}
.ld{width:12px;height:3px;border-radius:2px}
/* Connection badge */
.badge{display:inline-flex;align-items:center;gap:5px;
       font-size:11px;color:#64748B;margin-bottom:10px}
.bdot{width:7px;height:7px;border-radius:50%;background:#64748B;transition:background .3s}
.bdot.on{background:#10B981}
.bdot.warn{background:#F59E0B}
.sec{color:#94A3B8;font-size:10px;text-transform:uppercase;
     letter-spacing:.08em;font-weight:600;margin-bottom:8px}
</style></head><body>

<div class="badge"><div class="bdot" id="bd"></div><span id="bl">Bağlanıyor…</span></div>

<!-- Kafka -->
<div class="sec">Kafka Producer</div>
<div class="card" style="margin-bottom:12px">
  <div class="kafka-inner">
    <div class="k-dot" id="kd"></div>
    <div class="k-info">
      <div class="k-label" id="kl">—</div>
      <div class="k-sub">iot-network-traffic</div>
    </div>
    <div class="k-right">
      <div class="k-count" id="kc">—</div>
      <div class="k-rate" id="kr">— msg/s</div>
    </div>
  </div>
</div>

<!-- Layers -->
<div class="sec">Streaming Katmanlar</div>
<div class="row">
  <div class="lcard" id="lcard-Bronze" style="border-color:#CD7F32">
    <div class="licon">🪣</div>
    <div><div class="lname">Bronze</div><div class="lstatus" id="ls-Bronze">—</div></div>
    <div class="lmeta">
      <div class="lmeta-lbl">Son yazım</div>
      <div class="lmeta-val" id="lm-Bronze">—</div>
      <div class="lrows" id="lr-Bronze"></div>
    </div>
  </div>
  <div class="lcard" id="lcard-Silver" style="border-color:#C0C0C0">
    <div class="licon">🧹</div>
    <div><div class="lname">Silver</div><div class="lstatus" id="ls-Silver">—</div></div>
    <div class="lmeta">
      <div class="lmeta-lbl">Son yazım</div>
      <div class="lmeta-val" id="lm-Silver">—</div>
      <div class="lrows" id="lr-Silver"></div>
    </div>
  </div>
  <div class="lcard" id="lcard-Gold" style="border-color:#FFD700">
    <div class="licon">✨</div>
    <div><div class="lname">Gold</div><div class="lstatus" id="ls-Gold">—</div></div>
    <div class="lmeta">
      <div class="lmeta-lbl">Son yazım</div>
      <div class="lmeta-val" id="lm-Gold">—</div>
      <div class="lrows" id="lr-Gold"></div>
    </div>
  </div>
</div>

<!-- Animation -->
<div class="sec">Canlı Veri Akışı</div>
<div class="af" style="margin-bottom:12px">
  <div class="afbox" id="af-Kafka" style="background:rgba(99,102,241,.12);border:2px solid #6366F1;color:#A5B4FC;opacity:.4;--g:rgba(99,102,241,.65)">
    <div style="font-size:22px">📡</div><div style="margin-top:4px">Kafka</div>
    <div style="font-size:10px;margin-top:2px" id="af-st-Kafka">—</div>
  </div>
  <div class="pipe" id="pipe-K" style="background:rgba(205,127,50,.12)"></div>
  <div class="afbox" id="af-Bronze" style="background:rgba(205,127,50,.12);border:2px solid #CD7F32;color:#D4904E;opacity:.4;--g:rgba(205,127,50,.65)">
    <div style="font-size:22px">🪣</div><div style="margin-top:4px">Bronze</div>
    <div style="font-size:10px;margin-top:2px" id="af-st-Bronze">—</div>
  </div>
  <div class="pipe" id="pipe-B" style="background:rgba(192,192,192,.12)"></div>
  <div class="afbox" id="af-Silver" style="background:rgba(192,192,192,.08);border:2px solid #C0C0C0;color:#C0C0C0;opacity:.4;--g:rgba(192,192,192,.65)">
    <div style="font-size:22px">🧹</div><div style="margin-top:4px">Silver</div>
    <div style="font-size:10px;margin-top:2px" id="af-st-Silver">—</div>
  </div>
  <div class="pipe" id="pipe-S" style="background:rgba(255,215,0,.12)"></div>
  <div class="afbox" id="af-Gold" style="background:rgba(255,215,0,.08);border:2px solid #FFD700;color:#FFD700;opacity:.4;--g:rgba(255,215,0,.65)">
    <div style="font-size:22px">✨</div><div style="margin-top:4px">Gold</div>
    <div style="font-size:10px;margin-top:2px" id="af-st-Gold">—</div>
  </div>
</div>

<!-- Time series -->
<div class="sec">Satır Sayısı Geçmişi</div>
<div class="legend">
  <div class="li"><div class="ld" style="background:#CD7F32"></div>Bronze</div>
  <div class="li"><div class="ld" style="background:#C0C0C0"></div>Silver</div>
  <div class="li"><div class="ld" style="background:#FFD700"></div>Gold</div>
</div>
<canvas id="ts" height="180"></canvas>

<script>
const API = "http://localhost:5001";
const MAX = 120;
const hist = {Bronze:[], Silver:[], Gold:[]};
const activeDots = {};
let reconnectCount = 0;
let lastEventTime = 0;

// ── Canvas chart ──────────────────────────────────────────────────────────────
const cv = document.getElementById("ts");
const cx = cv.getContext("2d");

function drawChart() {
  const W = cv.offsetWidth || 640, H = 180;
  cv.width = W;
  cx.clearRect(0,0,W,H);
  cx.fillStyle="rgba(15,23,42,0.5)";
  if(cx.roundRect){cx.beginPath();cx.roundRect(0,0,W,H,10);cx.fill();}
  else{cx.fillRect(0,0,W,H);}

  const pl=52,pr=10,pt=10,pb=28, cw=W-pl-pr, ch=H-pt-pb;
  const all=[...hist.Bronze,...hist.Silver,...hist.Gold].filter(v=>v>0);
  const mx=all.length?Math.max(...all):1;
  const n=Math.max(hist.Bronze.length,hist.Silver.length,hist.Gold.length,2);

  // grid
  for(let i=0;i<=4;i++){
    const y=pt+ch*i/4;
    cx.strokeStyle="rgba(100,116,139,.2)";cx.lineWidth=1;
    cx.beginPath();cx.moveTo(pl,y);cx.lineTo(pl+cw,y);cx.stroke();
    const v=Math.round(mx*(4-i)/4);
    cx.fillStyle="#64748B";cx.font="9px sans-serif";cx.textAlign="right";
    cx.fillText(v>=1000?(v/1000).toFixed(1)+"k":v, pl-4, y+3);
  }

  // lines
  [["Bronze","#CD7F32"],["Silver","#C0C0C0"],["Gold","#FFD700"]].forEach(([k,c])=>{
    const d=hist[k]; if(d.length<2)return;
    cx.beginPath();cx.strokeStyle=c;cx.lineWidth=2;cx.lineJoin="round";
    d.forEach((v,i)=>{
      const x=pl+(i/(n-1))*cw, y=pt+ch-(v/mx)*ch;
      i===0?cx.moveTo(x,y):cx.lineTo(x,y);
    });
    cx.stroke();
  });

  cx.fillStyle="#64748B";cx.font="9px sans-serif";cx.textAlign="center";
  cx.fillText("son "+n+"s",pl+cw/2,H-8);
}

// ── Dot animation ─────────────────────────────────────────────────────────────
function clearDots(pipeId){
  const p=document.getElementById(pipeId);
  if(!p)return;
  p.querySelectorAll(".dot").forEach(d=>d.remove());
  if(activeDots[pipeId]){clearInterval(activeDots[pipeId]);delete activeDots[pipeId];}
}
function addDots(pipeId,color,count=3){
  clearDots(pipeId);
  const p=document.getElementById(pipeId);if(!p)return;
  const dur=1.2;
  for(let i=0;i<count;i++){
    const d=document.createElement("div");d.className="dot";
    d.style.cssText="background:"+color+";animation-duration:"+dur+"s;animation-delay:"+(i*dur/count).toFixed(2)+"s";
    p.appendChild(d);
  }
}

// ── Status helpers ────────────────────────────────────────────────────────────
function layerState(info){
  if(!info||!info.exists) return {text:"Oluşturulmadı",color:"#64748B",sym:"⚫"};
  if(info.active)          return {text:"Yazıyor",      color:"#10B981",sym:"🟢"};
  if(info.has_data)        return {text:"Tamamlandı",   color:"#3B82F6",sym:"🔵"};
  return                          {text:"Bekliyor",     color:"#F59E0B",sym:"🟡"};
}

function setNode(name, info, pipeSuffix, dotColor, pipeActive, pipeIdle){
  const el=document.getElementById("af-"+name);
  const st=document.getElementById("af-st-"+name);
  const pipe=pipeSuffix?document.getElementById("pipe-"+pipeSuffix):null;
  if(!el)return;
  const s=layerState(info);
  st.textContent=s.sym+" "+s.text; st.style.color=s.color;
  if(info&&info.active){
    el.style.opacity="1"; el.classList.add("w");
    if(pipe){pipe.style.background=pipeActive; addDots("pipe-"+pipeSuffix,dotColor);}
  }else if(info&&info.has_data){
    el.style.opacity="1"; el.classList.remove("w");
    if(pipe){pipe.style.background=pipeIdle; clearDots("pipe-"+pipeSuffix);}
  }else{
    el.style.opacity="0.4"; el.classList.remove("w");
    if(pipe){pipe.style.background=pipeIdle; clearDots("pipe-"+pipeSuffix);}
  }
}

// ── SSE ───────────────────────────────────────────────────────────────────────
function connect(){
  const es=new EventSource(API+"/api/stream");

  es.onopen=()=>{
    reconnectCount=0;
    document.getElementById("bd").className="bdot on";
    document.getElementById("bl").textContent="Canlı bağlantı aktif";
  };

  es.onmessage=function(e){
    lastEventTime=Date.now();
    const d=JSON.parse(e.data);
    const kafka=d.kafka, layers=d.layers;

    // Kafka card — bağlantı durumunu connected flag'den al
    const connected=kafka.connected||false;
    const hasData=kafka.total>0;
    const kDot=document.getElementById("kd");
    const kLabel=document.getElementById("kl");
    if(connected&&hasData){
      kDot.style.background="#10B981";
      kLabel.textContent="Bağlı · Veri akıyor";
    }else if(connected&&!hasData){
      kDot.style.background="#F59E0B";
      kLabel.textContent="Bağlı · Veri bekleniyor";
    }else{
      kDot.style.background="#EF4444";
      kLabel.textContent="Kafka bağlantısı kuruluyor…";
    }
    document.getElementById("kc").textContent=kafka.total.toLocaleString();
    document.getElementById("kr").textContent=kafka.rate+" msg/s";

    // Layer cards + animation nodes
    ["Bronze","Silver","Gold"].forEach(n=>{
      const info=layers[n]||{};
      const s=layerState(info);
      const lsEl=document.getElementById("ls-"+n);
      const lmEl=document.getElementById("lm-"+n);
      const lrEl=document.getElementById("lr-"+n);
      if(lsEl){lsEl.textContent=s.sym+" "+s.text; lsEl.style.color=s.color;}
      if(lmEl) lmEl.textContent=info.last_mod||"—";
      if(lrEl) lrEl.textContent=info.rows>0?info.rows.toLocaleString()+" satır":"";
    });

    // Animation
    // Kafka node glow — pipe-K is driven by Bronze.active (see setNode below)
    const kNode=document.getElementById("af-Kafka");
    const kSt=document.getElementById("af-st-Kafka");
    if(kNode){
      kNode.style.opacity=connected?"1":"0.4";
      connected?kNode.classList.add("w"):kNode.classList.remove("w");
      kSt.textContent=connected?"● Aktif":"● Kapalı";
      kSt.style.color=connected?"#10B981":"#EF4444";
    }
    // Each pipe animates when the DOWNSTREAM node is being written to:
    //   pipe-K → Bronze active (data flowing INTO Bronze)
    //   pipe-B → Silver active (data flowing INTO Silver)
    //   pipe-S → Gold active  (data flowing INTO Gold)
    setNode("Bronze",layers["Bronze"],"K","#E8964A","rgba(205,127,50,.45)","rgba(205,127,50,.12)");
    setNode("Silver",layers["Silver"],"B","#C0C0C0","rgba(192,192,192,.45)","rgba(192,192,192,.12)");
    setNode("Gold",  layers["Gold"],  "S","#FFD700","rgba(255,215,0,.45)","rgba(255,215,0,.12)");

    // Time series
    ["Bronze","Silver","Gold"].forEach(n=>{
      const r=(layers[n]||{}).rows||0;
      hist[n].push(r); if(hist[n].length>MAX)hist[n].shift();
    });
    if(hist.Bronze.some(v=>v>0)||hist.Silver.some(v=>v>0)||hist.Gold.some(v=>v>0))
      drawChart();
  };

  es.onerror=()=>{
    es.close();
    reconnectCount++;
    const delay=Math.min(reconnectCount*2,10);
    document.getElementById("bd").className="bdot warn";
    document.getElementById("bl").textContent=
      "Yeniden bağlanılıyor… (deneme "+reconnectCount+", "+delay+"s)";
    setTimeout(connect, delay*1000);
  };
}

connect();
window.addEventListener("resize", drawChart);

// Bağlantı sağlık kontrolü: 10s boyunca SSE event gelmezse uyar
setInterval(()=>{
  if(lastEventTime>0 && Date.now()-lastEventTime>10000){
    document.getElementById("bd").className="bdot warn";
    document.getElementById("bl").textContent="Veri gecikmesi algılandı…";
  }
},5000);
</script></body></html>
"""

components.html(_LIVE_HTML, height=680)

st.markdown("---")

# ── Katman detayları (statik, manuel yenile) ──────────────────────────────────
layer_df = get_layer_stats()

section("📦 Katman Detayları", "Her katmanın rolü ve mevcut durumu")

layer_meta = {
    "Bronze": {
        "icon": "🪣", "color": "#CD7F32",
        "purpose": "Kafka'dan gelen ham JSON mesajlarını şemasız olarak korur.",
        "writes": "writeStream → Delta · append-only",
        "schema": "json_payload (string) + Kafka metadata",
        "tech": ["Spark Structured Streaming", "Delta Lake", "Append mode"],
    },
    "Silver": {
        "icon": "🧹", "color": "#C0C0C0",
        "purpose": "Şema uygulanır, null/duplike/format hataları temizlenir.",
        "writes": "Bronze → from_json + filtre → Delta",
        "schema": "Edge-IIoTset şeması (60+ kolon) + Attack_type label",
        "tech": ["from_json + schema inference", "Null/dup filtreleme", "Append mode"],
    },
    "Gold": {
        "icon": "✨", "color": "#FFD700",
        "purpose": "ML'e hazır feature tablosu — 5 yeni türetilmiş özellik dahil.",
        "writes": "Silver → FeatureEngineer → ml_ready_compact",
        "schema": "Numerik feature'lar + label_indexed (StringIndexer)",
        "tech": ["FeatureEngineer", "5 yeni feature", "StringIndexer", "VectorAssembler"],
    },
}

for _, row in layer_df.iterrows():
    meta = layer_meta.get(row["layer"], {})
    color = meta.get("color", "#6366F1")
    try:
        rows_v = f"{int(row['rows']):,}"
    except (ValueError, TypeError):
        rows_v = "—"
    try:
        cols_v = str(int(row["columns"]))
    except (ValueError, TypeError):
        cols_v = "—"
    st.markdown(
        f"""
        <div class="info-card" style="border-left:4px solid {color}">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:14px">
                <div style="flex:2;min-width:280px">
                    <h4>{meta.get('icon','📦')} {row['layer']} Layer</h4>
                    <p>{meta.get('purpose','')}</p>
                    <p><b style="color:#94A3B8">Yazım:</b> <code>{meta.get('writes','')}</code></p>
                    <p><b style="color:#94A3B8">Şema:</b> {meta.get('schema','')}</p>
                    <div style="margin-top:8px">
                        {''.join(f'<span class="badge">{t}</span>' for t in meta.get('tech', []))}
                    </div>
                </div>
                <div style="flex:1;min-width:200px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
                    <div style="background:rgba(15,23,42,0.5);padding:10px;border-radius:10px;text-align:center">
                        <div style="color:#64748B;font-size:0.7rem;text-transform:uppercase">Satır</div>
                        <div style="color:#F1F5F9;font-weight:700;font-size:1.05rem">{rows_v}</div>
                    </div>
                    <div style="background:rgba(15,23,42,0.5);padding:10px;border-radius:10px;text-align:center">
                        <div style="color:#64748B;font-size:0.7rem;text-transform:uppercase">Kolon</div>
                        <div style="color:#F1F5F9;font-weight:700;font-size:1.05rem">{cols_v}</div>
                    </div>
                    <div style="background:rgba(15,23,42,0.5);padding:10px;border-radius:10px;text-align:center">
                        <div style="color:#64748B;font-size:0.7rem;text-transform:uppercase">Boyut</div>
                        <div style="color:#F1F5F9;font-weight:700;font-size:1.05rem">{row['size_mb']:.1f}<span style="font-size:0.7rem;color:#64748B"> MB</span></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

