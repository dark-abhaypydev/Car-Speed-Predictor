# ============================================================
# CAR DASHBOARD - SPEED & RPM PREDICTION
# Machine Learning Algorithm: Linear Regression
# UI: Speedometer, tachometer, gear indicator, fixed throttle slider
# Run command: streamlit run app.py
# ============================================================

# ===================== 1. LIBRARIES =====================
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Car Speed Predictor", layout="wide")


# ===================== 2. DATASET =====================
def make_demo_data():
    """Demo data (jab tak apna CSV upload na karo)."""
    rng = np.random.default_rng(42)
    rows = []
    max_speed = {1: 40, 2: 70, 3: 105, 4: 145, 5: 190}

    for gear in range(1, 6):
        for throttle in range(0, 101, 2):
            for _ in range(5):
                speed = throttle / 100 * max_speed[gear] + rng.normal(0, 3)
                rpm = 900 + throttle * 50 - gear * 80 + rng.normal(0, 80)
                rows.append([throttle, gear, speed, rpm])

    return pd.DataFrame(rows, columns=["throttle", "gear", "speed", "rpm"])


def guess(cols, key):
    """Column ke naam se andaza lagao (throttle/gear/speed/rpm)."""
    for i, c in enumerate(cols):
        if key in str(c).lower():
            return i
    return 0


with st.sidebar:
    st.header("Dataset")
    up = st.file_uploader("Apna CSV upload karo (optional)", type="csv")
    unit = st.selectbox("Speed unit", ["km/h", "mph"])
    raw, cols_map = None, None

    if up is not None:
        try:
            raw = pd.read_csv(up)
            st.caption(f"{raw.shape[0]} rows, {raw.shape[1]} columns")
            st.dataframe(raw.head(), use_container_width=True)
            cols = list(raw.columns)

            thr_c = st.selectbox(
                "Throttle column", cols, index=guess(cols, "throttle")
            )
            gear_c = st.selectbox(
                "Gear column", cols, index=guess(cols, "gear")
            )
            spd_c = st.selectbox(
                "Speed column (target)", cols, index=guess(cols, "speed")
            )

            has_rpm_col = any("rpm" in str(c).lower() for c in cols)
            rpm_c = st.selectbox(
                "RPM column (optional)",
                ["None"] + cols,
                index=(1 + guess(cols, "rpm")) if has_rpm_col else 0,
            )
            cols_map = (thr_c, gear_c, spd_c, rpm_c)

        except Exception as e:
            st.error(f"CSV read nahi hui: {e}")
            raw = None


# ===================== 3. CLEANING =====================
def clean(df):
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna().drop_duplicates()
    df = df[df["speed"] >= 0]

    if "rpm" in df.columns:
        df = df[df["rpm"] >= 0]

    df["gear"] = df["gear"].round().astype(int)
    return df


if raw is not None and cols_map is not None:
    thr_c, gear_c, spd_c, rpm_c = cols_map
    data = pd.DataFrame({
        "throttle": raw[thr_c],
        "gear": raw[gear_c],
        "speed": raw[spd_c],
    })

    if rpm_c != "None":
        data["rpm"] = raw[rpm_c]

    df = clean(data)
    using_demo = False

    if len(df) < 10:
        st.sidebar.error(
            "Clean karne ke baad data kam bacha (10 rows se kam). "
            "Demo data use ho raha hai."
        )
        df = clean(make_demo_data())
        using_demo = True
else:
    df = clean(make_demo_data())
    using_demo = True

HAS_RPM = "rpm" in df.columns


# ===================== 4. FEATURES + 5. SPLIT + 6. REGRESSION =====================
@st.cache_resource
def train_model(df):
    X = df[["throttle", "gear"]]
    y = df[["speed", "rpm"]] if "rpm" in df.columns else df[["speed"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = r2_score(y_test["speed"], pred[:, 0])

    return model, score


model, score = train_model(df)


# ===================== 7. PREDICTION =====================
def predict(throttle, gear):
    inp = pd.DataFrame([[throttle, gear]], columns=["throttle", "gear"])
    out = model.predict(inp)[0]
    return max(out[0], 0)


coef = np.atleast_2d(model.coef_)
bias = np.atleast_1d(model.intercept_)

T_MAX = float(df["throttle"].max()) or 1.0
gears = sorted(df["gear"].unique().tolist())

_m = float(df["speed"].max()) * 1.15
for SPEED_STEP in (10, 20, 25, 50, 100):
    if _m / SPEED_STEP <= 13:
        break

SPEED_MAX = int(math.ceil(_m / SPEED_STEP) * SPEED_STEP)
_r = float(df["rpm"].max()) / 1000 * 1.1 if HAS_RPM else 8
RPM_MAX = max(4, int(math.ceil(_r)))

cfg = {
    "speed_w": [float(coef[0][0]), float(coef[0][1])],
    "speed_b": float(bias[0]),
    "rpm_w": [float(coef[1][0]), float(coef[1][1])] if HAS_RPM else None,
    "rpm_b": float(bias[1]) if HAS_RPM else None,
    "t_max": T_MAX,
    "gears": gears,
    "speed_max": SPEED_MAX,
    "speed_step": SPEED_STEP,
    "rpm_max": RPM_MAX,
    "rpm_step": 1 if RPM_MAX <= 10 else 2,
    "red_from": RPM_MAX - 2,
    "idle": float(df["rpm"].min()) if HAS_RPM else 800.0,
    "unit": unit,
}


# ===================== 8. DASHBOARD HTML + JS =====================
HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  *{box-sizing:border-box;margin:0;padding:0;-webkit-user-select:none;user-select:none}
  body{background:#05070b;font-family:'Segoe UI',Arial,sans-serif;color:#fff;overflow:hidden}
  .dash{max-width:1100px;margin:0 auto;padding:14px 10px;
        background:radial-gradient(ellipse at 50% 0%,#101a2c 0%,#05070b 70%);
        border-radius:26px;border:1px solid #1d2a44;box-shadow:0 0 40px #0a1a3a inset}
  .accent{height:5px;border-radius:3px;margin:0 auto 10px;width:46%;
          background:linear-gradient(90deg,#4aa8ff 0 33%,#1b3a9a 33% 66%,#d8232a 66%)}
  .cluster{display:grid;grid-template-columns:1fr 0.62fr 1fr;align-items:center;gap:4px}
  svg.g{width:100%;height:auto;display:block}
  .center{text-align:center}
  .digital{font-size:64px;font-weight:700;letter-spacing:1px;text-shadow:0 0 14px #4aa8ff}
  .unit{color:#8fb7ff;font-size:15px;letter-spacing:3px;margin-top:-6px}
  .gearbox{display:inline-block;margin-top:14px;border:2px solid #4aa8ff;border-radius:10px;
           padding:2px 20px;font-size:40px;font-weight:700;color:#ffb066;background:#0a1222}
  .glabel{color:#6f86ad;font-size:11px;letter-spacing:3px;margin-top:4px}
  .bar{height:8px;background:#111b2e;border-radius:6px;margin:16px 14px 0;overflow:hidden;border:1px solid #1d2a44}
  .bar>div{height:100%;width:0;background:linear-gradient(90deg,#4aa8ff,#ff6a00)}
  .ctrl{display:grid;grid-template-columns:1fr 1fr 1fr;align-items:end;gap:10px;margin-top:8px;padding:0 14px}
  .gears{display:flex;gap:8px;flex-wrap:wrap}
  .gbtn{width:54px;height:54px;border-radius:12px;border:2px solid #24385f;background:#0b1424;color:#cfe0ff;
        font-size:22px;font-weight:700;cursor:pointer}
  .gbtn.on{border-color:#ff6a00;color:#fff;background:#2a1606;box-shadow:0 0 12px #ff6a0088}
  .gbtn:disabled{opacity:.35;cursor:not-allowed}
  .start{justify-self:center;width:130px;height:130px;border-radius:50%;cursor:pointer;
         border:4px solid #d8232a;background:radial-gradient(circle,#2a0b0d,#0b0405);
         color:#fff;font-size:14px;font-weight:700;letter-spacing:2px;line-height:1.5;
         box-shadow:0 0 18px #d8232a66}
  .start.on{border-color:#4aa8ff;box-shadow:0 0 18px #4aa8ff66;background:radial-gradient(circle,#0b1a33,#050a14)}
  .pedalwrap{justify-self:end;text-align:center;perspective:520px;padding-right:20px}
  .pedal{width:92px;height:190px;margin:0 auto;border-radius:12px 12px 20px 20px;
         transform-origin:50% 100%;position:relative;
         background:repeating-linear-gradient(180deg,#0d0d0f 0 7px,#2a2d33 7px 11px),
         linear-gradient(90deg,#444,#aaa 50%,#444);
         background-blend-mode:overlay;border:3px solid #6b7280;
         box-shadow:0 10px 22px #000,inset 0 0 14px #000;opacity:.4}
  .plabel{color:#8fb7ff;font-size:11px;letter-spacing:3px;margin-top:10px}
  .throttle-control{margin:14px 14px 0;padding:12px;background:#0b1424;
        border:1px solid #24385f;border-radius:12px;text-align:center}
  .throttle-title{color:#8fb7ff;font-size:12px;letter-spacing:2px;margin-bottom:8px}
  .throttle-control input{width:100%;accent-color:#ff6a00;cursor:pointer}
  .throttle-control input:disabled{opacity:.4;cursor:not-allowed}
  .throttle-labels{display:flex;justify-content:space-between;color:#6f86ad;font-size:10px}
  .hint{text-align:center;color:#5d7396;font-size:11px;margin-top:8px}
  @media(max-width:650px){
    .cluster{grid-template-columns:1fr 0.8fr 1fr}
    .digital{font-size:38px}
    .gearbox{font-size:28px;padding:2px 12px}
    .ctrl{grid-template-columns:1fr 1fr 1fr;padding:0 4px;gap:4px}
    .gbtn{width:38px;height:42px;font-size:17px}
    .start{width:90px;height:90px;font-size:11px}
    .pedal{width:55px;height:115px}
    .pedalwrap{padding-right:0}
  }
</style>
</head>
<body>
<div class="dash">
  <div class="accent"></div>
  <div class="cluster">
    <svg class="g" id="gs" viewBox="0 0 400 400"></svg>
    <div class="center">
      <div class="digital" id="dig">0</div>
      <div class="unit" id="unit"></div>
      <div class="gearbox" id="gearshow">N</div>
      <div class="glabel">GEAR</div>
      <div class="bar"><div id="tbar"></div></div>
      <div class="glabel">THROTTLE <span id="tpct">0</span>%</div>
    </div>
    <svg class="g" id="gr" viewBox="0 0 400 400"></svg>
  </div>

  <div class="ctrl">
    <div>
      <div class="glabel" style="margin-bottom:6px">GEAR</div>
      <div class="gears" id="gears"></div>
    </div>
    <button class="start" id="start">START<br>ENGINE</button>
    <div class="pedalwrap">
      <div class="pedal" id="pedal"></div>
      <div class="plabel">THROTTLE</div>
    </div>
  </div>

  <div class="throttle-control">
    <div class="throttle-title">
      FIXED THROTTLE: <span id="sliderPct">0</span>%
    </div>
    <input id="throttleSlider" type="range" min="0" max="100"
           step="1" value="0" disabled>
    <div class="throttle-labels"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
    <div class="hint">Engine start hone ke baad slider se throttle set karo. Gear: 1-5 keys.</div>
  </div>
</div>

<script>
const C = __CFG__;
const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs, parent){
  const e = document.createElementNS(NS, tag);
  for(const k in attrs) e.setAttribute(k, attrs[k]);
  if(parent) parent.appendChild(e);
  return e;
}

// ---------- gauge builder ----------
function buildGauge(svg, vmax, step, redFrom){
  const cx=200, cy=200, r=170;
  const pt=(v,rad)=>{
    const a=(225-270*v/vmax)*Math.PI/180;
    return [cx+rad*Math.cos(a), cy-rad*Math.sin(a)];
  };

  const defs=el('defs',{},svg);
  const rg=el('radialGradient',{id:'face'+svg.id,cx:'50%',cy:'45%',r:'60%'},defs);
  el('stop',{offset:'0%','stop-color':'#111a2b'},rg);
  el('stop',{offset:'100%','stop-color':'#04060a'},rg);
  el('circle',{cx,cy,r:192,fill:'#0a0f1a',stroke:'#2b3d62','stroke-width':5},svg);
  el('circle',{cx,cy,r:184,fill:'none',stroke:'#4aa8ff','stroke-width':1.5,opacity:.7},svg);
  el('circle',{cx,cy,r:180,fill:'url(#face'+svg.id+')'},svg);

  if(redFrom!==null){
    const [x1,y1]=pt(redFrom,r-6), [x2,y2]=pt(vmax,r-6);
    el('path',{d:`M ${x1} ${y1} A ${r-6} ${r-6} 0 0 1 ${x2} ${y2}`,
      stroke:'#d8232a','stroke-width':7,fill:'none'},svg);
  }

  const n=Math.round(vmax/step*2);
  for(let i=0;i<=n;i++){
    const v=i*step/2, major=i%2===0;
    const [xo,yo]=pt(v,r), [xi,yi]=pt(v,r-(major?22:12));
    const red=(redFrom!==null && v>=redFrom);
    el('line',{x1:xo,y1:yo,x2:xi,y2:yi,
      stroke:red?'#d8232a':'#e8eefc','stroke-width':major?3:1.5},svg);

    if(major){
      const [xl,yl]=pt(v,r-46);
      const t=el('text',{x:xl,y:yl,fill:red?'#ff5a5f':'#fff',
        'font-size':21,'font-family':'Segoe UI,Arial',
        'text-anchor':'middle','dominant-baseline':'middle','font-weight':600},svg);
      t.textContent=Number.isInteger(v)?v:v.toFixed(1);
    }
  }

  const needle=el('g',{},svg);
  el('polygon',{points:'196,215 204,215 200,40',fill:'#ff6a00',
    filter:'drop-shadow(0 0 4px #ff6a00)'},needle);
  el('circle',{cx,cy,r:17,fill:'#11182a',stroke:'#4aa8ff','stroke-width':3},svg);
  needle.style.transformOrigin='200px 200px';
  return needle;
}

function setNeedle(n,v,vmax){
  const f=Math.min(Math.max(v/vmax,0),1);
  n.style.transform='rotate('+(-135+270*f)+'deg)';
}

const nS=buildGauge(document.getElementById('gs'),C.speed_max,C.speed_step,null);
const nR=buildGauge(document.getElementById('gr'),C.rpm_max,C.rpm_step,C.red_from);

(function(){
  const a=el('text',{x:200,y:300,fill:'#8fb7ff','font-size':16,
    'text-anchor':'middle','letter-spacing':3},document.getElementById('gs'));
  a.textContent=C.unit.toUpperCase();

  const b=el('text',{x:200,y:300,fill:'#8fb7ff','font-size':16,
    'text-anchor':'middle','letter-spacing':3},document.getElementById('gr'));
  b.textContent='x1000 RPM';
})();

document.getElementById('unit').textContent=C.unit.toUpperCase();

// ---------- state ----------
let mode='off'; // off | sweep | on
let gear=C.gears[0];
let pct=0;      // fixed throttle percentage: 0..100
let speed=0, rpm=0, sweepT=0;

// ---------- throttle controls ----------
const throttleSlider=document.getElementById('throttleSlider');
const sliderPct=document.getElementById('sliderPct');
const pedal=document.getElementById('pedal');

throttleSlider.addEventListener('input',()=>{
  if(mode!=='on') return;
  pct=Number(throttleSlider.value);
  sliderPct.textContent=Math.round(pct);
});

// ---------- gears ----------
const gBox=document.getElementById('gears');

C.gears.forEach(g=>{
  const b=document.createElement('button');
  b.className='gbtn';
  b.textContent=g;
  b.id='g'+g;
  b.disabled=true;
  b.onclick=()=>setGear(g);
  gBox.appendChild(b);
});

function setGear(g){
  if(mode!=='on') return;
  gear=g;
  C.gears.forEach(x=>{
    document.getElementById('g'+x).classList.toggle('on',x===g);
  });
}

// ---------- start / stop ----------
const startBtn=document.getElementById('start');

startBtn.onclick=()=>{
  if(mode==='off'){
    mode='sweep';
    sweepT=0;
    pct=0;
    throttleSlider.value=0;
    sliderPct.textContent='0';
    throttleSlider.disabled=true;
    startBtn.classList.add('on');
    startBtn.innerHTML='STOP<br>ENGINE';
  } else {
    mode='off';
    pct=0;
    throttleSlider.value=0;
    sliderPct.textContent='0';
    throttleSlider.disabled=true;
    startBtn.classList.remove('on');
    startBtn.innerHTML='START<br>ENGINE';
    C.gears.forEach(x=>{
      const b=document.getElementById('g'+x);
      b.disabled=true;
      b.classList.remove('on');
    });
    document.getElementById('gearshow').textContent='N';
    pedal.style.transform='rotateX(0deg)';
  }
};

// ---------- keyboard gear selection ----------
window.addEventListener('keydown',e=>{
  if(e.code==='Space'||e.code==='ArrowUp'){
    e.preventDefault();
  }
  const d=parseInt(e.key);
  if(!isNaN(d)&&C.gears.includes(d)) setGear(d);
});

// ---------- linear regression prediction ----------
function predictSpeed(thr,g){
  return Math.max(C.speed_b+C.speed_w[0]*thr+C.speed_w[1]*g,0);
}

function predictRpm(thr,g){
  if(C.rpm_w){
    return Math.max(C.rpm_b+C.rpm_w[0]*thr+C.rpm_w[1]*g,0);
  }
  return 800+5000*(thr/C.t_max);
}

// ---------- main loop ----------
let last=performance.now();

function frame(now){
  const dt=Math.min((now-last)/1000,0.05);
  last=now;

  if(mode==='sweep'){
    sweepT+=dt;
    const T=2.6;
    const p=Math.min(sweepT/T,1);
    const f=p<0.5?(p/0.5):(1-(p-0.5)/0.5);
    const e=f*f*(3-2*f);

    speed=e*C.speed_max;
    rpm=e*C.rpm_max*1000;

    if(p>=1){
      mode='on';
      speed=0;
      rpm=C.idle;
      gear=C.gears[0];
      throttleSlider.disabled=false;

      C.gears.forEach(x=>{
        document.getElementById('g'+x).disabled=false;
      });
      setGear(gear);
    }
  } else if(mode==='on'){
    // Slider value remains fixed until the user changes it.
    const thr=(pct/100)*C.t_max;
    const tS=predictSpeed(thr,gear);
    const tR=Math.max(predictRpm(thr,gear),C.idle);

    // Vehicle-style smoothing/inertia for dashboard animation.
    speed+=(tS-speed)*Math.min(dt*1.6,1);
    rpm+=(tR-rpm)*Math.min(dt*5,1);

    document.getElementById('gearshow').textContent=gear;
  } else {
    speed*=Math.pow(0.02,dt);
    rpm*=Math.pow(0.02,dt);
  }

  setNeedle(nS,speed,C.speed_max);
  setNeedle(nR,rpm/1000,C.rpm_max);
  document.getElementById('dig').textContent=Math.round(speed);
  document.getElementById('tpct').textContent=Math.round(pct);
  document.getElementById('tbar').style.width=pct+'%';
  sliderPct.textContent=Math.round(pct);
  throttleSlider.value=pct;
  pedal.style.transform='rotateX('+(-pct*0.32)+'deg)';

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);
</script>
</body>
</html>
"""


# ===================== 9. STREAMLIT UI =====================
st.markdown(
    "<h3 style='text-align:center'>Car Speed Predictor</h3>",
    unsafe_allow_html=True,
)
st.caption(
    "Data: DEMO (synthetic)"
    if using_demo
    else f"Data: tumhara CSV ({len(df)} clean rows)"
)

components.html(HTML.replace("__CFG__", json.dumps(cfg)), height=790, scrolling=False)

eq = (
    f"speed = {bias[0]:.2f} + "
    f"{coef[0][0]:.3f} x throttle + "
    f"{coef[0][1]:.3f} x gear"
)
note = "" if HAS_RPM else " | RPM dataset mein nahi hai, isliye andaza hai"

st.caption(
    f"Model: Linear Regression | {eq} | Speed R² (test): {score:.3f}{note}"
)
st.caption(
    f"Check: full throttle + top gear par predicted speed = "
    f"{predict(T_MAX, gears[-1]):.0f} {unit}"
)