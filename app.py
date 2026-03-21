import streamlit as st
from model import MicrogridModel
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components

st.set_page_config(page_title="GSoC2026Smartgrid", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background:#f6f7f9; color:#1f2937;
           font-family:"Segoe UI",system-ui,-apple-system,sans-serif; }
  .main .block-container { padding-top:1.2rem; padding-bottom:2rem; max-width:1400px; }
  .app-title { font-size:2rem; font-weight:700; color:#111827; margin:0; line-height:1.2; }
  .app-sub   { font-size:.9rem; color:#6b7280; margin-top:.2rem; margin-bottom:.7rem; }
  .pill      { display:inline-block; padding:.3rem .75rem; border-radius:999px;
               font-size:.83rem; font-weight:600; border:1px solid transparent; }
  .panel     { background:#fff; border:1px solid #e5e7eb; border-radius:14px;
               padding:16px 18px; box-shadow:0 1px 3px rgba(0,0,0,.04); margin-bottom:4px; }
  .ptitle    { font-size:.93rem; font-weight:700; color:#111827; margin:0 0 10px 0; }
  .mcard     { background:#fff; border:1px solid #e5e7eb; border-radius:12px;
               padding:14px 16px; box-shadow:0 1px 2px rgba(0,0,0,.03); }
  .mlbl  { font-size:.76rem; color:#6b7280; margin-bottom:4px; }
  .mval  { font-size:1.38rem; font-weight:700; color:#111827; line-height:1.1; }
  .mnote { font-size:.72rem; color:#9ca3af; margin-top:4px; }
  .stext { font-size:.87rem; color:#4b5563; line-height:1.65; }
  .save-pos { font-weight:700; color:#059669; }
  .save-neg { font-weight:700; color:#dc2626; }
</style>
""", unsafe_allow_html=True)

# ── Assets ───────────────────────────────────────────────────────────────────
WIND = "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExZmp1NjY0ZTVudXByYmVqcnE0emk1ZGNwN21jOHE1azA5Z3htOHNucCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3MbRNvfnMyUJeKGlsw/giphy.gif"
LOAD = "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExNWY4cjBlM2ljemoya3J3a2d1OXFnc2Q5ZDY5ZmllZ3Q0dnAxZGJ3ZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/XZcwMvQLRf9aXRa3qW/giphy.gif"
BESS = "https://apatura.energy/wp-content/uploads/2024/12/BESS.jpg"
GRID = "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExbXUwN3JxZWM5Mjd3dGs3azJ3YTd2aGZqeXdicmg2d2J2dGtvdjdkNCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Qa0n2cnLuyDqYra4zk/giphy.gif"

# ── Session init ──────────────────────────────────────────────────────────────
if "model" not in st.session_state:
    st.session_state.model = MicrogridModel()
    st.session_state.steps = 0
model = st.session_state.model

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("System Control")
st.sidebar.caption("Configure and run the microgrid simulation.")
steps_input = st.sidebar.number_input(
    "Simulation steps", min_value=1, max_value=100000, value=100, step=1)
ca, cb = st.sidebar.columns(2)
run_clicked   = ca.button("Run")
reset_clicked = cb.button("Reset")

if run_clicked:
    total = int(steps_input)
    batch = max(1000, total // 20)
    tick  = max(1, total // 20)
    bar   = st.progress(0)
    done  = 0
    while done < total:
        n = min(batch, total - done)
        for _ in range(n):
            model.step()
            st.session_state.steps += 1
        done += n
        if done % tick < batch or done >= total:
            bar.progress(min(done / total, 1.0))
    st.sidebar.success(f"Completed {total} steps")

if reset_clicked:
    st.session_state.model = MicrogridModel()
    st.session_state.steps = 0
    st.rerun()

# ── Live values ───────────────────────────────────────────────────────────────
prod       = float(model.wind.production)
cons       = float(model.load1.consumption + model.load2.consumption)
soc        = float(model.battery.soc)
grid_imp   = float(model.grid_import)
price      = float(model.price)
total_cost = float(model.total_cost)
rule_cost  = float(model.rule_total_cost)
action     = model.history[-1]["action"] if model.history else "idle"
MODE       = {"charge": ("Charging","#1d4ed8"), "discharge": ("Discharging","#b45309")}
mode_text, mode_color = MODE.get(action, ("Standby","#6b7280"))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">GSoC2026Smartgrid</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Microgrid simulation — renewable generation, '
            'battery dispatch, and grid interaction using Mesa + Random Forest.</div>',
            unsafe_allow_html=True)
st.markdown(f'<span class="pill" style="background:{mode_color}15;color:{mode_color};'
            f'border-color:{mode_color}30;">System mode: {mode_text}</span>',
            unsafe_allow_html=True)
st.write("")

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown('<div class="panel"><div class="ptitle">Key metrics</div></div>',
            unsafe_allow_html=True)
cols = st.columns(5)
for col, lbl, val, note in [
    (cols[0], "Wind generation",   f"{prod:.1f} kW",     "Current renewable output"),
    (cols[1], "Load demand",       f"{cons:.1f} kW",     "Total consumption"),
    (cols[2], "Battery SOC",       f"{soc:.1f}%",        "State of charge"),
    (cols[3], "Grid import",       f"{grid_imp:.1f} kW", "External supply"),
    (cols[4], "Electricity price", f"{price:.2f} $/kWh", "Current tariff"),
]:
    with col:
        st.markdown(f'<div class="mcard"><div class="mlbl">{lbl}</div>'
                    f'<div class="mval">{val}</div>'
                    f'<div class="mnote">{note}</div></div>',
                    unsafe_allow_html=True)
st.write("")

# ── Summary + battery ─────────────────────────────────────────────────────────
saving     = rule_cost - total_cost
saving_pct = (saving / rule_cost * 100) if rule_cost > 0 else 0.0
scls       = "save-pos" if saving >= 0 else "save-neg"
sstr       = (f"+${saving:.2f} ({saving_pct:+.1f}%)" if saving >= 0
              else f"-${abs(saving):.2f} ({saving_pct:.1f}%)")
r1, r2 = st.columns(2)
with r1:
    st.markdown(f"""<div class="panel"><div class="ptitle">System summary</div>
    <div class="stext">
      Current action (RandomForest): <b>{action}</b><br>
      RandomForest total cost: <b>${total_cost:.2f}</b><br>
      Rule-based total cost: <b>${rule_cost:.2f}</b><br>
      Saving vs rule-based: <span class="{scls}">{sstr}</span><br>
      Simulation steps: <b>{st.session_state.steps}</b>
    </div></div>""", unsafe_allow_html=True)
with r2:
    st.markdown('<div class="panel"><div class="ptitle">Battery state of charge'
                '</div><div class="stext">Current SOC level</div></div>',
                unsafe_allow_html=True)
    st.progress(max(0.0, min(1.0, soc / 100.0)))
st.write("")

# ── Flow diagram ──────────────────────────────────────────────────────────────
#
# Node edges (these are the ONLY valid start/end points for the dot):
#   Grid    bottom  = y:110   centre_x = 350
#   Wind    right   = x:150   centre_y = 230
#   BUS     covers x:280-420, y:200-260  (dot hides behind it, z-index trick)
#   Load    left    = x:550   centre_y = 230
#   Battery top     = y:340   centre_x = 350
#
# Wire segments:
#   Grid→BUS  : x=350, y=110 → y=200
#   Wind→BUS  : y=230, x=150 → x=280
#   BUS→Load  : y=230, x=420 → x=550
#   BUS→Batt  : x=350, y=260 → y=340
#
# Dot = 14×14px so top-left = centre − 7.
# Dot starts and ends at wire endpoints (node edges), travels through BUS
# centre (hidden behind BUS box via z-index).

if action == "charge":
    STATUS = "#059669"
elif action == "discharge":
    STATUS = "#2563eb"
else:
    STATUS = "#dc2626" if (soc < 15 or grid_imp > 6) else "#d97706" if grid_imp > 2 else "#64748b"

INACTIVE = "#e2e8f0"
c_wind_bus = STATUS   if action == "charge"               else INACTIVE
c_bus_load = STATUS   if action in ("discharge", "idle")  else INACTIVE
c_grid_bus = STATUS   if action == "idle"                 else INACTIVE
c_bus_batt = STATUS   if action in ("charge","discharge") else INACTIVE

# Keyframe: dot starts at wire start, travels through BUS (hidden), ends at wire end.
# charge   : Wind right(150,230) → BUS centre(350,230) → Battery top(350,340)
# discharge: Battery top(350,340) → BUS centre(350,230) → Load left(550,230)
# idle     : Grid bottom(350,110) → BUS centre(350,230) → Load left(550,230)
if action == "charge":
    kf = """
    @keyframes dot_path {
      0%   { left:143px; top:223px; opacity:0; }
      5%   { left:143px; top:223px; opacity:1; }
      45%  { left:343px; top:223px; opacity:1; }
      55%  { left:343px; top:223px; }
      95%  { left:343px; top:333px; opacity:1; }
      100% { left:343px; top:333px; opacity:0; }
    }"""
elif action == "discharge":
    kf = """
    @keyframes dot_path {
      0%   { left:343px; top:333px; opacity:0; }
      5%   { left:343px; top:333px; opacity:1; }
      45%  { left:343px; top:223px; opacity:1; }
      55%  { left:343px; top:223px; }
      95%  { left:543px; top:223px; opacity:1; }
      100% { left:543px; top:223px; opacity:0; }
    }"""
else:
    kf = """
    @keyframes dot_path {
      0%   { left:343px; top:103px; opacity:0; }
      5%   { left:343px; top:103px; opacity:1; }
      45%  { left:343px; top:223px; opacity:1; }
      55%  { left:343px; top:223px; }
      95%  { left:543px; top:223px; opacity:1; }
      100% { left:543px; top:223px; opacity:0; }
    }"""

def _node(img_url, label, is_bus=False, z=1):
    bg  = "#eff6ff" if is_bus else "#ffffff"
    bdr = "#bfdbfe" if is_bus else "#e2e8f0"
    lc  = "#1e3a8a" if is_bus else "#111827"
    img = (f'<img src="{img_url}" style="width:64px;height:56px;'
           f'object-fit:cover;border-radius:8px;margin-bottom:5px;"/>' if img_url else "")
    sub = ('<div style="font-size:9px;color:#93c5fd;margin-top:2px;">'
           'power balance node</div>' if is_bus else "")
    return (f'<div style="width:100%;height:100%;display:flex;flex-direction:column;'
            f'align-items:center;justify-content:center;background:{bg};'
            f'border:1.5px solid {bdr};border-radius:12px;'
            f'box-shadow:0 1px 4px rgba(0,0,0,.08);position:relative;z-index:{z};">'
            f'{img}<span style="font-size:13px;font-weight:700;color:{lc};">{label}</span>'
            f'{sub}</div>')

def _abs(l, t, w, h, content, z=1):
    return (f'<div style="position:absolute;left:{l}px;top:{t}px;'
            f'width:{w}px;height:{h}px;z-index:{z};">{content}</div>')

def wh(c, w):  # horizontal wire
    return f'<div style="width:{w}px;height:4px;background:{c};border-radius:2px;"></div>'

def wv(c, h):  # vertical wire
    return f'<div style="width:4px;height:{h}px;background:{c};border-radius:2px;"></div>'

diagram = f"""
<style>{kf}</style>
<div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;
            padding:18px 24px;box-shadow:0 1px 3px rgba(0,0,0,.04);">
  <div style="position:relative;width:700px;height:460px;margin:0 auto;">

    <!-- Wires z=1 (behind everything) -->
    {_abs(348,110, 4, 90, wv(c_grid_bus,90), z=1)}
    {_abs(150,228,130,  4, wh(c_wind_bus,130), z=1)}
    {_abs(420,228,130,  4, wh(c_bus_load,130), z=1)}
    {_abs(348,260, 4, 80, wv(c_bus_batt,80), z=1)}

    <!-- Regular nodes z=1 -->
    {_abs(290, 10,120,100, _node(GRID,"Grid"),    z=1)}
    {_abs( 30,180,120,100, _node(WIND,"Wind"),    z=1)}
    {_abs(550,180,120,100, _node(LOAD,"Load"),    z=1)}
    {_abs(290,340,120,100, _node(BESS,"Battery"), z=1)}

    <!-- BUS z=2 — sits ON TOP of dot so corner is hidden cleanly -->
    {_abs(280,200,140, 60, _node(None,"BUS",is_bus=True,z=2), z=2)}

    <!-- Dot z=1 — travels behind BUS, starts/ends exactly at wire endpoints -->
    <div style="position:absolute;width:14px;height:14px;border-radius:50%;
                background:{STATUS};box-shadow:0 0 8px {STATUS}88;z-index:1;
                animation:dot_path 2s ease-in-out infinite;"></div>

  </div>
</div>
"""

st.markdown('<div class="panel"><div class="ptitle">Energy flow</div></div>', unsafe_allow_html=True)
components.html(diagram, height=500)
st.write("")

st.write("")

# ── Performance charts ────────────────────────────────────────────────────────
st.markdown('<div class="panel"><div class="ptitle">System performance over time'
            '</div></div>', unsafe_allow_html=True)

if model.history:
    df  = pd.DataFrame(model.history)
    ds  = max(1, len(df) // 2000)
    dfp = df.iloc[::ds].reset_index(drop=True)

    fig = go.Figure()
    for y, name in [("production","Wind generation"),("consumption","Load demand"),
                    ("battery","Battery SOC (%)"),("grid_import","Grid import"),
                    ("cost","Step cost")]:
        fig.add_trace(go.Scatter(y=dfp[y], name=name, mode="lines", line=dict(width=2)))
    fig.update_layout(template="simple_white", height=380,
                      margin=dict(l=20,r=20,t=40,b=20),
                      legend=dict(orientation="h",yanchor="bottom",y=1.02,x=0),
                      xaxis_title="Simulation step", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="panel"><div class="ptitle">'
                'RandomForest controller vs rule-based: cumulative cost'
                '</div></div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=dfp["total_cost"], name="RandomForest controller",
                              mode="lines", line=dict(width=2.5, color="#2563eb")))
    fig2.add_trace(go.Scatter(y=dfp["rule_total_cost"], name="Rule-based baseline",
                              mode="lines", line=dict(width=2.5, color="#9ca3af", dash="dash")))
    fig2.update_layout(template="simple_white", height=320,
                       margin=dict(l=20,r=20,t=40,b=20),
                       legend=dict(orientation="h",yanchor="bottom",y=1.02,x=0),
                       xaxis_title="Simulation step", yaxis_title="Cumulative cost ($)")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Press Run to start the simulation and generate performance data.")

st.caption(f"Simulation steps: {st.session_state.steps}  ·  GSoC 2026")