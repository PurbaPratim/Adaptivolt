"""
AdaptiVolt — Interactive Dashboard
Run:  python -m streamlit run dashboard.py
Needs: gain_net_weights.json in the same folder.
       pip install streamlit numpy plotly
"""
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="AdaptiVolt", layout="wide", page_icon="⚡")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
.stApp { background: #0C0E14; }
* { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1240px; }
.hero { border-bottom: 1px solid #1F2430; padding-bottom: 18px; margin-bottom: 24px; }
.hero h1 { font-size: 30px; font-weight: 800; color: #F0F2F6; letter-spacing: -0.6px; margin: 0; }
.hero h1 .bolt { color: #4DD8C0; }
.hero p { color: #6B7280; font-size: 14px; margin: 4px 0 0; }
.lbl { font-size: 12px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: #4DD8C0; margin: 8px 0 14px; }
.metric { background: #141821; border: 1px solid #1F2430; border-radius: 10px; padding: 16px; text-align: center; }
.metric .mname { font-size: 11px; color: #6B7280; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
.metric .mval { font-size: 26px; font-weight: 700; color: #F0F2F6; font-family: 'JetBrains Mono', monospace; margin: 4px 0; }
.metric .mimp { font-size: 13px; font-weight: 700; }
.metric .mimp.good { color: #4DD8C0; } .metric .mimp.bad { color: #F87171; }
.foot { color: #4B5563; font-size: 12px; border-top: 1px solid #1F2430; padding-top: 14px; margin-top: 8px; }
.stSlider label { color: #9CA3AF !important; font-weight: 600 !important; font-size: 13px !important; }
div[data-testid="stHorizontalBlock"] button { background: #141821; color: #9CA3AF; border: 1px solid #1F2430;
    border-radius: 8px; font-weight: 600; font-size: 13px; }
div[data-testid="stHorizontalBlock"] button:hover { border-color: #4DD8C0; color: #4DD8C0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_weights():
    with open("gain_net_weights.json") as f:
        return json.load(f)

W = load_weights()
W0=np.array(W["W0"]); b0=np.array(W["b0"]); W1=np.array(W["W1"]); b1=np.array(W["b1"])
W2=np.array(W["W2"]); b2=np.array(W["b2"])
x_mean=np.array(W["x_mean"]); x_std=np.array(W["x_std"])
g_lo=np.array(W["g_lo"]); g_hi=np.array(W["g_hi"])
w_min_fixed=W["w_min_fixed"]; w_max_fixed=W["w_max_fixed"]

def nn_gains(Vin, Iload):
    x=(np.array([Vin,Iload])-x_mean)/x_std
    h1=np.maximum(W0@x+b0,0); h2=np.maximum(W1@h1+b1,0)
    s=1/(1+np.exp(-(W2@h2+b2)))
    g=g_lo+(g_hi-g_lo)*s
    return dict(Kp_v=g[0],Ki_v=g[1],K_vin=g[2],w_min=w_min_fixed,w_max=w_max_fixed)

def sim_tfdobc(Vin_op, Iload_op, g):
    Vref=24.0; L=500e-6; C=220e-6; fs=50e3; Ts=1/fs; n_sub=10; dt=Ts/n_sub
    R_op=Vref/Iload_op; D_MIN=0.05; D_MAX=0.95
    T_sim=60e-3; N=int(round(T_sim/Ts)); t=np.arange(N)*Ts
    t_load_dn=20e-3; t_Vin_dn=30e-3; t_load_up=40e-3; t_Vin_up=45e-3; T_ramp=10e-3
    Dp_op=Vin_op/Vref; wn=Dp_op/np.sqrt(L*C)
    zn_n=0.30; zd_n=0.95; Wn=2/Ts*np.tan(wn*Ts/2)
    bz=[1,2*zn_n*Wn,Wn**2]; az=[1,2*zd_n*Wn,Wn**2]; K=2/Ts
    a0=az[0]*K*K+az[1]*K+az[2]
    b_n=[(bz[0]*K*K+bz[1]*K+bz[2])/a0,(2*bz[2]-2*bz[0]*K*K)/a0,(bz[0]*K*K-bz[1]*K+bz[2])/a0]
    a_n=[1.0,(2*az[2]-2*az[0]*K*K)/a0,(az[0]*K*K-az[1]*K+az[2])/a0]
    omega_i=900.0; omega_oi_min=g["w_min"]*wn; omega_oi_max=g["w_max"]*wn
    adapt_threshold=10.0; adapt_tau=3e-3; alpha_adapt=Ts/(adapt_tau+Ts)
    tau_vin=1e-3; alpha_vin=Ts/(tau_vin+Ts)
    slew=50000.0; f_lpf=8e3; alpha_lpf=(2*np.pi*f_lpf*Ts)/(1+2*np.pi*f_lpf*Ts)
    int_max=0.4; iL_ref_max=3.5*Vref**2/(Vin_op*R_op)
    vC=np.zeros(N); iL=np.zeros(N)
    z1=z2=int_v=0.0; xn1=xn2=0.0
    D0=1-Vin_op/Vref; d_prev=D0; d_lpf=D0; iLref_prev=0.0; activity=0.0; Vin_filt=Vin_op
    for k in range(N):
        Vin_k=Vin_op if not (t_Vin_dn<=t[k]<t_Vin_up) else 0.75*Vin_op
        R_k=R_op if not (t_load_dn<=t[k]<t_load_up) else 2*R_op
        D_nom=1-Vin_k/Vref; Vref_k=Vref*min(t[k]/T_ramp,1)
        Vin_fn=Vin_filt+alpha_vin*(Vin_k-Vin_filt); dVin=(Vin_fn-Vin_filt)/Ts; Vin_filt=Vin_fn
        iL_FF_vin=-g["K_vin"]*dVin
        ev=Vref_k-vC[k]
        evn=b_n[0]*ev+xn1; xn1=b_n[1]*ev-a_n[1]*evn+xn2; xn2=b_n[2]*ev-a_n[2]*evn
        iL_FF_nom=Vref**2/(Vin_k*R_k)
        iLref_u=iL_FF_nom+iL_FF_vin+g["Kp_v"]*evn+int_v
        iLref_s=min(max(iLref_u,0),iL_ref_max)
        if 0<=iLref_u<=iL_ref_max: int_v+=g["Ki_v"]*evn*Ts
        int_v=min(max(int_v,-int_max),int_max)
        step=slew*Ts
        iLref=iLref_prev+min(max(iLref_s-iLref_prev,-step),step); iLref_prev=iLref
        activity+=alpha_adapt*(abs(evn)-activity); af=min(activity/adapt_threshold,1.0)
        omega_oi=omega_oi_min+(omega_oi_max-omega_oi_min)*af
        beta1=2*omega_oi; beta2=omega_oi**2
        b0e=0.7*max(vC[k],5)/L; e_eso=z1-iL[k]; u_eso=d_prev-D_nom
        z1=z1+Ts*(z2-beta1*e_eso+b0e*u_eso)
        if D_MIN+0.002<d_prev<D_MAX-0.002: z2=z2+Ts*(-beta2*e_eso)
        z2=min(max(z2,-5e6),5e6)
        u0=omega_i*(iLref-z1); dr=(u0-z2)/b0e+D_nom
        d_raw=min(max(dr,D_MIN),D_MAX); d_lpf=d_lpf+alpha_lpf*(d_raw-d_lpf)
        dk=min(max(d_lpf,D_MIN),D_MAX); d_prev=dk
        if k<N-1:
            Dp=1-dk; iLk=iL[k]; vCk=vC[k]
            for _ in range(n_sub):
                diL=(Vin_k-Dp*vCk)/L; dvC=(Dp*iLk-vCk/R_k)/C
                iLk=max(iLk+dt*diL,0); vCk=vCk+dt*dvC
            iL[k+1]=iLk; vC[k+1]=vCk
    op=t>=15e-3; vd=(t>=t_Vin_dn)&(t<t_Vin_up); ld=t<t_load_dn
    rms=float(np.sqrt(np.mean((vC[op]-Vref)**2)))
    os=float((vC[ld].max()-Vref)/Vref*100); dip=float(Vref-vC[vd].min())
    return t, vC, rms, os, dip

# zoom state
if "zoom" not in st.session_state:
    st.session_state.zoom = 1.0

st.markdown('<div class="hero"><h1><span class="bolt">⚡</span> AdaptiVolt</h1>'
            '<p>Edge AI gain scheduling for TF-DOBC boost converter control · 747-param MLP · 0.73 KB int8</p></div>',
            unsafe_allow_html=True)

col1, col2 = st.columns([1, 2.3], gap="large")

with col1:
    st.markdown('<div class="lbl">Operating Point</div>', unsafe_allow_html=True)
    Vin = st.slider("Input voltage  Vin (V)", 9.0, 15.0, 9.0, 0.5)
    Iload = st.slider("Load current  Iload (A)", 0.6, 2.4, 2.4, 0.1)

    g_nn = nn_gains(Vin, Iload)
    g_fix = dict(Kp_v=2.0, Ki_v=40.0, K_vin=1.5e-4, w_min=1.8, w_max=3.0)

    # ---- circular gauges (car-meter style) ----
    st.markdown('<div class="lbl" style="margin-top:14px">NN-Predicted Gains</div>', unsafe_allow_html=True)

    def gauge(value, lo, hi, title, fixed, fmt):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number={'font': {'color': '#F0F2F6', 'size': 26, 'family': 'JetBrains Mono'},
                    'valueformat': fmt},
            title={'text': title, 'font': {'color': '#6B7280', 'size': 12}},
            gauge={
                'axis': {'range': [lo, hi], 'tickcolor': '#3A4150',
                         'tickfont': {'color': '#4B5563', 'size': 9}},
                'bar': {'color': '#4DD8C0', 'thickness': 0.28},
                'bgcolor': '#0F1219',
                'borderwidth': 1, 'bordercolor': '#1F2430',
                'steps': [{'range': [lo, hi], 'color': '#141821'}],
                'threshold': {'line': {'color': '#F87171', 'width': 3},
                              'thickness': 0.75, 'value': fixed},
            }))
        fig.update_layout(height=150, margin=dict(l=18, r=18, t=34, b=6),
                          paper_bgcolor='rgba(0,0,0,0)', font={'family': 'Inter'})
        return fig

    st.plotly_chart(gauge(g_nn['Kp_v'], g_lo[0], g_hi[0], "Kp_v · proportional",
                          g_fix['Kp_v'], ".3f"), use_container_width=True,
                    config={'displayModeBar': False})
    st.plotly_chart(gauge(g_nn['Ki_v'], g_lo[1], g_hi[1], "Ki_v · integral",
                          g_fix['Ki_v'], ".1f"), use_container_width=True,
                    config={'displayModeBar': False})
    st.plotly_chart(gauge(g_nn['K_vin']*1e4, g_lo[2]*1e4, g_hi[2]*1e4,
                          "K_vin · feedforward (×10⁻⁴)", g_fix['K_vin']*1e4, ".2f"),
                    use_container_width=True, config={'displayModeBar': False})
    st.markdown('<div style="font-size:11px;color:#4B5563;text-align:center;margin-top:-6px">'
                '🔴 red marker = fixed-gain baseline · ω bounds fixed 1.8 / 3.0</div>',
                unsafe_allow_html=True)

with col2:
    t, vf, rf, of_, df_ = sim_tfdobc(Vin, Iload, g_fix)
    _, vn, rn, on_, dn_ = sim_tfdobc(Vin, Iload, g_nn)

    # zoom buttons
    zc1, zc2, zc3, _ = st.columns([1,1,1,3])
    if zc1.button("➖ Zoom out"): st.session_state.zoom = min(st.session_state.zoom*1.4, 4.0)
    if zc2.button("➕ Zoom in"):  st.session_state.zoom = max(st.session_state.zoom/1.4, 0.3)
    if zc3.button("⟳ Reset"):    st.session_state.zoom = 1.0
    z = st.session_state.zoom

    # WIDER axis range: base full-range with generous padding, scaled by zoom
    ymin = min(vf.min(), vn.min()); ymax = max(vf.max(), vn.max())
    ymid = 24.0
    half = max(abs(ymax-ymid), abs(ymid-ymin)) * 1.25 * z
    half = max(half, 4.0)
    ylo, yhi = ymid - half, ymid + half

    fig = go.Figure()
    fig.add_vrect(x0=30, x1=45, fillcolor="#4DD8C0", opacity=0.06, line_width=0)
    fig.add_trace(go.Scatter(x=t*1e3, y=vf, name="Fixed gain",
                             line=dict(color="#F87171", dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=t*1e3, y=vn, name="NN-tuned",
                             line=dict(color="#4DD8C0", width=2.8)))
    fig.add_hline(y=24, line=dict(color="#4B5563", dash="dot", width=1),
                  annotation_text="24 V ref", annotation_position="right",
                  annotation_font_color="#6B7280")
    fig.update_layout(
        height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0F1219",
        font=dict(color="#9CA3AF", family="Inter"),
        xaxis=dict(title="Time (ms)", gridcolor="#1A1E28", zeroline=False, range=[0, 60]),
        yaxis=dict(title="Output Voltage (V)", gridcolor="#1A1E28", zeroline=False, range=[ylo, yhi]),
        legend=dict(x=0.012, y=0.02, xanchor="left", yanchor="bottom",
                    bgcolor="rgba(20,24,33,0.85)", bordercolor="#1F2430", borderwidth=1),
        margin=dict(l=10, r=10, t=8, b=10),
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": True, "scrollZoom": True,
                            "modeBarButtonsToRemove": ["select2d","lasso2d","autoScale2d"]})

    def metric(name, val, imp, unit):
        good = imp >= 0
        return (f'<div class="metric"><div class="mname">{name}</div>'
                f'<div class="mval">{val}{unit}</div>'
                f'<div class="mimp {"good" if good else "bad"}">'
                f'{"▼" if good else "▲"} {abs(imp):.0f}% {"better" if good else "worse"}</div></div>')

    rms_imp=(rf-rn)/rf*100; os_imp=(of_-on_)/max(of_,0.01)*100; dip_imp=(df_-dn_)/df_*100
    a,b,c = st.columns(3)
    a.markdown(metric("Tracking RMS", f"{rn:.3f}", rms_imp, " V"), unsafe_allow_html=True)
    b.markdown(metric("Overshoot", f"{on_:.1f}", os_imp, " %"), unsafe_allow_html=True)
    c.markdown(metric("Disturbance Dip", f"{dn_:.3f}", dip_imp, " V"), unsafe_allow_html=True)

st.markdown('<div class="foot">Disturbances: load step at 20 / 40 ms, 25% input-voltage sag at 30–45 ms (shaded). '
            'Use ➕ / ➖ / ⟳ to zoom the voltage axis, or scroll/drag on the plot. '
            'Network inference target: &lt;100 µs on ARM Cortex-M.</div>', unsafe_allow_html=True)
