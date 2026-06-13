"""
AdaptiVolt — Interactive Dashboard
Run:  streamlit run dashboard.py
Needs: gain_net_weights.json in the same folder.
       pip install streamlit numpy plotly
"""
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="AdaptiVolt", layout="wide")

# ---------- load NN weights ----------
@st.cache_resource
def load_weights():
    with open("gain_net_weights.json") as f:
        return json.load(f)

W = load_weights()
W0 = np.array(W["W0"]); b0 = np.array(W["b0"])
W1 = np.array(W["W1"]); b1 = np.array(W["b1"])
W2 = np.array(W["W2"]); b2 = np.array(W["b2"])
x_mean = np.array(W["x_mean"]); x_std = np.array(W["x_std"])
g_lo = np.array(W["g_lo"]); g_hi = np.array(W["g_hi"])
w_min_fixed = W["w_min_fixed"]; w_max_fixed = W["w_max_fixed"]

def nn_gains(Vin, Iload):
    x = (np.array([Vin, Iload]) - x_mean) / x_std
    h1 = np.maximum(W0 @ x + b0, 0)
    h2 = np.maximum(W1 @ h1 + b1, 0)
    s  = 1/(1+np.exp(-(W2 @ h2 + b2)))
    g = g_lo + (g_hi - g_lo) * s
    return dict(Kp_v=g[0], Ki_v=g[1], K_vin=g[2], w_min=w_min_fixed, w_max=w_max_fixed)

# ---------- TF-DOBC simulator (ported from sim_tfdobc_trace.m) ----------
def sim_tfdobc(Vin_op, Iload_op, g):
    Vref=24.0; L=500e-6; C=220e-6; fs=50e3; Ts=1/fs; n_sub=10; dt=Ts/n_sub
    R_op=Vref/Iload_op
    D_MIN=0.05; D_MAX=0.95
    T_sim=60e-3; N=int(round(T_sim/Ts)); t=np.arange(N)*Ts
    t_load_dn=20e-3; t_Vin_dn=30e-3; t_load_up=40e-3; t_Vin_up=45e-3; T_ramp=10e-3
    Dp_op=Vin_op/Vref; wn=Dp_op/np.sqrt(L*C)

    # notch (bilinear) — second-order, implemented as direct-form biquad
    zn_n=0.30; zd_n=0.95; Wn=2/Ts*np.tan(wn*Ts/2)
    bz=[1,2*zn_n*Wn,Wn**2]; az=[1,2*zd_n*Wn,Wn**2]
    K=2/Ts
    a0=az[0]*K*K+az[1]*K+az[2]
    b_n=[(bz[0]*K*K+bz[1]*K+bz[2])/a0,(2*bz[2]-2*bz[0]*K*K)/a0,(bz[0]*K*K-bz[1]*K+bz[2])/a0]
    a_n=[1.0,(2*az[2]-2*az[0]*K*K)/a0,(az[0]*K*K-az[1]*K+az[2])/a0]

    omega_i=900.0
    omega_oi_min=g["w_min"]*wn; omega_oi_max=g["w_max"]*wn
    adapt_threshold=10.0; adapt_tau=3e-3; alpha_adapt=Ts/(adapt_tau+Ts)
    tau_vin=1e-3; alpha_vin=Ts/(tau_vin+Ts)
    slew=50000.0; f_lpf=8e3; alpha_lpf=(2*np.pi*f_lpf*Ts)/(1+2*np.pi*f_lpf*Ts)
    int_max=0.4; iL_ref_max=3.5*Vref**2/(Vin_op*R_op)

    vC=np.zeros(N); iL=np.zeros(N); d_log=np.zeros(N)
    z1=z2=int_v=0.0; xn1=xn2=0.0
    D0=1-Vin_op/Vref; d_prev=D0; d_lpf=D0; iLref_prev=0.0; activity=0.0; Vin_filt=Vin_op

    for k in range(N):
        Vin_k=Vin_op if not (t_Vin_dn<=t[k]<t_Vin_up) else 0.75*Vin_op
        R_k=R_op if not (t_load_dn<=t[k]<t_load_up) else 2*R_op
        D_nom=1-Vin_k/Vref
        Vref_k=Vref*min(t[k]/T_ramp,1)

        Vin_fn=Vin_filt+alpha_vin*(Vin_k-Vin_filt)
        dVin=(Vin_fn-Vin_filt)/Ts; Vin_filt=Vin_fn
        iL_FF_vin=-g["K_vin"]*dVin
        ev=Vref_k-vC[k]
        evn=b_n[0]*ev+xn1
        xn1=b_n[1]*ev-a_n[1]*evn+xn2
        xn2=b_n[2]*ev-a_n[2]*evn
        iL_FF_nom=Vref**2/(Vin_k*R_k)
        iLref_u=iL_FF_nom+iL_FF_vin+g["Kp_v"]*evn+int_v
        iLref_s=min(max(iLref_u,0),iL_ref_max)
        if 0<=iLref_u<=iL_ref_max: int_v+=g["Ki_v"]*evn*Ts
        int_v=min(max(int_v,-int_max),int_max)
        step=slew*Ts
        iLref=iLref_prev+min(max(iLref_s-iLref_prev,-step),step); iLref_prev=iLref
        activity+=alpha_adapt*(abs(evn)-activity)
        af=min(activity/adapt_threshold,1.0)
        omega_oi=omega_oi_min+(omega_oi_max-omega_oi_min)*af
        beta1=2*omega_oi; beta2=omega_oi**2
        b0e=0.7*max(vC[k],5)/L
        e_eso=z1-iL[k]; u_eso=d_prev-D_nom
        z1=z1+Ts*(z2-beta1*e_eso+b0e*u_eso)
        if D_MIN+0.002<d_prev<D_MAX-0.002: z2=z2+Ts*(-beta2*e_eso)
        z2=min(max(z2,-5e6),5e6)
        u0=omega_i*(iLref-z1); dr=(u0-z2)/b0e+D_nom
        d_raw=min(max(dr,D_MIN),D_MAX)
        d_lpf=d_lpf+alpha_lpf*(d_raw-d_lpf)
        dk=min(max(d_lpf,D_MIN),D_MAX); d_prev=dk; d_log[k]=dk
        if k<N-1:
            Dp=1-dk; iLk=iL[k]; vCk=vC[k]
            for _ in range(n_sub):
                diL=(Vin_k-Dp*vCk)/L; dvC=(Dp*iLk-vCk/R_k)/C
                iLk=max(iLk+dt*diL,0); vCk=vCk+dt*dvC
            iL[k+1]=iLk; vC[k+1]=vCk

    op=t>=15e-3; vd=(t>=t_Vin_dn)&(t<t_Vin_up); ld=t<t_load_dn
    rms=float(np.sqrt(np.mean((vC[op]-Vref)**2)))
    os=float((vC[ld].max()-Vref)/Vref*100)
    dip=float(Vref-vC[vd].min())
    return t, vC, rms, os, dip

# ---------- UI ----------
st.title("AdaptiVolt — Neural Gain Scheduling for TF-DOBC")
st.caption("Edge AI-augmented disturbance rejection control for EV power conversion")

c1, c2 = st.columns([1, 2.4])
with c1:
    st.subheader("Operating Point")
    Vin = st.slider("Input voltage Vin (V)", 9.0, 15.0, 9.0, 0.5)
    Iload = st.slider("Load current Iload (A)", 0.6, 2.4, 2.4, 0.1)

    g_nn = nn_gains(Vin, Iload)
    g_fix = dict(Kp_v=2.0, Ki_v=40.0, K_vin=1.5e-4, w_min=1.8, w_max=3.0)

    st.subheader("NN-Predicted Gains")
    st.metric("Kp_v", f"{g_nn['Kp_v']:.3f}", f"{g_nn['Kp_v']-g_fix['Kp_v']:+.3f} vs fixed")
    st.metric("Ki_v", f"{g_nn['Ki_v']:.2f}", f"{g_nn['Ki_v']-g_fix['Ki_v']:+.2f} vs fixed")
    st.metric("K_vin", f"{g_nn['K_vin']:.2e}")
    st.caption(f"w_min={g_nn['w_min']:.1f}, w_max={g_nn['w_max']:.1f} (fixed)")

with c2:
    t, vf, rf, of_, df_ = sim_tfdobc(Vin, Iload, g_fix)
    _, vn, rn, on_, dn_ = sim_tfdobc(Vin, Iload, g_nn)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t*1e3, y=vf, name="Fixed gain",
                             line=dict(color="#D63333", dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=t*1e3, y=vn, name="NN-tuned",
                             line=dict(color="#1A5BD9", width=2.5)))
    fig.add_hline(y=24, line=dict(color="gray", dash="dot"))
    fig.update_layout(height=430, xaxis_title="Time (ms)", yaxis_title="Output Voltage (V)",
                      yaxis_range=[20, 27], legend=dict(x=0.01, y=0.99),
                      margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Tracking RMS", f"{rn:.3f} V", f"{(rn-rf)/rf*100:+.1f}%", delta_color="inverse")
    m2.metric("Overshoot", f"{on_:.1f} %", f"{(on_-of_)/max(of_,0.01)*100:+.0f}%", delta_color="inverse")
    m3.metric("Disturbance dip", f"{dn_:.3f} V", f"{(dn_-df_)/df_*100:+.1f}%", delta_color="inverse")

st.divider()
st.caption("Model: 747-param MLP (2→24→24→3), int8 0.73 KB, <100 µs Cortex-M target. "
           "Disturbances: load step at 20/40 ms, 25% Vin sag at 30–45 ms.")
