import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title='OpticVision AI | 가상 시력 검사', page_icon='🐭', layout='wide')
st.markdown('''<style>
.block-container {padding-top: 1.3rem;}
.hero {padding:18px 22px;border-radius:16px;background:linear-gradient(110deg,#102f4b,#24577a);color:white;margin-bottom:16px}
.hero h1{margin:0;font-size:2rem}.hero p{margin:7px 0 0;color:#dcebf7}
.note{padding:12px;border-radius:10px;background:#fff5db;color:#614a12;margin-top:10px}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🐭 OpticVision AI</h1><p>시신경 역노화 치료 효능 평가를 위한 가상 시각 반응 측정 장비</p></div>',unsafe_allow_html=True)
st.info('교육용 가상 시뮬레이터입니다. 모든 반응은 임의로 설정한 수학적 모델에서 생성되며, 실제 동물·치료 효능·AI 측정 데이터가 아닙니다.')

GROUPS = {'정상군':0.46,'시신경 손상군':0.15,'OSK 치료군 (가정)':0.29}
colors = {'정상군':'#2278b8','시신경 손상군':'#cc5656','OSK 치료군 (가정)':'#2c9b75'}

def response_probability(freq, threshold, contrast, speed):
    # 임의의 교육용 반응 모형. 문헌을 적합한 결과가 아님.
    effective_threshold = threshold * (0.45 + 0.55*contrast/100)
    speed_factor = max(0.5, 1-0.025*abs(speed-8))
    return np.clip((1 / (1+np.exp((np.asarray(freq)-effective_threshold)/0.045))) * speed_factor, 0, 1)

def run_trial(group, frequency, contrast, speed, noise_percent, n_trials, seed):
    rng=np.random.default_rng(seed)
    p=float(response_probability(frequency,GROUPS[group],contrast,speed))
    visible=rng.random(n_trials)<p
    # true tracking angle velocity vs measured noisy head angular velocity
    stim=np.full(n_trials,float(speed))
    tracking_velocity=np.where(visible, speed*rng.normal(0.83,0.12,n_trials),rng.normal(0,1.5,n_trials))
    artifact=rng.random(n_trials)<noise_percent/100
    measured=tracking_velocity + np.where(artifact, rng.normal(0,12,n_trials),rng.normal(0,0.65,n_trials))
    detected=(measured>speed*0.38)&(measured<speed*1.65)
    df=pd.DataFrame({'시행':np.arange(1,n_trials+1),'자극 속도 (°/s)':stim,'가상 머리 속도 (°/s)':np.round(measured,2),'가상 반응 여부':visible,'잡동작 발생':artifact,'알고리즘 추적 판정':detected})
    return df,p

with st.sidebar:
    st.header('⚙️ 실험 설정')
    group=st.selectbox('가상 실험군',list(GROUPS),index=2)
    frequency=st.slider('공간주파수 (cycles/°)',0.05,0.65,0.20,0.05)
    speed=st.slider('줄무늬 회전 속도 (°/s)',2,15,8)
    contrast=st.slider('줄무늬 대비 (%)',10,100,100,10)
    noise=st.slider('잡동작 발생 확률 (%)',0,40,10,5)
    n_trials=st.slider('가상 측정 횟수',10,100,40,10)
    seed=st.number_input('난수 시드 (재현용)',value=42,min_value=0,max_value=100000,step=1)
    simulate=st.button('▶ 검사 실행',type='primary',use_container_width=True)

if 'last' not in st.session_state: st.session_state.last=None
if simulate:
    df,p=run_trial(group,frequency,contrast,speed,noise,n_trials,int(seed))
    st.session_state.last={'data':df,'p':p,'group':group,'frequency':frequency,'contrast':contrast,'speed':speed,'noise':noise,'n_trials':n_trials,'seed':seed}

left,right=st.columns([1.2,1])
with left:
    st.subheader('① 가상 장비 / 회전 줄무늬')
    # Animated, purely illustrative top-down rendering.
    components.html('''<!doctype html><html><head><style>
    body{margin:0;background:#e9f3f7;font-family:Arial;border-radius:14px;overflow:hidden}
    .stage{position:relative;height:320px;display:grid;place-items:center}
    .arena{width:272px;height:272px;border-radius:50%;border:9px solid #5b7b90;box-shadow:0 5px 22px #aac3ce;position:relative;overflow:hidden;background:repeating-conic-gradient(#152431 0deg 14deg,#ecf3f7 14deg 28deg);animation:spin 8s linear infinite}
    .platform{position:absolute;width:150px;height:150px;border-radius:50%;background:#d5e2ea;border:5px solid #7898aa;top:61px;left:61px;box-sizing:border-box}
    .mouse{position:absolute;top:102px;left:108px;font-size:57px;transform:rotate(-30deg);animation:follow 3s ease-in-out infinite}
    .camera{position:absolute;top:4px;left:calc(50% - 19px);background:#153d58;color:white;padding:7px 10px;border-radius:7px;font-size:12px}
    .label{position:absolute;bottom:6px;font-size:12px;color:#31526b}
    @keyframes spin{to{transform:rotate(360deg)}}@keyframes follow{0%,100%{transform:rotate(-27deg)}50%{transform:rotate(16deg)}}
    </style></head><body><div class="stage"><div class="arena"><div class="platform"></div></div><div class="mouse">🐁</div><div class="camera">◉ 카메라</div><div class="label">시각 설명용 애니메이션 · 실제 동물 추적 영상 아님</div></div></body></html>''',height=330)
    st.caption('위 애니메이션은 원리 설명용이며, 실제 공간주파수나 쥐의 반응과 실시간 연동된 물리 시뮬레이션은 아닙니다.')
with right:
    st.subheader('② 선택한 실험 조건')
    st.write(f'**실험군:** {group}')
    st.write(f'**공간주파수:** {frequency:.2f} cycles/°')
    st.write(f'**회전 속도:** {speed} °/s')
    st.write(f'**대비 / 잡동작 확률:** {contrast}% / {noise}%')
    if st.session_state.last:
        result=st.session_state.last
        measured=result['data']['알고리즘 추적 판정'].mean()*100
        st.metric('최근 검사: 추적 판정 비율',f'{measured:.1f}%')
        st.caption('반응 비율이지 실제 시력 또는 치료 회복률(%)이 아닙니다.')
    else:
        st.write('왼쪽 사이드바에서 조건을 정한 뒤 **검사 실행**을 눌러주세요.')

st.subheader('③ 줄무늬 난이도별 가상 시각 반응')
frequencies=np.linspace(0.05,0.65,121)
fig=go.Figure()
for g,threshold in GROUPS.items():
    fig.add_trace(go.Scatter(x=frequencies,y=100*response_probability(frequencies,threshold,contrast,speed),name=g,mode='lines',line=dict(color=colors[g],width=3)))
fig.add_vline(x=frequency,line_dash='dot',line_color='gray')
fig.update_layout(xaxis_title='공간주파수 (cycles/°)',yaxis_title='모형상 반응 확률 (%)',yaxis_range=[0,105],height=365,margin=dict(l=20,r=20,t=15,b=15),legend=dict(orientation='h',y=1.15))
st.plotly_chart(fig,use_container_width=True)
st.caption('임의 설정한 역치(정상 0.46, 손상 0.15, 치료 가정 0.29 cycles/°)를 사용합니다. 과학 논문에서 추정한 OSK 치료 효과 값이 아닙니다.')

if st.session_state.last:
    result=st.session_state.last
    df=result['data']
    st.subheader('④ 가상 행동 신호 분석')
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=df['시행'],y=df['자극 속도 (°/s)'],name='줄무늬 자극 속도',mode='lines',line=dict(dash='dash')))
    fig2.add_trace(go.Scatter(x=df['시행'],y=df['가상 머리 속도 (°/s)'],name='가상 쥐 머리 회전 속도',mode='lines+markers'))
    fig2.update_layout(xaxis_title='가상 시행 번호',yaxis_title='각속도 (°/s)',height=330,margin=dict(l=20,r=20,t=10,b=20))
    st.plotly_chart(fig2,use_container_width=True)
    a,b,c=st.columns(3)
    a.metric('모형 반응 확률',f"{result['p']*100:.1f}%")
    b.metric('추적 판정',f"{int(df['알고리즘 추적 판정'].sum())} / {len(df)}")
    c.metric('잡동작 발생',f"{int(df['잡동작 발생'].sum())}회")
    st.dataframe(df,use_container_width=True,hide_index=True)
    buffer=io.BytesIO()
    with pd.ExcelWriter(buffer,engine='openpyxl') as writer:
        df.to_excel(writer,sheet_name='가상 측정 데이터',index=False)
        pd.DataFrame([{'실험군':result['group'],'공간주파수':result['frequency'],'회전속도':result['speed'],'대비':result['contrast'],'잡동작 확률':result['noise'],'모형 반응 확률':result['p'],'추적 판정 비율':float(df['알고리즘 추적 판정'].mean()),'주의':'모든 수치는 가상 데이터이며 실제 치료 효능 평가에 사용할 수 없음'}]).to_excel(writer,sheet_name='결과 요약',index=False)
    st.download_button('📥 가상 실험 결과 엑셀 다운로드',data=buffer.getvalue(),file_name='opticvision_simulated_results.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

st.markdown('---')
st.subheader('⑤ 공모전 설명에 넣을 과학적 한계')
st.write('이 작품은 실제 시력 측정 장비의 **작동 원리와 분석 흐름**을 보여주는 프로토타입입니다. 가상 치료군의 반응 회복은 입력한 가정에 따른 결과로서, 야마나카 인자의 효능을 입증하지 않습니다. 실제 장비 검증에는 쥐 행동 영상, 측정 정확도 평가, 대조 실험과 적절한 생물학적 분석이 필요합니다.')
