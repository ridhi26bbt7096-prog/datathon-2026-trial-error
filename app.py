import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import io

st.set_page_config(page_title="SlotSwap | Hospital Flow Intelligence", page_icon="🏥", layout="wide")

# SlotSwap is a decision-support demo, not a clinical system.
LANGS = {
    "English": {"tag":"EN","title":"Hospital Flow Intelligence","sub":"See the bottlenecks. Prioritise fairly. Keep patients informed.",
        "overview":"Overview","ops":"Operations","patient":"Patient portal","model":"Wait-time model","data":"Dataset & setup",
        "wait":"Average wait","long":"Waiting over 60 min","left":"Left before care","visits":"Encounters loaded",
        "queue":"Investigate first","dept":"Department performance","triage":"Wait by triage level","search":"Search patient or encounter ID",
        "upload":"Upload your CSV files","uploadhelp":"Upload encounters.csv to unlock analysis and model training. Other files are optional.",
        "predict":"Predict estimated wait","run":"Estimate wait","warning":"Demo decision support only. Not for clinical decisions.",
        "no_data":"Upload encounters.csv in Dataset & setup to analyse your actual data.",
        "priority":"Priority queue","why":"Why flagged","status":"Status","record":"Record simulated update","message":"Message",
        "modelnote":"Model uses a held-out test set; metrics describe this dataset only."},
    "Hindi": {"tag":"HI","title":"अस्पताल कार्यप्रवाह विश्लेषण","sub":"समस्या पहचानें। निष्पक्ष प्राथमिकता तय करें। मरीजों को जानकारी दें।","overview":"अवलोकन","ops":"प्रबंधन","patient":"मरीज पोर्टल","model":"प्रतीक्षा समय मॉडल","data":"डेटा और सेटअप","wait":"औसत प्रतीक्षा","long":"60 मिनट से अधिक","left":"इलाज से पहले चले गए","visits":"कुल विज़िट","queue":"पहले जाँचें","dept":"विभाग प्रदर्शन","triage":"ट्रायेज के अनुसार प्रतीक्षा","search":"मरीज या विज़िट ID खोजें","upload":"CSV फ़ाइलें अपलोड करें","uploadhelp":"वास्तविक डेटा और मॉडल के लिए encounters.csv अपलोड करें।","predict":"प्रतीक्षा का अनुमान","run":"अनुमान लगाएँ","warning":"केवल डेमो सहायता। चिकित्सकीय निर्णयों के लिए नहीं।","no_data":"अपने डेटा का विश्लेषण करने के लिए encounters.csv अपलोड करें।","priority":"प्राथमिकता सूची","why":"कारण","status":"स्थिति","record":"डेमो अपडेट दर्ज करें","message":"संदेश","modelnote":"मॉडल का मूल्यांकन अलग रखे गए डेटा पर होता है।"},
    "Telugu": {"tag":"TE","title":"ఆసుపత్రి కార్యకలాపాల విశ్లేషణ","sub":"అడ్డంకులను గుర్తించండి. న్యాయంగా ప్రాధాన్యత ఇవ్వండి.","overview":"సారాంశం","ops":"నిర్వహణ","patient":"రోగి పోర్టల్","model":"వేచిచూసే సమయ నమూనా","data":"డేటా & సెటప్","wait":"సగటు నిరీక్షణ","long":"60 నిమిషాలకుపైగా","left":"చికిత్సకు ముందు వెళ్లినవారు","visits":"సందర్శనలు","queue":"ముందుగా పరిశీలించండి","dept":"విభాగ పనితీరు","triage":"ట్రయాజ్ ప్రకారం నిరీక్షణ","search":"రోగి లేదా సందర్శన ID","upload":"CSV ఫైళ్లను అప్‌లోడ్ చేయండి","uploadhelp":"వాస్తవ విశ్లేషణకు encounters.csv అవసరం.","predict":"నిరీక్షణ అంచనా","run":"అంచనా వేయండి","warning":"డెమో మాత్రమే; వైద్య నిర్ణయాలకు కాదు.","no_data":"మీ డేటా కోసం encounters.csv అప్‌లోడ్ చేయండి.","priority":"ప్రాధాన్యత జాబితా","why":"కారణం","status":"స్థితి","record":"డెమో అప్‌డేట్ నమోదు","message":"సందేశం","modelnote":"మోడల్ పరీక్ష డేటాపై మూల్యాంకనం చేయబడుతుంది."},
    "Tamil": {"tag":"TA","title":"மருத்துவமனை செயல்பாட்டு நுண்ணறிவு","sub":"தாமதங்களைக் கண்டறிந்து நியாயமாக முன்னுரிமை அளிக்கவும்.","overview":"கண்ணோட்டம்","ops":"நிர்வாகம்","patient":"நோயாளர் தளம்","model":"காத்திருப்பு நேர மாதிரி","data":"தரவு & அமைப்பு","wait":"சராசரி காத்திருப்பு","long":"60 நிமிடங்களுக்கு மேல்","left":"சிகிச்சைக்கு முன் சென்றவர்கள்","visits":"வருகைகள்","queue":"முதலில் ஆய்வு","dept":"துறை செயல்திறன்","triage":"ட்ரையாஜ் அடிப்படையில்","search":"நோயாளர் அல்லது வருகை ID","upload":"CSV கோப்புகளைப் பதிவேற்றவும்","uploadhelp":"உண்மையான பகுப்பாய்விற்கு encounters.csv தேவை.","predict":"காத்திருப்பு கணிப்பு","run":"கணிக்கவும்","warning":"டெமோ மட்டும்; மருத்துவ முடிவுகளுக்கல்ல.","no_data":"உங்கள் தரவிற்காக encounters.csv பதிவேற்றவும்.","priority":"முன்னுரிமைப் பட்டியல்","why":"காரணம்","status":"நிலை","record":"டெமோ புதுப்பிப்பைப் பதிவு செய்","message":"செய்தி","modelnote":"மாதிரி தனி சோதனைத் தரவில் மதிப்பிடப்படுகிறது."},
    "Bengali": {"tag":"BN","title":"হাসপাতাল কার্যপ্রবাহ বিশ্লেষণ","sub":"বিলম্ব শনাক্ত করুন। ন্যায্যভাবে অগ্রাধিকার দিন।","overview":"সারাংশ","ops":"পরিচালনা","patient":"রোগী পোর্টাল","model":"অপেক্ষার সময় মডেল","data":"ডেটা ও সেটআপ","wait":"গড় অপেক্ষা","long":"৬০ মিনিটের বেশি","left":"চিকিৎসার আগে চলে গেছেন","visits":"ভিজিট","queue":"আগে পরীক্ষা করুন","dept":"বিভাগের কার্যকারিতা","triage":"ট্রায়াজ অনুযায়ী অপেক্ষা","search":"রোগী বা ভিজিট ID খুঁজুন","upload":"CSV ফাইল আপলোড করুন","uploadhelp":"বিশ্লেষণের জন্য encounters.csv আপলোড করুন।","predict":"অপেক্ষার সময় অনুমান","run":"অনুমান করুন","warning":"শুধু ডেমো; চিকিৎসার সিদ্ধান্তের জন্য নয়।","no_data":"আপনার ডেটা বিশ্লেষণে encounters.csv আপলোড করুন।","priority":"অগ্রাধিকার তালিকা","why":"কারণ","status":"অবস্থা","record":"ডেমো আপডেট নথিভুক্ত করুন","message":"বার্তা","modelnote":"মডেল আলাদা পরীক্ষার ডেটায় মূল্যায়িত।"},
    "Punjabi": {"tag":"PA","title":"ਹਸਪਤਾਲ ਕਾਰਜ-ਪ੍ਰਵਾਹ ਵਿਸ਼ਲੇਸ਼ਣ","sub":"ਦੇਰੀ ਪਛਾਣੋ। ਨਿਰਪੱਖ ਤਰਜੀਹ ਦਿਓ।","overview":"ਝਲਕ","ops":"ਪ੍ਰਬੰਧਨ","patient":"ਮਰੀਜ਼ ਪੋਰਟਲ","model":"ਉਡੀਕ ਸਮਾਂ ਮਾਡਲ","data":"ਡਾਟਾ ਅਤੇ ਸੈੱਟਅੱਪ","wait":"ਔਸਤ ਉਡੀਕ","long":"60 ਮਿੰਟ ਤੋਂ ਵੱਧ","left":"ਇਲਾਜ ਤੋਂ ਪਹਿਲਾਂ ਚਲੇ ਗਏ","visits":"ਮੁਲਾਕਾਤਾਂ","queue":"ਪਹਿਲਾਂ ਜਾਂਚੋ","dept":"ਵਿਭਾਗੀ ਕਾਰਗੁਜ਼ਾਰੀ","triage":"ਟ੍ਰਾਇਏਜ ਅਨੁਸਾਰ ਉਡੀਕ","search":"ਮਰੀਜ਼ ਜਾਂ ਮੁਲਾਕਾਤ ID","upload":"CSV ਫਾਈਲਾਂ ਅੱਪਲੋਡ ਕਰੋ","uploadhelp":"ਵਿਸ਼ਲੇਸ਼ਣ ਲਈ encounters.csv ਅੱਪਲੋਡ ਕਰੋ।","predict":"ਉਡੀਕ ਦਾ ਅੰਦਾਜ਼ਾ","run":"ਅੰਦਾਜ਼ਾ ਲਗਾਓ","warning":"ਸਿਰਫ਼ ਡੈਮੋ; ਡਾਕਟਰੀ ਫ਼ੈਸਲਿਆਂ ਲਈ ਨਹੀਂ।","no_data":"ਆਪਣਾ ਡਾਟਾ ਵੇਖਣ ਲਈ encounters.csv ਅੱਪਲੋਡ ਕਰੋ।","priority":"ਤਰਜੀਹ ਸੂਚੀ","why":"ਕਾਰਨ","status":"ਸਥਿਤੀ","record":"ਡੈਮੋ ਅੱਪਡੇਟ ਦਰਜ ਕਰੋ","message":"ਸੁਨੇਹਾ","modelnote":"ਮਾਡਲ ਵੱਖਰੇ ਟੈਸਟ ਡਾਟੇ 'ਤੇ ਜਾਂਚਿਆ ਜਾਂਦਾ ਹੈ।"},
    "Kannada": {"tag":"KN","title":"ಆಸ್ಪತ್ರೆ ಕಾರ್ಯಪ್ರವಾಹ ವಿಶ್ಲೇಷಣೆ","sub":"ವಿಳಂಬಗಳನ್ನು ಗುರುತಿಸಿ. ನ್ಯಾಯಯುತ ಆದ್ಯತೆ ನೀಡಿ.","overview":"ಅವಲೋಕನ","ops":"ನಿರ್ವಹಣೆ","patient":"ರೋಗಿ ಪೋರ್ಟಲ್","model":"ಕಾಯುವ ಸಮಯ ಮಾದರಿ","data":"ಡೇಟಾ ಮತ್ತು ಸೆಟಪ್","wait":"ಸರಾಸರಿ ಕಾಯುವಿಕೆ","long":"60 ನಿಮಿಷಕ್ಕಿಂತ ಹೆಚ್ಚು","left":"ಚಿಕಿತ್ಸೆಗೆ ಮೊದಲು ತೆರಳಿದವರು","visits":"ಭೇಟಿಗಳು","queue":"ಮೊದಲು ಪರಿಶೀಲಿಸಿ","dept":"ವಿಭಾಗದ ಕಾರ್ಯಕ್ಷಮತೆ","triage":"ಟ್ರಯಾಜ್ ಪ್ರಕಾರ ಕಾಯುವಿಕೆ","search":"ರೋಗಿ ಅಥವಾ ಭೇಟಿ ID","upload":"CSV ಫೈಲ್‌ಗಳನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ","uploadhelp":"ವಿಶ್ಲೇಷಣೆಗೆ encounters.csv ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.","predict":"ಕಾಯುವ ಸಮಯ ಅಂದಾಜು","run":"ಅಂದಾಜಿಸಿ","warning":"ಡೆಮೊ ಮಾತ್ರ; ವೈದ್ಯಕೀಯ ನಿರ್ಧಾರಗಳಿಗೆ ಅಲ್ಲ.","no_data":"ನಿಮ್ಮ ಡೇಟಾಕ್ಕಾಗಿ encounters.csv ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.","priority":"ಆದ್ಯತಾ ಪಟ್ಟಿ","why":"ಕಾರಣ","status":"ಸ್ಥಿತಿ","record":"ಡೆಮೊ ನವೀಕರಣ ದಾಖಲಿಸಿ","message":"ಸಂದೇಶ","modelnote":"ಮಾದರಿಯನ್ನು ಪ್ರತ್ಯೇಕ ಪರೀಕ್ಷಾ ಡೇಟಾದಲ್ಲಿ ಮೌಲ್ಯಮಾಪನ ಮಾಡಲಾಗುತ್ತದೆ."},
    "Malayalam": {"tag":"ML","title":"ആശുപത്രി പ്രവർത്തന വിശകലനം","sub":"താമസങ്ങൾ കണ്ടെത്തുക. നീതിപൂർവം മുൻഗണന നൽകുക.","overview":"അവലോകനം","ops":"നിർവഹണം","patient":"രോഗി പോർട്ടൽ","model":"കാത്തിരിപ്പ് സമയ മാതൃക","data":"ഡാറ്റയും സജ്ജീകരണവും","wait":"ശരാശരി കാത്തിരിപ്പ്","long":"60 മിനിറ്റിൽ കൂടുതൽ","left":"ചികിത്സയ്ക്ക് മുമ്പ് പോയവർ","visits":"സന്ദർശനങ്ങൾ","queue":"ആദ്യം പരിശോധിക്കുക","dept":"വിഭാഗ പ്രകടനം","triage":"ട്രയാജ് അനുസരിച്ച് കാത്തിരിപ്പ്","search":"രോഗി അല്ലെങ്കിൽ സന്ദർശന ID","upload":"CSV ഫയലുകൾ അപ്‌ലോഡ് ചെയ്യുക","uploadhelp":"വിശകലനത്തിന് encounters.csv അപ്‌ലോഡ് ചെയ്യുക.","predict":"കാത്തിരിപ്പ് കണക്കാക്കുക","run":"കണക്കാക്കുക","warning":"ഡെമോ മാത്രം; ചികിത്സാ തീരുമാനങ്ങൾക്ക് ഉപയോഗിക്കരുത്.","no_data":"നിങ്ങളുടെ ഡാറ്റയ്ക്കായി encounters.csv അപ്‌ലോഡ് ചെയ്യുക.","priority":"മുൻഗണന പട്ടിക","why":"കാരണം","status":"നില","record":"ഡെമോ അപ്‌ഡേറ്റ് രേഖപ്പെടുത്തുക","message":"സന്ദേശം","modelnote":"മാതൃക വേർതിരിച്ച ടെസ്റ്റ് ഡാറ്റയിൽ വിലയിരുത്തുന്നു."}
}

st.markdown("""<style>
.block-container{padding-top:1.4rem;padding-bottom:2rem;max-width:1500px}
.hero{padding:1.5rem 1.7rem;border-radius:20px;background:linear-gradient(120deg,#0c3344,#146b70);color:white;margin-bottom:1rem}
.hero h1{font-size:2.2rem;margin:0 0 .4rem 0;color:white}.hero p{font-size:1rem;margin:0;color:#e0f4f4}
.kpi{border:1px solid rgba(120,150,160,.25);border-radius:16px;padding:1rem 1.1rem;background:rgba(120,150,160,.06)}
.kpi-label{font-size:.85rem;opacity:.75}.kpi-value{font-size:1.8rem;font-weight:750;letter-spacing:-.03em}.kpi-foot{font-size:.78rem;opacity:.7}
div[data-testid="stMetric"]{border:1px solid rgba(120,150,160,.25);padding:1rem;border-radius:14px}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🏥 SlotSwap")
    language = st.selectbox("🌐 Language / भाषा / భాష", list(LANGS.keys()), index=0)
    T = LANGS[language]
    page = st.radio("Navigate", [T["overview"], T["ops"], T["patient"], T["model"], T["data"]], label_visibility="collapsed")
    st.divider()
    st.caption("DATATHON • HEALTH TRACK")
    st.caption("Decision support prototype • Synthetic dataset")

st.markdown(f'<div class="hero"><h1>🏥 SlotSwap</h1><p>{T["title"]} · {T["sub"]}</p></div>', unsafe_allow_html=True)
st.warning("⚕️ " + T["warning"])

# Uploads are session-local, so this single file can work without bundling the data folder.
with st.sidebar:
    with st.expander("📂 Load dataset", expanded=False):
        encounters_file = st.file_uploader("encounters.csv (required for real results)", type="csv", key="enc_file")
        patients_file = st.file_uploader("patients.csv (optional)", type="csv", key="pat_file")
        departments_file = st.file_uploader("departments.csv (optional)", type="csv", key="dep_file")
        tests_file = st.file_uploader("diagnostic_tests.csv (optional)", type="csv", key="test_file")

def read_upload(file):
    if file is None: return None
    try:
        return pd.read_csv(file)
    except Exception as e:
        st.error(f"Could not read {file.name}: {e}")
        return None

enc = read_upload(encounters_file)
patients = read_upload(patients_file)
departments = read_upload(departments_file)
tests = read_upload(tests_file)

def find_col(df, candidates):
    if df is None: return None
    lower = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower: return lower[cand.lower()]
    for c in df.columns:
        if any(x in str(c).lower() for x in candidates): return c
    return None

wait_col = find_col(enc, ["wait_time_minutes","waiting_time","wait_minutes","wait_time","wait"])
dept_col = find_col(enc, ["department","department_name","dept"])
triage_col = find_col(enc, ["triage_level","triage","acuity"])
patient_col = find_col(enc, ["patient_id","patientid"])
enc_id_col = find_col(enc, ["encounter_id","visit_id","encounterid"])
outcome_col = find_col(enc, ["visit_outcome","outcome","disposition","status"])
arrival_col = find_col(enc, ["arrival_time","arrival_datetime","check_in_time"])
service_col = find_col(enc, ["service_time_minutes","service_minutes","duration_minutes"])
visit_type_col = find_col(enc, ["visit_type","type"])
def numeric(series):
    return pd.to_numeric(series, errors="coerce")

def section_title(emoji, title, subtitle=None):
    st.markdown(f"### {emoji} {title}")
    if subtitle: st.caption(subtitle)

if enc is None:
    st.info("👈 " + T["uploadhelp"])
    st.markdown("#### " + T["upload"])
    st.write("Upload your supplied `encounters.csv` using **Load dataset** in the sidebar. Add the other CSVs there too if you want patient lookup, department context and diagnostics.")
    c1,c2,c3 = st.columns(3)
    c1.metric("Product", "SlotSwap")
    c2.metric("Languages", "8")
    c3.metric("Model", "Random Forest")
    st.markdown("#### What this prototype does")
    st.markdown("- **Operations:** waiting-time monitoring and an investigate-first queue.\n- **Patient portal:** find an encounter and show its recorded visit information.\n- **ML workspace:** train and evaluate a wait-time regression model from your encounters.\n- **Multilingual interface:** English, Hindi, Telugu, Tamil, Bengali, Punjabi, Kannada and Malayalam.")
    st.stop()

if wait_col is None:
    st.error("Could not identify a waiting-time column in encounters.csv. Columns found: " + ", ".join(map(str,enc.columns)))
    st.stop()

enc = enc.copy()
enc["_wait"] = numeric(enc[wait_col])
enc = enc[enc["_wait"].notna()].copy()
if enc.empty:
    st.error("The waiting-time column has no numeric values.")
    st.stop()
if dept_col is None: enc["_department"] = "Unknown department"; dept_col="_department"
if triage_col is None: enc["_triage"] = "Unknown"; triage_col="_triage"
enc["_department"] = enc[dept_col].fillna("Unknown").astype(str)
enc["_triage"] = enc[triage_col].fillna("Unknown").astype(str)
if patient_col is None: enc["_patient"] = "Unknown"; patient_col="_patient"
if enc_id_col is None: enc["_encid"] = enc.index.astype(str); enc_id_col="_encid"

avg_wait = float(enc["_wait"].mean())
long_wait = float((enc["_wait"] > 60).mean()*100)
left_count = 0
if outcome_col:
    out = enc[outcome_col].astype(str).str.lower()
    left_count = int(out.str.contains("left|lwbs|without being seen|left before", regex=True).sum())

def kpi(label, value, foot):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-foot">{foot}</div></div>', unsafe_allow_html=True)

if page == T["overview"]:
    section_title("📊", T["overview"], f"Analysis of {len(enc):,} records loaded from your encounters CSV.")
    a,b,c,d=st.columns(4)
    with a: kpi(T["wait"],f"{avg_wait:.1f} min","Mean recorded wait")
    with b: kpi(T["long"],f"{long_wait:.1f}%","Share above 60 minutes")
    with c: kpi(T["left"],f"{left_count:,}","Outcome keyword match" if outcome_col else "Outcome column not found")
    with d: kpi(T["visits"],f"{len(enc):,}","Valid wait-time records")
    l,r=st.columns([1.35,1])
    with l:
        section_title("🏢",T["dept"])
        dep=enc.groupby("_department")["_wait"].agg(["mean","median","count"]).sort_values("mean",ascending=False).reset_index()
        st.bar_chart(dep.set_index("_department")["mean"], horizontal=True, x_label="Average wait (minutes)")
        st.dataframe(dep.rename(columns={"_department":"Department","mean":"Mean wait (min)","median":"Median wait (min)","count":"Encounters"}).round(1),use_container_width=True,hide_index=True)
    with r:
        section_title("🚦",T["triage"])
        tri=enc.groupby("_triage")["_wait"].agg(["mean","median","count"]).sort_values("mean",ascending=False)
        st.bar_chart(tri["mean"],horizontal=True)
        st.dataframe(tri.round(1),use_container_width=True)
    st.caption("Fairness note: the queue prioritises excess waiting relative to department and triage patterns. It does not use age, sex, insurance or diagnosis to rank cases.")

elif page == T["ops"]:
    section_title("🧭",T["priority"],"A transparent operational review queue, not a clinical triage recommendation.")
    c1,c2,c3=st.columns(3)
    depts=["All"]+sorted(enc["_department"].unique().tolist())
    triages=["All"]+sorted(enc["_triage"].unique().tolist())
    with c1: dep_filter=st.selectbox("Department",depts)
    with c2: tri_filter=st.selectbox("Triage",triages)
    with c3: threshold=st.slider("Minimum excess wait (minutes)",0,60,15)
    baselines=enc.groupby(["_department","_triage"])["_wait"].transform("median")
    queue=enc.copy()
    queue["_baseline"]=baselines
    queue["_excess"]=queue["_wait"]-queue["_baseline"]
    queue=queue[queue["_excess"]>=threshold]
    if dep_filter!="All": queue=queue[queue["_department"]==dep_filter]
    if tri_filter!="All": queue=queue[queue["_triage"]==tri_filter]
    queue=queue.sort_values("_excess",ascending=False)
    st.metric("Cases to review",f"{len(queue):,}")
    display_cols=[enc_id_col,patient_col,"_department","_triage","_wait","_baseline","_excess"]
    display_cols=list(dict.fromkeys([c for c in display_cols if c in queue.columns]))
    qdisp=queue[display_cols].rename(columns={enc_id_col:"Encounter ID",patient_col:"Patient ID","_department":"Department","_triage":"Triage","_wait":"Wait (min)","_baseline":"Typical median (min)","_excess":"Excess wait (min)"})
    st.dataframe(qdisp.round(1),use_container_width=True,hide_index=True)
    st.download_button("⬇️ Export investigation queue CSV",qdisp.to_csv(index=False).encode("utf-8"),"slotswap_investigation_queue.csv","text/csv")
    st.markdown("**Why a case appears:** its recorded wait exceeds the historical median for the same department and triage group by at least the selected threshold.")
    st.info("This is an operational review aid. Urgent clinical decisions must follow hospital triage protocols and qualified staff.")

elif page == T["patient"]:
    section_title("🪪",T["patient"],"Lookup uses the records present in the uploaded dataset.")
    patients_list=sorted(enc[patient_col].dropna().astype(str).unique().tolist())
    selected=st.selectbox(T["search"],patients_list)
    records=enc[enc[patient_col].astype(str)==selected].copy()
    if enc_id_col:
        records=records.sort_values(arrival_col) if arrival_col else records
    st.metric("Visits found",len(records))
    st.dataframe(records[[c for c in [enc_id_col,patient_col,dept_col,triage_col,wait_col,arrival_col,service_col,outcome_col] if c and c in records.columns]],use_container_width=True,hide_index=True)
    if patients is not None and patient_col in patients.columns:
        pr=patients[patients[patient_col].astype(str)==selected]
        if not pr.empty:
            with st.expander("Patient details from supplied CSV"):
                st.dataframe(pr,use_container_width=True,hide_index=True)
    st.caption("No scheduled slot or doctor assignment is inferred if those fields are absent from the source data.")

elif page == T["model"]:
    section_title("🤖",T["model"],"Random Forest regression · held-out evaluation · editable prediction inputs.")
    st.write(T["modelnote"])
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        features=pd.DataFrame(index=enc.index)
        used=[]
        cat_candidates=[dept_col,triage_col,visit_type_col]
        for col in cat_candidates:
            if col and col in enc.columns and col not in used:
                features[col]=enc[col].fillna("Unknown").astype(str); used.append(col)
        if arrival_col and arrival_col in enc.columns:
            dt=pd.to_datetime(enc[arrival_col],errors="coerce")
            if dt.notna().sum()>0:
                features["arrival_hour"]=dt.dt.hour
                features["arrival_day_of_week"]=dt.dt.dayofweek
        if len(features.columns)==0:
            st.error("Not enough usable feature columns were found for training.")
            st.stop()
        valid=enc["_wait"].notna() & features.notna().any(axis=1)
        X=features.loc[valid].copy()
        y=enc.loc[valid,"_wait"].clip(lower=0)
        if len(X)<30 or y.nunique()<2:
            st.error("At least 30 usable encounters with varying wait times are needed to evaluate this model.")
            st.stop()
        categorical=list(X.select_dtypes(include=["object"]).columns)
        numeric_cols=[c for c in X.columns if c not in categorical]
        pre=ColumnTransformer([("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),categorical),
                               ("num",Pipeline([("impute",SimpleImputer(strategy="median"))]),numeric_cols)])
        model=Pipeline([("prep",pre),("rf",RandomForestRegressor(n_estimators=160,min_samples_leaf=3,random_state=42,n_jobs=-1))])
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
        model.fit(Xtr,ytr)
        pred=model.predict(Xte)
        mae=mean_absolute_error(yte,pred); rmse=float(np.sqrt(mean_squared_error(yte,pred))); r2=r2_score(yte,pred)
        a,b,c,d=st.columns(4)
        a.metric("Test records",f"{len(yte):,}")
        b.metric("MAE",f"{mae:.1f} min")
        c.metric("RMSE",f"{rmse:.1f} min")
        d.metric("R²",f"{r2:.3f}")
        plot=pd.DataFrame({"Actual wait (min)":yte.values,"Predicted wait (min)":pred})
        st.scatter_chart(plot,x="Actual wait (min)",y="Predicted wait (min)")
        st.caption(f"Features used: {', '.join(map(str,X.columns))}. Sensitive demographic attributes are not used. Metrics are from a random held-out split and may be optimistic if records from the same patients overlap.")
        st.markdown("#### " + T["predict"])
        with st.form("prediction_form"):
            chosen={}
            for col in categorical:
                vals=sorted(X[col].dropna().astype(str).unique().tolist())
                chosen[col]=st.selectbox(str(col).replace("_"," ").title(),vals)
            for col in numeric_cols:
                vals=pd.to_numeric(X[col],errors="coerce").dropna()
                if col=="arrival_hour": chosen[col]=st.slider("Arrival hour",0,23,12)
                elif col=="arrival_day_of_week": chosen[col]=st.selectbox("Arrival day",list(range(7)),format_func=lambda x:["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
                else: chosen[col]=st.number_input(col, value=float(vals.median()) if not vals.empty else 0.0)
            submitted=st.form_submit_button(T["run"],type="primary",use_container_width=True)
        if submitted:
            row=pd.DataFrame([chosen],columns=X.columns)
            estimate=max(0,float(model.predict(row)[0]))
            st.success(f"Estimated wait: **{estimate:.0f} minutes**")
            st.caption("Estimate only. Real waiting time depends on live capacity, arrivals and clinical priority; this prototype has no live hospital feed.")
    except ImportError:
        st.error("The ML dependencies are missing. Install them with: pip install scikit-learn")
    except Exception as e:
        st.error(f"Model could not train on these CSV columns: {e}")
        st.caption("Check that the wait-time field is numeric and the file has enough records.")

elif page == T["data"]:
    section_title("📁",T["data"],"Upload CSVs in the sidebar. This app reads your files locally in the session.")
    st.write(T["uploadhelp"])
    for name,df in [("encounters.csv",enc),("patients.csv",patients),("departments.csv",departments),("diagnostic_tests.csv",tests)]:
        if df is not None:
            with st.expander(f"Preview {name} · {len(df):,} rows"):
                st.dataframe(df.head(100),use_container_width=True,hide_index=True)
    st.markdown("#### Data quality checks")
    quality=pd.DataFrame([
        {"Check":"Encounter rows loaded","Result":f"{len(enc):,}"},
        {"Check":"Missing wait values removed","Result":str(int(pd.to_numeric(read_upload(encounters_file)[wait_col],errors="coerce").isna().sum())) if encounters_file else "—"},
        {"Check":"Department categories","Result":str(enc["_department"].nunique())},
        {"Check":"Triage categories","Result":str(enc["_triage"].nunique())},
    ])
    st.dataframe(quality,use_container_width=True,hide_index=True)
    st.download_button("⬇️ Download cleaned encounters CSV",enc.to_csv(index=False).encode("utf-8"),"slotswap_cleaned_encounters.csv","text/csv")

st.divider()
st.caption("SlotSwap • Datathon prototype | Use synthetic data for demonstration only. Not a medical device, not connected to a live hospital system.")
