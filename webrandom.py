import pandas as pd
import numpy as np
import streamlit as st
import csv
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


st.markdown('<div style="text-align: justify;font-size:300%"><b><h1>Analisis Sentimen Kenaikan Harga BBM menggunakan metode Random Forest Classifier</h1></b></div>',unsafe_allow_html=True)


# LOAD RANDOM FOREST MODEL & TF-IDF VECTORIZATION
filename_model = 'finalized_model_random.sav'
filename_tfidf = 'vectorizerrr.pickle'
# model = pkl.load(open(filename, 'rb'))
model = pkl.load(open(filename_model, 'rb'))
vect = pkl.load(open(filename_tfidf, 'rb'))


st.sidebar.image("upy.png")

## ADD HORIZONTAL LINE
st.markdown("""---""")

## ADD TEXT INPUT & SUBMIT BUTTON
text = st.text_input('Masukkan kalimat yang akan dianalisis sentimennya', placeholder='Contoh : Setujuuu uang pensiun DPR alihkan buat rakyat')
submit = st.button("Submit")

## SAVE INPUT IN DATAFRAME
data_result = pd.DataFrame({'Text':[text]})


## SENTIMENT ANALYSIS
y_pred = model.predict(vect.transform(data_result['Text'].values))
y_pred_proba = model.predict_proba(vect.transform(data_result['Text'].values))

## DISPLAY OUTPUT
if text:
    if y_pred == 2:
        result = 'Kalimat di atas memiliki sentimen POSITIF dengan akurasi ' + str(np.round(np.max(y_pred_proba, axis=1)*100,2))[1:3]+'%'
        st.success(result)
    elif y_pred == 1:
        result = 'Kalimat di atas memiliki sentimen NEGATIF dengan akurasi ' + str(np.round(np.max(y_pred_proba, axis=1)*100,2))[1:3]+'%'
        st.error(result)
    else:
        result = 'Kalimat di atas memiliki sentimen NETRAL dengan akurasi ' + str(np.round(np.max(y_pred_proba, axis=1)*100,2))[1:3]+'%'
        st.warning(result)

st.title('Hasil Implementasi Dari Penelitian')
st.write("By : Karandi Nurbagja")


st.write(f"## Dataset Kenaikan Harga BBM")

algoritma = st.sidebar.selectbox(
    'Random Forest',
    ('Random Forest','')
)


dataku = pd.read_excel('data_clear_2.xlsx')
if st.checkbox("Show Data BBM"):
    st.write(dataku.head(1800))


@st.cache
def convert_data(dataku):
# IMPORTANT: Cache the conversion to prevent computation on every rerun
    return dataku.to_csv().encode('utf-8')

csv = convert_data(dataku)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='sentiment.csv',
    mime='text/csv',
)


dataaa = st.write("Keterangan :  "),st.write("1 (Negatif)"),st.write("0 (Netral)"),st.write("2 (Positif)")

st.write(f"### Pedoman Label")


datakuu = pd.read_excel('opini.xlsx')
if st.checkbox("Show Pedoman Label"):
    st.write(datakuu.head(5))


@st.cache
def convert_data(datakuu):
# IMPORTANT: Cache the conversion to prevent computation on every rerun
    return datakuu.to_csv().encode('utf-8')

csv = convert_data(datakuu)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='sentiment.csv',
    mime='text/csv',
)

data = pd.read_excel('data_clear_2.xlsx')
if st.write(""):
    st.write(data.head(1800))
#X = data.Komentar
y = dataku['Label']
st.write('Jumlah Baris dan Kolom : ', data.shape)
st.write('Jumlah Kelas : ',len(np.unique(y)))


def tambah_parameter(nama_algoritma):
    params = dict()
    if nama_algoritma == 'Random Forest':
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params
params = tambah_parameter(algoritma)

def pilih_klasifikasi(nama_algoritma, params):
    #algo = None
    if nama_algoritma == 'Random Forest':
        algo = RandomForestClassifier(n_estimators=params['n_estimators'], random_state= 0)
    return algo
algo = pilih_klasifikasi(algoritma, params)

### PROSES KLASIFIKASI ###
cv = TfidfVectorizer()
#X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=1234)
X = data['Komentar']
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
xv_train = cv.fit_transform(X_train)
xv_test = cv.transform(X_test)
algo.fit(xv_train, y_train)

y_pred = algo.predict(xv_test)

acc = round(accuracy_score(y_test, y_pred),2)

st.write(f'Algoritma = {algoritma}')
st.write(f'Akurasi = ', str(round(acc*100)) + ' %')


select=st.sidebar.selectbox('Visualisasi Label',['Histogram','Pie Chart'],key=1)
Label=data['Label'].value_counts()
Label=pd.DataFrame({'Label':Label.index,'komentar':Label.values})
st.markdown("###  Visualisasi Label")
if select == "Histogram":
        fig = px.bar(Label, x='Label', y='komentar', color = 'komentar', height= 500)
        st.plotly_chart(fig)
else:
        fig = px.pie(Label, values='komentar', names='Label')
        st.plotly_chart(fig)

        
cek_datasets = st.sidebar.write('Thank you')
cek_datasets = st.sidebar.write('by : Karandi Nurbagja')
cek_datasets = st.sidebar.text('')
