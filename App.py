from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌱",
    layout="wide"
)



# Load the model
model = load_model("leaf_disease_model.keras")





st.markdown("""
<style>

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Dark Background */
.stApp{
    background-color:#0E1117;
    color:white;
}

.main-title{
    text-align:center;
    font-size:110px;
    font-weight:900;
    color:#4CAF50;
    letter-spacing:3px;
    text-shadow:0px 0px 20px rgba(76,175,80,0.45);
    margin-bottom:10px;
    line-height:1.1;
}

/* Subtitle */
.sub-title{
    text-align:right;
    color:#CFCFCF;
    font-size:20px;
    margin-bottom:30px;
}

/* Paragraphs */
p{
    color:white;
}

/* Card */
.card{
    background:#1C1F26;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.5);
}

/* Disease Result */
.prediction{
    background:#D4EDDA;
    color:black;              /* <-- Changed to black */
    padding:18px;
    border-radius:12px;
    border-left:8px solid #28A745;
    font-size:22px;
    font-weight:bold;
}

/* Confidence Result */
.confidence{
    background:#D1ECF1;
    color:black;              /* <-- Changed to black */
    padding:18px;
    border-radius:12px;
    border-left:8px solid #007BFF;
    font-size:20px;
    font-weight:bold;
}

/* Upload label */
label{
    color:white !important;
}

/* Horizontal line */
hr{
    border:1px solid #444;
}

</style>
""", unsafe_allow_html=True)















class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
# print("Model loaded successfully!")
# model.summary()


st.markdown("""
<h1 style="
text-align:center;
font-size:70px;
font-weight:900;
color:#4CAF50;
letter-spacing:3px;
text-shadow:0px 0px 20px rgba(76,175,80,0.5);
margin-bottom:0;
">
Plant Leaf Disease Detection 🌱
</h1>
""", unsafe_allow_html=True)
st.markdown(
'<p class="sub-title">Powered by Deep Learning •</p>',
unsafe_allow_html=True)
st.write("")
st.write("")
st.markdown(
    """
    This application uses a **Convolutional Neural Network (CNN)** to detect
    plant leaf diseases from uploaded images. Upload a clear leaf image to
    receive the predicted disease class.
    """
)

col1, col2 = st.columns([1,1])

with col1:

    uploaded_file = st.file_uploader(
        "📂 Upload Leaf Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image,width=300)

with col2:

    st.markdown("### Prediction Result")

    if uploaded_file:

        if st.button("🔍 Predict Disease",use_container_width=True):

            with st.spinner("Analyzing Leaf..."):

                img=image.resize((224,224))
                img=np.array(img)

                if img.shape[-1]==4:
                    img=img[:,:,:3]

                img = img.astype("float32")
                img=np.expand_dims(img,axis=0)

                prediction=model.predict(img)

                predicted_index=np.argmax(prediction)
                confidence=np.max(prediction)*100

            st.balloons()

            st.markdown(
                f"""
                <div class="prediction">
                🌿 Disease<br><br>
                {class_names[predicted_index]}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>",unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="confidence">
                🎯 Confidence<br><br>
                {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("---")

st.markdown(
"""
<center>
Project is made using limited Data and Epochs because of CPU constraints and hence may predict wrong sometimes
©Developed By Nihal Tiwari
</center>
""",
unsafe_allow_html=True
)