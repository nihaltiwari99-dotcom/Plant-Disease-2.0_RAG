from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np

# ---------------- LANGCHAIN IMPORTS ----------------
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os
load_dotenv()

st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌱",
    layout="wide"
)
model_path = "leaf_disease_model.keras"
model = load_model(model_path)

# ---------------- RAG PIPELINE ----------------
@st.cache_resource
def load_rag_chain():

    # Load document
    loader = Docx2txtLoader("Plant_Disease_Guide_Improved_RAG.docx")
    documents = loader.load()

    # Split document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n"],
    )

    docs = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load/Create Chroma DB
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="chroma_db",
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        },
    )

    # API Key
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        st.error("❌ GROQ_API_KEY not found.")
        st.stop()

    # LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=groq_api_key,
    )

    prompt = ChatPromptTemplate.from_template("""
You are an agricultural expert.

Use the retrieved context to answer the user's question.

If the context contains the requested disease, answer it directly.

If the retrieved context contains multiple diseases, identify the disease that best matches the user's question and ignore the others.

Never say the answer is unavailable if the requested disease appears anywhere in the context.

Only reply with "I couldn't find that information in the provided documents." if the requested disease is completely absent from the context.

Give clear and concise answers.

Context:
{context}

Question:
{question}

Answer:
""")

    # Format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # LangChain 1.x LCEL chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
rag_chain = load_rag_chain()

st.markdown("""
<style>

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Dark Background */
.stApp{
    background-color:white;
    color:white;
}

.main-title{
    text-align:center;
    font-size:110px;
    font-weight:900;
    color:black;
    letter-spacing:3px;
    text-shadow:0px 0px 20px rgba(76,175,80,0.45);
    margin-bottom:10px;
    line-height:1.1;
}

/* Subtitle */
.sub-title{
    text-align:right;
    color:black;
    font-size:20px;
    margin-bottom:30px;
}

.sub-title2{
    text-align:left;
    color:black;
    font-size:20px;
    margin-bottom:30px;
}





/* Paragraphs */
p{
    color:black;
}

/* Card */
.card{
    background:white;
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
    color:black !important;
}

/* Horizontal line */
hr{
    border:1px solid #444;
}

/* App Background */
.stApp{
    background-color: white;
    color: black;
}

/* Make all normal text black */
html, body, [class*="css"]{
    color: black;
}

/* Streamlit Markdown/Text */
.stMarkdown,
.stText,
.stCaption,
.stInfo,
.stSuccess,
.stWarning,
.stError{
    color: black !important;
}

/* Text input */
.stTextInput label{
    color: black !important;
}

.stTextInput input{
    color: black !important;
    background-color: white !important;
    border: 1px solid #ccc;
}

/* Placeholder */
.stTextInput input::placeholder{
    color: #666 !important;
}

/* File uploader */
.stFileUploader label{
    color: black !important;
}

/* Info/Warning boxes */
.stAlert{
    color: black !important;
}

/* Buttons */
.stButton button{
    color: white;
    background-color: #4CAF50;
    border-radius: 8px;
    border: none;
}

.stButton button:hover{
    background-color: #45a049;
}

/* Make spinner text visible */
.stSpinner{
    color: black !important;
}

/* AI Answer */
.stChatMessage,
.stMarkdown p,
div[data-testid="stMarkdownContainer"]{
    color: black !important;
}

/* Success/Info text */
div[data-testid="stNotificationContent"]{
    color: black !important;
}


/* File uploader box */
[data-testid="stFileUploaderDropzone"]{
    background-color: #E8F5E9 !important;
    border: 2px dashed #4CAF50 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    color: black !important;
}

/* Upload icon */
[data-testid="stFileUploaderDropzone"] svg{
    fill: #4CAF50 !important;
}

/* Upload text */
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span{
    color: black !important;
    font-weight: 600;
}

/* Browse files button */
[data-testid="stBaseButton-secondary"]{
    background-color: #4CAF50 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}

[data-testid="stBaseButton-secondary"]:hover{
    background-color: #43A047 !important;
}

</style>
""", unsafe_allow_html=True)















class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']



st.markdown("""
<h1 style="
text-align:center;
font-size:65px;
font-weight:900;
color:#4CAF50;
letter-spacing:3px;
text-shadow:0px 0px 20px rgba(76,175,80,0.5);
margin-bottom:0;
">
 Plant Leaf Disease Detection 2.0 🌱
</h1>
""", unsafe_allow_html=True)
st.markdown("""
<div style="
    display:flex;
    justify-content:center;
    align-items:center;
    gap:40px;
    font-size:18px;
    color:black;
    margin-top:10px;
    margin-bottom:25px;
    font-weight:500;
">
    <span>🧠 <b>Powered by</b> Deep Learning</span>
    <span>✨ <b>Assisted by</b> Groq Llama 3.3</span>
    <span>👨‍💻 <b>Developed by</b> Nihal Tiwari</span>
</div>
""", unsafe_allow_html=True)
st.write("")
st.write("")
st.markdown(
    """
    This application uses a **Convolutional Neural Network (CNN)** to detect
    plant leaf diseases from uploaded images. Upload a clear leaf image to
    receive the predicted disease class. Use AI to answer your queries about the predicted disease.
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


                predicted_index = np.argmax(prediction)
                confidence = np.max(prediction) * 100

                # Save prediction for RAG
                predicted_disease = class_names[predicted_index]
                st.session_state["predicted_disease"] = predicted_disease

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

# ---------------- QUESTION BOX ----------------

if "predicted_disease" in st.session_state:

    st.markdown("## 💬 Ask a Question About the Predicted Disease")

    user_question = st.text_input(
        "Enter your question",
        placeholder="What causes this disease and how can I treat it?"
    )

    if st.button("🔍 Ask AI", use_container_width=True):

        if user_question.strip():

            with st.spinner("Searching knowledge base and generating answer..."):
                disease = st.session_state["predicted_disease"]
                disease = disease.replace("___", " ").replace("_", " ")

                query = f"""
                Predicted Disease: {disease}

                User Question: {user_question}

                If the user asks about another disease instead of the predicted disease,
                answer about the disease mentioned in the user's question.
                """

                response = rag_chain.invoke(query)
                st.markdown("### 🤖 AI Answer")
                st.write(response)



        else:
            st.warning("Please enter a question.")

else:
    st.info("Chat Bot(Predict first to use)")








st.markdown("---")
st.markdown(
    """
    <center>
    CNN + LangChain (RAG) + ChromaDB + Groq API
    </center>
    """,
    unsafe_allow_html=True
)
