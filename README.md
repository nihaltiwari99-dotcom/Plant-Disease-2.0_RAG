# 🌱 Plant Disease Detector 2.0 with AI Assistance

An intelligent web application that combines **Deep Learning** and **Retrieval-Augmented Generation (RAG)** to detect plant leaf diseases and provide AI-powered answers about the detected disease.

The application uses a **Convolutional Neural Network (CNN)** for image classification and integrates **LangChain**, **ChromaDB**, and **Groq Llama 3.3** to answer user queries based on a curated plant disease knowledge base.

---

## ✨ Features

* 🌿 Detects plant leaf diseases from uploaded images
* 🧠 CNN model built with TensorFlow/Keras
* 📊 Displays prediction confidence
* 🤖 AI-powered question answering using RAG
* 📚 Uses a plant disease knowledge base (.docx)
* ⚡ Fast inference using Groq Llama 3.3
* 💾 Automatic Chroma vector database generation
* 🎨 Clean and interactive Streamlit interface

---

## 🛠️ Tech Stack

### Deep Learning

* TensorFlow
* Keras
* NumPy

### Frontend

* Streamlit

### RAG Pipeline

* LangChain
* ChromaDB
* HuggingFace Embeddings
* Groq API (Llama 3.3 70B)
* Docx2txt

### Vector Embeddings

* sentence-transformers/all-MiniLM-L6-v2

---

## 📂 Project Structure

```text
Plant-Disease-Detector-2.0/
│
├── main.py
├── requirements.txt
├── runtime.txt
├── leaf_disease_model.keras
├── Plant_Disease_Guide_Improved_RAG.docx
├── chroma_db/              # Auto-generated on first run
├── .env
└── README.md
```

---

## 🚀 How It Works

### Step 1 – Upload Image

Upload a leaf image (.jpg, .jpeg, .png).

↓

### Step 2 – Disease Prediction

The CNN model predicts the plant disease and confidence score.

↓

### Step 3 – Knowledge Retrieval

Relevant disease information is retrieved from the knowledge base using ChromaDB.

↓

### Step 4 – AI Response

Groq Llama 3.3 generates an answer based only on the retrieved documents.

---

## 📸 Application Workflow

```text
Leaf Image
      │
      ▼
TensorFlow CNN
      │
      ▼
Predicted Disease
      │
      ▼
User Question
      │
      ▼
LangChain Retriever
      │
      ▼
ChromaDB
      │
      ▼
Relevant Context
      │
      ▼
Groq Llama 3.3
      │
      ▼
AI Answer
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Plant-Disease-Detector-2.0.git
cd Plant-Disease-Detector-2.0
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

Get a free API key from:

https://console.groq.com/keys

---

## ▶️ Run the Application

```bash
streamlit run main.py
```

---

## 💬 Example Questions

* What causes this disease?
* What are the symptoms?
* How can I treat it?
* How can I prevent it?
* What fungicides are recommended?
* Is this disease contagious?
* Which season is this disease common?
* How does this disease spread?

---

## 📌 Supported Plant Diseases

The model supports diseases from multiple crops including:

* Apple
* Cherry
* Corn
* Grape
* Orange
* Peach
* Pepper
* Potato
* Raspberry
* Soybean
* Squash
* Strawberry
* Tomato
* Blueberry

along with healthy leaf classification.

---

## 🧠 AI Assistant

The AI assistant uses a Retrieval-Augmented Generation (RAG) pipeline instead of relying solely on the LLM's knowledge.

Pipeline:

```text
Knowledge Document
        │
        ▼
Text Splitter
        │
        ▼
HuggingFace Embeddings
        │
        ▼
Chroma Vector Database
        │
        ▼
Retriever
        │
        ▼
Groq Llama 3.3
        │
        ▼
Final Answer
```

---

## 📈 Future Improvements

* 🎥 Video-based disease detection
* 📱 Mobile-friendly interface
* 🌍 Multi-language support
* 📄 PDF report generation
* 🔊 Voice-based interaction
* ☁️ Cloud database for knowledge storage
* 📸 Real-time camera prediction
* 🌾 More crop and disease support

---

## 👨‍💻 Developer

**Nihal Tiwari**

If you found this project helpful, consider giving it a ⭐ on GitHub.

---

## 📄 License

This project is intended for educational and research purposes.
