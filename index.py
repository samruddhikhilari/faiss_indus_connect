import os
import streamlit as st
import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
VECTOR_DB_PATH = "."
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Page configuration
st.set_page_config(
    page_title="Indus Connect Club Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #2d2d2d;
        padding: 2rem;
        border-radius: 10px;
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        color: white;
    }
    
    .header-logo {
        width: 80px;
        height: 80px;
        margin-right: 2rem;
        flex-shrink: 0;
    }
    
    .header-content {
        flex: 1;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        max-width: 80%;
    }
    .user-message {
        background-color: #3a3a3a;
        margin-left: 20%;
        border-left: 4px solid #2196f3;
        color: #ffffff;
    }
    .bot-message {
        background-color: #2a4a2a;
        margin-right: 20%;
        border-left: 4px solid #4caf50;
        color: #ffffff;
    }
    .source-doc {
        background-color: #404040;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        margin-top: 0.5rem;
        color: #e0e0e0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a1a;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #3a3a3a;
        color: #ffffff;
        border: 1px solid #555555;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #4a4a4a;
        color: #ffffff;
        border: 1px solid #666666;
    }
    
    .stButton > button:hover {
        background-color: #5a5a5a;
        border-color: #777777;
    }
</style>
""", unsafe_allow_html=True)

class IndusConnectChatbot:
    def __init__(self):
        self.vector_db = None
        self.qa_chain = None
        self.embedder = None
        
        # Custom prompt template
        self.custom_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are a helpful assistant for Indus Connect Club FAQ. Based only on the below context, answer the user's question concisely and helpfully.

If you can't find the answer in the context, respond with:
"Sorry for the inconvenience. For now, we can't answer this question. For more info, you can contact the club community members."

Context:
{context}

Question: {question}

Answer:
""".strip()
        )
    
    def initialize_embeddings(self):
        """Initialize HuggingFace embeddings"""
        try:
            if not self.embedder:
                self.embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
                logger.info("✅ Embeddings initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error initializing embeddings: {str(e)}")
            return False
    
    def initialize_llm(self):
        """Initialize Google Gemini LLM"""
        try:
            api_key = "AIzaSyCd88Qhb_l6I3udt5CR2zeEsS07fwt-8o0"
            self.llm = ChatGoogleGenerativeAI(
                model="models/gemini-1.5-flash-latest",
                google_api_key=api_key,
                temperature=0.7
            )
            logger.info("✅ LLM initialized successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Error initializing LLM: {str(e)}")
            return False
    
    def create_sample_data(self):
        """Create sample CSV data for demonstration"""
        sample_data = {
            "Question": [
                "What is Indus Connect (Icon) Club?",
                "Which college started the Indus Connect Club?",
                "What is the main aim of the Indus Connect Club?",
                "What makes the Indus Connect Club unique?",
                "When was the Indus Connect Club established?",
                "What type of events or activities does the club organize?",
                "Why should I join the Indus Connect Club?",
                "What is the vision of the Indus Connect Club?",
                "What is the mission of the Indus Connect Club?",
                "How does the club prepare students for future careers?"
            ],
            "Answer": [
                "It is a startup club initiated by the AIML department of Vishwakarma Institute of Technology in Feb 2024 to bridge the gap between theoretical learning and practical industry skills.",
                "The club was started by the AIML department of Vishwakarma Institute of Technology (VIT), Pune.",
                "To provide students with real-world project experience and exposure to corporate environments, enhancing their employability.",
                "It focuses on bridging the gap between academic theory and practical industry application through immersive experiences.",
                "The club was established in February 2024.",
                "The club organizes technical, management, creative, and project-based events to provide hands-on experience.",
                "Joining provides opportunities to connect with industry experts, work on real-world projects, and gain valuable experience for the job market.",
                "To be a leading platform that integrates academic learning with industry practices, preparing members to thrive in corporate environments.",
                "To bridge the gap between academia and industry by offering hands-on, corporate-like experiences to students.",
                "By offering real-world projects and industry exposure, equipping students with critical experience and skills for future success."
            ]
        }
        return pd.DataFrame(sample_data)
    
    def load_or_create_vector_store(self, csv_data=None, csv_file=None):
        """Load existing vector store or create new one"""
        try:
            if not self.initialize_embeddings():
                return False
            
            # Check if existing vector store exists
            if os.path.exists(VECTOR_DB_PATH):
                try:
                    self.vector_db = FAISS.load_local(
                        VECTOR_DB_PATH, 
                        self.embedder, 
                        allow_dangerous_deserialization=True
                    )
                    logger.info("📂 Loaded existing FAISS index")
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ Could not load existing index: {str(e)}")
            
            # Create new vector store
            logger.info("📄 Creating new FAISS index...")
            
            if csv_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
                    csv_data.to_csv(tmp_file.name, index=False)
                    temp_path = tmp_file.name
                
                loader = CSVLoader(file_path=temp_path, source_column="Question")
                documents = loader.load()
                os.unlink(temp_path)  # Clean up temp file
                
            elif csv_data is not None:
                # Use provided DataFrame
                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
                    csv_data.to_csv(tmp_file.name, index=False)
                    temp_path = tmp_file.name
                
                loader = CSVLoader(file_path=temp_path, source_column="Question")
                documents = loader.load()
                os.unlink(temp_path)  # Clean up temp file
                
            else:
                # Use sample data
                sample_df = self.create_sample_data()
                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_file:
                    sample_df.to_csv(tmp_file.name, index=False)
                    temp_path = tmp_file.name
                
                loader = CSVLoader(file_path=temp_path, source_column="Question")
                documents = loader.load()
                os.unlink(temp_path)  # Clean up temp file
            
            if not documents:
                logger.error("❌ No documents loaded")
                return False
            
            self.vector_db = FAISS.from_documents(documents, self.embedder)
            self.vector_db.save_local(VECTOR_DB_PATH)
            logger.info(f"✅ Created FAISS index with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating vector store: {str(e)}")
            return False
    
    def setup_qa_chain(self):
        """Setup the RetrievalQA chain"""
        try:
            if not self.vector_db or not self.llm:
                return False
            
            retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": self.custom_prompt},
                return_source_documents=True
            )
            logger.info("✅ QA Chain setup successful")
            return True
        except Exception as e:
            logger.error(f"❌ Error setting up QA chain: {str(e)}")
            return False
    
    def get_answer(self, query: str) -> Dict[str, Any]:
        """Get answer for a query"""
        try:
            if not self.qa_chain:
                return {
                    "answer": "❌ Chatbot not properly initialized. Please check your setup.",
                    "sources": [],
                    "error": True
                }
            
            response = self.qa_chain.invoke({"query": query})
            
            sources = []
            for doc in response.get("source_documents", []):
                sources.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get('source', 'unknown')
                })
            
            return {
                "answer": response["result"],
                "sources": sources,
                "error": False
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting answer: {str(e)}")
            return {
                "answer": f"❌ Error processing query: {str(e)}",
                "sources": [],
                "error": True
            }

def main():
    # Header with VIT logo
    st.markdown("""
    <div class="main-header">
        <img src="vit_logo.png" class="header-logo" alt="VIT Logo">
        <div class="header-content">
            <h1>🤖 Indus Connect Club Chatbot</h1>
            <p>Your AI assistant for Indus Connect Club FAQ | VIT, Pune</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = IndusConnectChatbot()
        st.session_state.initialized = False
        st.session_state.chat_history = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key is hardcoded
        st.info("🔑 API Key: Configured")
        
        # CSV file upload
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload FAQ CSV file",
            type="csv",
            help="Upload a CSV with 'Question' and 'Answer' columns"
        )
        
        # Use sample data option
        use_sample = st.checkbox(
            "Use sample data",
            value=True,
            help="Use built-in sample data for demonstration"
        )
        
        # Initialize button
        if st.button("🚀 Initialize Chatbot", type="primary"):
            with st.spinner("Initializing chatbot..."):
                # Initialize LLM
                if not st.session_state.chatbot.initialize_llm():
                    st.error("❌ Failed to initialize LLM")
                else:
                    # Load data
                    csv_data = None
                    if uploaded_file:
                        csv_data = pd.read_csv(uploaded_file)
                    elif use_sample:
                        csv_data = st.session_state.chatbot.create_sample_data()
                    
                    # Create vector store
                    if st.session_state.chatbot.load_or_create_vector_store(csv_data):
                        if st.session_state.chatbot.setup_qa_chain():
                            st.session_state.initialized = True
                            st.success("✅ Chatbot initialized successfully!")
                        else:
                            st.error("❌ Failed to setup QA chain")
                    else:
                        st.error("❌ Failed to create vector store")
        
        # Status indicator
        if st.session_state.initialized:
            st.success("🟢 Chatbot Ready")
        else:
            st.warning("🟡 Chatbot Not Initialized")
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {question}
                </div>
                """, unsafe_allow_html=True)
                
                # Bot message
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>Indus Connect Bot:</strong> {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Sources (if any)
                if sources:
                    with st.expander(f"📚 View Sources for Question {i+1}"):
                        for j, source in enumerate(sources):
                            st.markdown(f"""
                            <div class="source-doc">
                                <strong>Source {j+1}:</strong> {source['content']}
                            </div>
                            """, unsafe_allow_html=True)
        
        # Query input
        query = st.text_input(
            "Ask a question about Indus Connect Club:",
            placeholder="e.g., When was the club established?",
            disabled=not st.session_state.initialized
        )
        
        col_send, col_example = st.columns([1, 2])
        
        with col_send:
            send_button = st.button(
                "Send 📤",
                disabled=not st.session_state.initialized or not query
            )
        
        with col_example:
            if st.button("Try Example: Club Establishment"):
                query = "When was the Indus Connect Club established?"
                send_button = True
        
        # Process query
        if send_button and query and st.session_state.initialized:
            with st.spinner("🤔 Thinking..."):
                result = st.session_state.chatbot.get_answer(query)
                
                # Add to chat history
                st.session_state.chat_history.append((
                    query,
                    result["answer"],
                    result["sources"]
                ))
                
                st.rerun()
    
    with col2:
        st.subheader("📊 Statistics")
        
        # Chat statistics
        total_queries = len(st.session_state.chat_history)
        st.metric("Total Queries", total_queries)
        
        if total_queries > 0:
            successful_queries = sum(1 for _, answer, _ in st.session_state.chat_history 
                                   if not answer.startswith("❌"))
            success_rate = (successful_queries / total_queries) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What is Indus Connect Club?",
            "When was the club established?",
            "What activities does the club organize?",
            "Why should I join the club?",
            "What is the club's mission?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}"):
                if st.session_state.initialized:
                    result = st.session_state.chatbot.get_answer(question)
                    st.session_state.chat_history.append((
                        question,
                        result["answer"],
                        result["sources"]
                    ))
                    st.rerun()
                else:
                    st.warning("Please initialize the chatbot first!")

if __name__ == "__main__":
    main()