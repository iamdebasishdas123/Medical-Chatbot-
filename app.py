import streamlit as st
import base64
import os
from groq import Groq # type: ignore
import pyttsx3
from langchain_core.prompts import PromptTemplate,MessagesPlaceholder,ChatPromptTemplate
from dotenv import load_dotenv # type: ignore
from langchain_groq import ChatGroq # type: ignore
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from pydantic import Field
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel
import logging
import json
from faiss import IndexFlatL2 # type: ignore
import torch
# torch.classes.load_library("__dummy__")  # Workaround for torch path error


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)





os.makedirs("vectorstore/db_faiss", exist_ok=True)


load_dotenv() # type: ignore



CUSTOM_PROMPT_TEMPLATE = PromptTemplate(
    template="""Use the context below to answer the question. 
    Context: {context}
    Question: {question}
    Answer directly without preamble. If unsure and results is not in database then say you don't know becasue i dont have Data for this Disease.""",
    input_variables=["context", "question"]  # Add context here
)


class Disease_found(BaseModel):
    """Model for chatbot response"""
    Disease: Literal["Yes","No"] 
    
schema = [
    ResponseSchema(name='Disease_1', description='1st Most recommeneded Disease',type="string"),
    ResponseSchema(name='Disease_2', description='2nd Most recommeneded Disease',type="string"),
    ResponseSchema(name='Disease_3', description='3rd Most recommeneded Disease',type="string")
]

parser= StructuredOutputParser.from_response_schemas(schema)

DB_FAISS_PATH="vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    if not os.path.exists(DB_FAISS_PATH):
        # Create empty FAISS index with correct dimensions

        index = IndexFlatL2(384)  # Dimension matching all-MiniLM-L6-v2 embeddings
        db = FAISS(embedding_model.embed_query, index, None, {})
        db.save_local(DB_FAISS_PATH)
    
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


def get_retriever_llm(text, model):
    """Get retriever for LLM model"""
    # Define prompts
    prompt1 = PromptTemplate(template="Analyze the following text: {text}. Determine if any disease is mentioned. Respond with 'Yes' if a disease is found and 'No' otherwise.")
    prompt2 = PromptTemplate(template="Find three diseases related to {text}.\n\n{format_instruction}",
                              input_variables=["text"], partial_variables={"format_instruction": parser.get_format_instructions()})
    
    # Structured output model
    structured_model = model.bind_tools([Disease_found], tool_choice="Disease_found")
    
    # Chain 1: Check for disease presence
    chain1 = prompt1 | structured_model
    
    try:
        response = chain1.invoke({"text": text})
        # print("Response Disease or Not: %s", response)
        tool_calls = response.additional_kwargs.get('tool_calls', [])
        if not tool_calls:
            logger.error("No tool calls in response")
            return {"answer": "Analysis failed"}
        args = json.loads(tool_calls[0]['function']['arguments'])
        parsed_response = Disease_found(**args)
        # print("Parsed Response:", parsed_response)
    except Exception as e:
        logger.error("Error during chain execution: %s", e)
        return {"answer": "An error occurred while processing your request."}
    
    if parsed_response.Disease == "Yes":
        # Chain 2: Find related diseases
        print("Debasish")
        chain2 = prompt2 | model | parser
        
        try:
            response = chain2.invoke({"text": text})
            logger.info("Response 3 related Diseases: %s", response)
        except Exception as e:
            logger.error("Error during chain execution: %s", e)
            return {"error": "An error occurred while processing your request."}
        
        # Retrieve information for each disease
        results_store = []
        db = get_vectorstore()
        
        for disease in response.values():
            RQA = RetrievalQA.from_chain_type(
                llm=model,
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={
                    'prompt': CUSTOM_PROMPT_TEMPLATE,
                    'document_variable_name': 'context'  # Explicitly set document variable
                }
            )

            
            try:
                # Invoke RQA with the query
                results = RQA.invoke({"query": f"How to cure {disease}?"})
                print("Result:", results)
                
                # Check if 'answer' and 'source_documents' keys exist in result
                if 'result' in results and 'source_documents' in results:
                    source_documents = results["source_documents"]
                    results_store.append({
                        "disease": disease,
                        "answer": results["result"],
                        "source_documents": source_documents
                    })
                else:
                    # Handle missing keys gracefully
                    print(f"Error: Missing expected keys in result for disease '{disease}'.")
            except Exception as e:
                # Handle any other errors during retrieval
                print(f"ERROR: Error during retrieval for disease '{disease}': {e}")
                print("results",results_store)
        
        return results_store
    
    else:
        print("No disease found in the text.")
        return {"message": "Sorry, I cannot find any disease in the text."}

        


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content='You are a helpful Medical AI assistant. You have access to a medical image and can provide analysis and recommendations based on the image and user input.'),]  # Persistent chat history


# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# App configuration
st.set_page_config(page_title="AI Doctor Chat", page_icon="🩺")
st.title("AI Doctor Assistance")
st.sidebar.header("Medical Tools")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_model = ChatGroq(model="llama-3.1-8b-instant")
    

def handle_image_upload():
    """
    Process image upload and text input from sidebar with a submit button.
    Allows users to upload medical images and provide additional context or symptoms.
    """

    # File uploader for medical images
    uploaded_file = st.sidebar.file_uploader(
        "Upload Medical Image",
        type=["jpg", "png", "jpeg"],
        help="Upload medical scans or photos for analysis"
    )

    # Text input for symptoms/questions in sidebar
    user_text = st.sidebar.text_area(
        "Additional Context/Symptoms",
        placeholder="Describe symptoms or ask specific questions about the image",
        height=100
    )

    # Submit button to confirm upload and text input
    submit_button = st.sidebar.button("Submit")

    # Process the uploaded file and user text after submission
    if submit_button:
        if uploaded_file:
            # Read and encode the uploaded image
            image_bytes = uploaded_file.getvalue()
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            return encoded_image, user_text
        else:
            st.sidebar.warning("Please upload an image before submitting.")
    
    return None, user_text


def display_chat_history():
    """Display chat messages from history"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("image"):
                st.image(base64.b64decode(message["image"]), caption="Medical Image", width=300)
            else:
                st.markdown(message["content"])
                
            if message.get("audio"):
                st.audio(message["audio"], format='audio/mp3')

def analyze_image_with_query(query, model, encoded_image=None):
    """Send query to Groq API with optional image"""
    messages = [{"role": "user", "content": []}]
    
    # Add text content if provided
    if query:
        messages[0]["content"].append({"type": "text", "text": query})
    
    # Add image content if provided
    if encoded_image:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
        })
    
    # Handle empty request case
    if not messages[0]["content"]:
        return "Error: No input provided"
    
    # Get API response
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )
    return chat_completion.choices[0].message.content

def analyze_chat_with_llm(prompt_temp,prompt,model,chat_history):
    """Send prompt to LLM and return response"""
    chain = prompt_temp | model
    response = chain.invoke({
        "query": prompt,
        "chat_history": chat_history
    })
    return response.content

def text_to_speech(text):
    """Convert text to audio using Pyttsx3 (offline)"""
    engine = pyttsx3.init()
    engine.save_to_file(text, 'output.mp3')
    engine.runAndWait()
    with open('output.mp3', 'rb') as audio_file:
        return audio_file.read()



def main():
    # Initialize session state for image analysis flag
    if "image_analyzed" not in st.session_state:
        st.session_state.image_analyzed = False

    # Sidebar controls
    with st.sidebar:
        encoded_image, user_text = handle_image_upload()
        enable_voice = st.checkbox("Enable Voice Output", value=True)
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.image_analyzed = False

    # Run initial image analysis only once when image is uploaded
    if encoded_image and not st.session_state.image_analyzed:
        vision_model = "llama-3.2-90b-vision-preview"
        
        # Combine user text with analysis protocol
        analysis_prompt = f"""USER CONTEXT: {user_text}

        As a diagnostic imaging specialist AI, carefully analyze the provided medical image following this protocol:
        Output will be :
        1. **Structured Observation Report**  
           a) Anatomical structures: Identify visible organs/tissues  
           b) Abnormalities: Describe size, shape, density, margins  
           c) Urgency indicators: Highlight acute findings

        2. **potential Disease**  
           - tell me possible disease based on the image

        3. **Recommendations test and Food **
           - Suggest further imaging or tests 
           - Recommend dietary changes or supplements
           - Provide lifestyle changes or medications
           - Specify limitations  

        **Safety Protocols:**  
        - Avoid absolute diagnostic terms  
        - Flag uncertain findings for radiologist review"""

        with st.spinner("Analyzing medical image..."):
            response = analyze_image_with_query(analysis_prompt, vision_model, encoded_image)
            
            # Store in chat history
            st.session_state.messages.append({
                "role": "user",
                "content": user_text,
                "image": encoded_image
            })
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "audio": text_to_speech(response) if enable_voice else None
            })
            st.session_state.chat_history.append(HumanMessage(content=user_text))
            
            st.session_state.chat_history.append(AIMessage(content=response))
            # print("Chat History Vision:", st.session_state.chat_history)
            
            st.session_state.image_analyzed = True
    
    # print(st.session_state.chat_history)
            
            db=get_retriever_llm(st.session_state.chat_history,st.session_state.chat_model)
            all_results = []
            if "message" not in db:
                for entry in db:
                    disease = entry['disease']
                    answer = entry['answer']
                    source_docs = entry['source_documents']
                    result_to_show=disease.upper()+"\n"+answer+"\nSource Docs:\n"+source_docs[0].metadata["file_path"]+"\n"
                    all_results.append(result_to_show)
                        
                st.session_state.messages.append({
                    "role": "Database",
                    "content": (
    str(all_results[0]) + "\n" + 
    str(all_results[1]) + "\n" +  
    str(all_results[2])          
                    ),
                    "audio": text_to_speech((
    str(all_results[0]) + "\n" + 
    str(all_results[1]) + "\n" +  
    str(all_results[2])          
                    )) if enable_voice else None
                }) 
            else:
                st.session_state.messages.append({
                    "role": "Database",
                    "content": db["message"],
                    "audio": None
                })          
            

    # Display chat history
    display_chat_history()
    
    # Chat input and processing (runs in loop)
    if prompt := st.chat_input("Describe your symptoms or ask a question..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response
        with st.spinner("Analyzing..."):
            chat_template =ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a helpful medical assistant."),
                MessagesPlaceholder(variable_name="chat_history"),
                ('user', "{query}")
            ])
            
            response = analyze_chat_with_llm(
                chat_template, 
                prompt, 
                st.session_state.chat_model, 
                st.session_state.chat_history
            )
            
            # Update chat history
            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=response))
            
            # Generate audio if enabled
            audio_bytes = text_to_speech(response) if enable_voice else None
            
            # Add AI response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "audio": audio_bytes
            })
        print("Chat History llm",st.session_state.chat_history)
            
        # Rerun to update chat display
        st.rerun()


if __name__ == "__main__":
    main()

