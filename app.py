import pandas as pd
import numpy as np
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from transformers import BitsAndBytesConfig, AutoTokenizer
import torch

filenames = []

documents = []


def load_data(uploaded_file):
    if not uploaded_file:

        raise ValueError("No se hay archivos para cargar.")
    if not uploaded_file.name.lower().endswith('.pdf'):
        raise ValueError("Error: El archivo debe ser un PDF.")
    
    
    temp_path = None
    try:
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        loaded_document_pages = loader.load()

        if not loaded_document_pages:
            raise ValueError(f"El PDF '{uploaded_file.name}' está vacio o no se pudo extraer texto.")
            
        # 4. Enriquecer los metadatos (lógica corregida)
        base_filename = uploaded_file.name
        year_extracted = "Unknown"
        # Extraer año (tu lógica está bien, la mantenemos)
        try:
            parts = base_filename.split('-')
            if len(parts) > 1:
                potential_year = parts[1][:4]
                if potential_year.isdigit() and len(potential_year) == 4:
                    year_extracted = potential_year
        except Exception:
            pass # Si falla, se queda como "Unknown"

        # CORRECCIÓN IMPORTANTE: Itera y modifica cada página individualmente.
        for page in loaded_document_pages:
            page.metadata["source"] = base_filename
            page.metadata["year"] = year_extracted
            
        print(f"\nTotal de páginas cargadas y procesadas: {len(loaded_document_pages)}")

        vector_store, num_chunks = create_vectorstore_func(loaded_document_pages)

        return vector_store, num_chunks
        
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

#EMBEDDINGS Y VECTOR STORE
def create_vectorstore_func(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunked_docs = text_splitter.split_documents(documents)
    print(f"Numero total de chunks creados: {len(chunked_docs)}")

    embedding_function = SentenceTransformerEmbeddings(
    model_name="BAAI/bge-base-en-v1.5", 
    model_kwargs={'device': 'cpu'}, 
    encode_kwargs={'normalize_embeddings': True} 
    ) 
    
    try:
        valid_docs = [doc for doc in chunked_docs if doc.page_content.strip()]
        
        if not valid_docs:
            raise ValueError("No hay documentos validos para procesar")

        vector_store = Chroma.from_documents(
            documents=valid_docs,
            embedding=embedding_function,
            persist_directory=None  # En memoria
        )
        
        print(f"Vector store creado con {len(valid_docs)} documentos")
        return vector_store, len(valid_docs)
        
    except Exception as e:
        print(f"Error creando vector store: {e}")
        raise  

# quantization_config_8bit_with_offload = BitsAndBytesConfig(
#     load_in_8bit=True,
#     llm_int8_enable_fp32_cpu_offload=True,
# )


#Modelo

@st.cache_resource
def load_model():
    model_id = "google/gemma-2-2b-it"
    quantization_config_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        quantization_config=quantization_config_4bit,

    )
    print("Modelo cargado")
    if torch.cuda.is_available():
        model.to('cuda')
        print(f" Modelo movido a la GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No se encontró GPU. El modelo permanecera en la CPU.")


    # model = model.to_empty(device="cuda")
    # model.tie_weights()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    text_pipeline = pipeline(
        "text-generation",
        model=model,          
        tokenizer=tokenizer,   
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    llm = HuggingFacePipeline(pipeline=text_pipeline)
    return llm

def qa_chain(vector_store, llm):


    retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

    chain = RetrievalQA.from_chain_type(
        llm=llm,                                  
        chain_type="stuff",
        retriever=retriever,                         
        return_source_documents=True,                
    )
    return chain

def prompt_value_to_str(prompt_value) -> str:
  if hasattr(prompt_value, 'to_string'):
    print(f"Advertencia: Error ({type(prompt_value)}) en extract_text_from_prompt_value. Intentando convertir a str.")
    return prompt_value.to_string()
  return str(prompt_value)

def query(qa_chain, question):
    try:
        retriever = qa_chain.retriever
        retrieved_docs = retriever.invoke(question)

        respuesta = qa_chain.invoke({"query": question})
        answer = respuesta.get('result', 'No se pudo obtener una respuesta.')
        source_docs = respuesta.get('source_documents', [])

        print(f"\nPregunta: {question}")
        print(f"\nRespuesta: {answer}")
        
        if retrieved_docs:
            print("El retriever encontro documentos")
            for i, doc in enumerate(retrieved_docs):
                print(f"\n--- Documento {i+1} (Página: {doc.metadata.get('page', 'N/A')}) ---")
                print(doc.page_content[:300] + "...") 
            
        print("="*50 + "\n")
        return respuesta

    except Exception as e:
        print(f"Error en consulta: {str(e)}")
        raise

def main():
    st.set_page_config(
        page_title="Consultas PDF con IA",
        page_icon="📄",
        layout="wide"
    )
    llm = load_model()

    
    template="""
      Eres un bot servicial e informativo que hace resumenes de los PDF que el USUARIO ingresa.
      Tu tarea requiere de encontrar la relacion semantica entre lo que el USUARIO ingresa y el PDF.
      Asegúrate de responder en una oración completa, sin repetir frases y proporcionando toda la información de fondo y su contexto relevante.
      Al finalizar cada respuesta debes preguntar por la siguiente solicitud al USUARIO y esperar una nueva pregunta.
      Por favor, responde en el idioma de la pregunta.

      PREGUNTA: {query}
      CONTEXT: {context}

      RESPUESTA:
    """
    prompt = PromptTemplate(
    template=template, 
    input_variables=["query", "contexy"] 
    )

    st.title("📄 Consulta tus PDFs")
    st.markdown("---")
    
    if 'answer' not in st.session_state:
        st.session_state.answer = ""

    col_izquierda, col_derecha = st.columns([1, 2])
    
    # --- COLUMNA IZQUIERDA: Cargar PDF ---
    with col_izquierda:
        st.header("1. Cargar PDF")
        uploaded_file = st.file_uploader(
            "Selecciona tu archivo PDF",
            type=['pdf'],
            help="Sube un archivo PDF para hacer consultas sobre su contenido"
        )
        
        if uploaded_file is not None:
            # Boton para procesar el archivo subido
            if st.button("Procesar PDF", type="primary"):
                with st.spinner("Procesando PDF..."):
                    try:
                        vector_store, chunks_count = load_data(uploaded_file)
                        qa = qa_chain(vector_store, llm)
                        st.session_state.qa_chain = qa
                        st.session_state.pdf_name = uploaded_file.name
                        st.session_state.answer = "" # Limpiar respuesta anterior
                        
                        st.success(f"PDF '{uploaded_file.name}' procesado")
                    
                    except Exception as e:
                        st.error(f"Error al procesar el PDF: {str(e)}")

    # --- COLUMNA DERECHA: Consultas y Respuestas ---
    with col_derecha:
        st.header("2. Realizar Consulta")
        
        # Solo mostrar el área de consulta si un PDF ha sido procesado
        if 'qa_chain' in st.session_state:
            st.info(f"{st.session_state.pdf_name} cargado")

            st.markdown("### Respuesta:")
            response_container = st.expander("Ver respuesta completa", expanded=True)
            with response_container:
                if st.session_state.answer:
                    st.write(st.session_state.answer)
                else:
                    st.info("La respuesta aparecerá aquí.")
                response_placeholder = st.empty()

            query_text = st.text_area(
                "Escribe tu pregunta aquí:",
                height=100,
                placeholder="Ejemplo: ¿Cuál es el tema principal del documento?"
            )
            
            # Boton de consulta
            if st.button("Consultar"):
                if query_text:
                    with st.spinner("Pensando..."):
                        try:
                            # Llamar a la funcion de consulta
                            response = query(st.session_state.qa_chain, query_text) 
                            answer_full = response.get('result', 'No se pudo obtener una respuesta.')
                            
                            # Buscar específicamente "Helpful Answer:" en la respuesta
                            if "Helpful Answer:" in answer_full:
                                # Extraer solo el texto después de "Helpful Answer:"
                                answer_clean = answer_full.split("Helpful Answer:")[1].strip()
                                st.session_state.answer = answer_clean
                            else:
                                # Si no encuentra "Helpful Answer:", usar la respuesta completa
                                st.session_state.answer = answer_full

                            # Actualizar el placeholder con la nueva respuesta
                            response_placeholder.markdown(st.session_state.answer)
                            
                        except Exception as e:
                            st.error(f"Error al procesar la consulta: {str(e)}")
                else:
                    st.warning("Por favor, escribe una pregunta.")
        else:
            st.warning("Primero debes cargar y procesar un archivo PDF en la columna de la izquierda.")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            💡 Interfaz simplificada para consultas
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
