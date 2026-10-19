class temps :
    
    def split_documents(docs, chunk_size, chunk_overlap):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        """Step 1: split documents into overlapping chunks."""
        splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_documents(docs)
   
   
    def load_documents(path: str):
     from langchain_community.document_loaders import PyPDFLoader
     from langchain_community.document_loaders import TextLoader
     """Load a PDF or text file into LangChain Documents."""
     if path.lower().endswith(".pdf"):
      return PyPDFLoader(path).load()
     return TextLoader(path, encoding="utf-8").load()

    def get_retriever(get_vector_store,k: int = 3):
     from langchain_classic.retrievers import MultiQueryRetriever
     from langchain_ollama import ChatOllama
     """Retriever you can plug into a RAG chain."""
     return MultiQueryRetriever.from_llm(retriever= get_vector_store.as_retriever(search_kwargs={"k": k}),llm=ChatOllama(model="llama3.2",temperature=0,disable_streaming=False))
    
    def get_BM25retriever(chunks,k: int = 3):
         from langchain_community.retrievers import BM25Retriever
         from langchain_classic.retrievers import MultiQueryRetriever
         from langchain_ollama import ChatOllama
         """Retriever you can plug into a RAG chain."""
         return MultiQueryRetriever.from_llm(retriever=BM25Retriever.from_documents(chunks),llm=ChatOllama(model="llama3.2",temperature=0,disable_streaming=False))
    
    def get_EnsembleRetriever(vector_retriever,BM25_retriever):
         from langchain_classic.retrievers import EnsembleRetriever
         """Retriever you can plug into a RAG chain."""
         return EnsembleRetriever(retrievers=[vector_retriever,BM25_retriever],weights=[0.5,0.5])
   
   
    def ask_ollama( prom ,result = "" , model: str ="llama3.2",is_retiver : bool = True , is_stream : bool=True) -> dict:
        from langchain_ollama import ChatOllama
        if is_retiver==True :
            context = "\n\n".join(d.page_content for d in result)
        else:
           pass
        # 2. Build the prompt
        prompt = prom
    
        # 3. Call the local Ollama model
        llm = ChatOllama(model="llama3.2",temperature=0,disable_streaming=False)
        #response = llm.invoke(prompt)
        if is_stream == True:
            count=0
            for response in llm.stream(prompt) :
                count+=1
                if count==10:
                   print("\n")
                   count=0
                print(response.content,end="",flush=True)
        else:
           response=llm.invoke(prompt)        
        return {
            "answer": response.content
        }
    def get_vector_store(chunks) :
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        import chromadb
        """Open (or create) the persistent vector store."""
        return Chroma.from_documents(chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))



    def reload():
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic('load_ext', 'autoreload')
            ipython.run_line_magic('autoreload', '2')

   # prompt = (
   #             "Answer the question using only the context below. "
    #            "If the answer is not in the context, say you don't know.\n\n"
     #           f"Context:\n{context}\n\n"
      #          f"Question: {question}\n\nAnswer:"
       #     )