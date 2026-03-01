from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class RetrievalChain:
    """Wrapper class to maintain compatibility with RetrievalQA interface"""
    def __init__(self, llm, prompt, retriever):
        self.llm = llm
        self.prompt = prompt
        self.retriever = retriever
        self.chain = (
            {"context": self._format_retriever(), "question": lambda x: x.get("question") or x.get("query")}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _format_retriever(self):
        """Format retrieved documents as context string"""
        def format_docs(x):
            query_input = x.get("question") or x.get("query")
            docs = self.retriever.get_relevant_documents(query_input)
            return "\n\n".join(doc.page_content for doc in docs)
        return format_docs
    
    def __call__(self, input_dict):
        """Call the chain and return result with source documents"""
        query_input = input_dict.get("question") or input_dict.get("query")
        
        # Get retrieved documents for source tracking
        source_docs = self.retriever.get_relevant_documents(query_input)
        
        # Run the chain
        response = self.chain.invoke(input_dict)
        
        return {
            "result": response,
            "source_documents": source_docs
        }

def get_llm_chain(retriever):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile"
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are **MediBot**, an AI-powered assistant trained to help users understand medical documents and health-related questions.

Your job is to provide clear, accurate, and helpful responses based **only on the provided context**.

---

🔍 **Context**:
{context}

🙋‍♂️ **User Question**:
{question}


---

💬 **Answer**:
- Respond in a calm, factual, and respectful tone.
- Use simple explanations when needed.
- If the context does not contain the answer, say: "I'm sorry, but I couldn't find relevant information in the provided documents."
- Do NOT make up facts.
- Do NOT give medical advice or diagnoses.
"""
    )

    return RetrievalChain(llm, prompt, retriever)
