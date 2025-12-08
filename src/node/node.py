import os
import sys

from src.logger.logger import logging
from src.exception.exception import RAGException
from src.config.config import Config
from src.state.state import RAGState, Router
from src.vectorstore.vectorstore import RAGVectorStore
from langchain_core.documents import Document
from typing import List
import textwrap

from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser


class RAGNodes:
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
           
        
    def rewrite_query(self, state:RAGState):
        
        try:
            history = state.get("chat_history", [])
            question = state['question']
            
            prompt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder(variable_name='chat_history'),
                    ("user", "{question}"),
                    ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation.")  
                ]
            )
            rewriter_chain = prompt | self.llm | StrOutputParser()
            
            rewritten_query = rewriter_chain.invoke(
                {
                    "chat_history": history,
                    "question": question
                }
            )
            
            return {"rewritten_query": rewritten_query}
        except Exception as e:
            raise RAGException(e, sys)
        
        
    def route_selector(self, state:RAGState):
        try:
            router = self.llm.with_structured_output(Router)
            
            decision = router.invoke(
                [
                    (SystemMessage(content=
                                   """
                                    You are an expert query router. Your goal is to classify the user's query into one of two categories based on the nature of the information requested.

                                    Classification Rules:
                                    1. RAG: The question is specific to company documents, policies, procedures, or historical data contained in the internal knowledge base.
                                    2. CONVERSATIONAL: **The question is a greeting, simple comment, or a General Knowledge/Basic Math question that the language model can answer instantly without needing a search tool or internal documents (e.g., "Hi," "What is 5+5?").**

                                    Return ONLY one of the following two strings: rag, conversational
                                    """
                                   
                                   )),
                    HumanMessage(content=state["rewritten_query"])
                ]
            )
            
            return {"next_step": decision}
        except Exception as e:
            raise RAGException(e, sys)
        
    def retrieve_documents(self, state:RAGState) -> RAGState:
        try:
            logging.info("Retrieving relevant documents")
            query = state.get("rewritten_query", state["question"])
            docs = self.retriever.invoke(query)
            
            logging.info("Documents have been retrieved successfully.")        
            return {"retrieved_docs":docs}
        
        except Exception as e:
            raise RAGException(e, sys)
                
    def rerank_documents(self, state:RAGState):
        pass
    
    
    def generate_answer(self, state:RAGState) -> RAGState:
        
        try:
            logging.info("Answer generation has started")
            context = "\n\n".join([doc.page_content for doc in state['retrieved_docs']])
            chat_history = state.get('chat_history', [])
            question = state['question']
            
            prompt =  ChatPromptTemplate.from_messages(
                [

                    SystemMessage(content= 
                        """
                        You are a RAG assistant that MUST be aware of the conversation history.

                        ***INSTRUCTIONS FOR ANSWERING THE LATEST USER QUESTION:***

                        1.  **Context Resolution:** FIRST, analyze the current user question alongside the "chat_history." If the question contains a pronoun or vague reference (e.g., "it," "that," "the previous result"), you MUST resolve it using the immediate past conversation.
                        2.  **RAG Context Use:** ONLY use the "RAG CONTEXT" for factual grounding and answering questions about specific documents.
                        3.  **General Knowledge/Reasoning:** If the resolved question (e.g., "Divide 10 by 2") requires simple math or general knowledge, use your internal reasoning.
                        4.  **CLARIFICATION PROHIBITED:** If the "chat_history" allows you to resolve the pronoun or reference, you MUST NOT ask the user to clarify. Proceed directly to the answer.

                        You are being tested on your ability to maintain context.
                        """
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("system", "RAG CONTEXT: \n{context}"),
                    ("human", "{question}")
                ]
            )
            
            final_messages = prompt.format_messages(
            chat_history=chat_history,
            context=context,
            question=question
            )
        
            
            response_message = self.llm.invoke(final_messages)
            
            logging.info("Answer has been generated successfully.")
        
            return {"answer": response_message.content}
        
        except Exception as e:
            raise RAGException(e, sys)
        
    def expand_query(self, state:RAGState):
        pass
        """
        try:
            return {"rewritten_query": state["rewritten_query"] + " (EXPANDED)"}
        
        except Exception as e:
            raise RAGException(e, sys)
            
        """        
    def conversational_query(self, state:RAGState):
        try:
            return {"retrieved_docs": []}
        except Exception as e:
            raise RAGException(e, sys)
        
    def web_search(self, state:RAGState):
        pass
        """
        try:
            return {"tool_search_results": "Web search placeholder content."}
        except Exception as e:
            raise RAGException(e, sys)
        """