import os
import sys
from dotenv import load_dotenv

from src.logger.logger import logging
from src.exception.exception import RAGException
from src.config.config import Config
from src.state.state import RAGState, Router, Decision
from src.vectorstore.vectorstore import RAGVectorStore
from src.utils.summarization import create_master_summary
from langchain_core.documents import Document
from typing import List
import uuid

from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import WikipediaAPIWrapper, tavily_search
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool, tool
from langchain.agents import create_agent

load_dotenv()

class RAGNodes:
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.agent = None
        
        
    def _build_tools(self):
        
        os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
        @tool
        def tavily_search(query: str):
            """Web search for up-to-date information."""
            tavily = TavilySearch()
            
            return tavily.run({"query": query})

        @tool
        def wikipedia_search(query: str):
            """Search Wikipedia for general knowledge."""
            api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=3, lang="en")
            wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
            
            return wiki.run({"query": query})
        
        return [tavily_search, wikipedia_search]
    
    def _build_agent(self):
        
        tools = self._build_tools()
        
        agent_prompt =  """
                        You are an expert external knowledge assistant. Your task is to answer the user's question
                        using only external sources, such as Wikipedia, web tools, or APIs. 

                        Guidelines:
                        1. Use your external tools to retrieve up-to-date factual information.
                        2. If a question can be answered without tools, do so based on your general knowledge.
                        3. Do not use any internal company documents or knowledge base content.
                        4. Provide clear and concise answers.
                        5. If the user question refers to previous conversation, use the chat history to understand context.

                        Only answer the user question provided.
                        """

        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=agent_prompt
        )
        
        self.agent = agent
        
        
        
    def invoke_agent(self, state:RAGState):
        
        if not self.agent:
            self._build_agent()
        
        
            
        result = self.agent.invoke(
            {
                "chat_history": state["chat_history"],
                "input": state["rewritten_query"] + "\nPrevious answer: " + state.get("answer", "")
                
            }
            )
        
        if isinstance(result, dict):
        # If it's a dict, the answer is usually in 'output'
            answer = result.get('output', '') or result.get('result', '')
        elif hasattr(result, 'content'):
            # If it's a message object
            answer = result.content
        elif hasattr(result, 'output'):
            # If it's an AgentFinish object
            answer = result.output
        else:
            # Fallback: convert to string
            answer = str(result)
        
        return {"answer":answer}
    
    
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
            
            master_summary = create_master_summary("./ingested_documents/ingested_docs.json")
            
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=
                                  """
                                    You are an expert query router for a hybrid RAG + Agent system.

                                    Your task is to classify the user's query into EXACTLY one of two categories:

                                    ----------------------------------------------------------------------

                                    ROUTE: "rag"
                                    Choose this when:
                                    - The query concerns information that would logically exist in the internal knowledge base.
                                    - The knowledge base contains the following types of content:\n

                                    {master_summary}

                                    - Also choose "rag" for conversational clarifications that depend on previous context.

                                    ROUTE: "agent"
                                    Choose this when:
                                    - The query requires external or open-world knowledge.
                                    - The question is about general facts, real-world data, current events, public figures,
                                    Wikipedia topics, or anything unlikely to exist in the internal vector store.
                                    - The query explicitly requests web search, external info, or up-to-date knowledge.

                                    IMPORTANT:
                                    - Base your classification on the content described above in the knowledge base.
                                    - Return ONLY one of the following two strings: rag, agent
                                    """
                                  
                                  
                                  ),
                    HumanMessage(content="Query: {rewritten_query}")
                    
                ]
                
            )

            formatted_prompt = prompt.format_messages(
                master_summary= master_summary,
                rewritten_query= state["rewritten_query"]
                )

            decision = router.invoke(formatted_prompt)
            
            return {"next_step": decision}
        except Exception as e:
            raise RAGException(e, sys)
        
    def retrieve_documents(self, state:RAGState) -> RAGState:
        try:
            logging.info("Retrieving relevant documents")
            query = state.get("rewritten_query", state["question"])
            docs = self.retriever.invoke(query)
            doc_ids = [doc.metadata.get("_id", None) for doc in docs]
            
            last_ids = state.get("retrieved_doc_ids", [])
            
            all_ids = state.get('all_retrieval_doc_ids', [])
            all_ids = list(set(all_ids + doc_ids))
                        
            
            logging.info("Documents have been retrieved successfully.")        
            return {
                "retrieved_docs":docs,
                "retrieved_doc_ids": doc_ids,
                "last_retrieved_doc_ids": last_ids,
                "all_retrieval_doc_ids": all_ids
                }
        
        except Exception as e:
            raise RAGException(e, sys)
                

    
    
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
        try:
            
            prompt = ChatPromptTemplate(
                [
                    SystemMessage(content=
                                  """
                                    You are an expert search query refiner for a Retrieval-Augmented Generation (RAG) system. 
                                    The initial attempt to answer the user's question failed because the retrieved documents were insufficient, irrelevant, or non-existent.

                                    Your task is to **reflect** on the original question and generate a single, highly-optimized, semantically distinct search query for a second retrieval attempt.

                                    --- INSTRUCTIONS ---
                                    1.  **Analyze the Failure:** Assume the vocabulary or phrasing of the original question did not match the vocabulary in the documents.
                                    2.  **Refine the Query:** Generate a new query that is more specific, uses synonyms, includes technical terms, or focuses on a different but related aspect of the original question.
                                    3.  **Output Format:** Provide ONLY the single, revised query string contained within the 'rewritten_query' field of the Pydantic schema.

                                    Original Question: {question}
                                  """
                                  ),
                    HumanMessage(content="{question}") 
                    
                ]
            )
            
            example_query = "What was the specific financial outcome detailed in the Q3 2024 quarterly earnings report for the 'Digital Transformation' division, and how did it affect the stock dividend paid to GlobalFinance Corp shareholders?"
            
            formatted_prompt = prompt.format_messages(question=state["question"])
            
            expanded_query_string = self.llm.invoke(formatted_prompt).content
            
            expanded_queries = state.get("expanded_query", [])
            if not expanded_queries and state.get("rewritten_query"):
                expanded_queries.append(state["rewritten_query"])
                
            expansion_counter = state.get("expansion_counter", 0)
                
            expanded_queries.append(expanded_query_string)
            
            
            return {
                "rewritten_query": expanded_query_string,
                "expanded_query": expanded_queries,
                "previous_queries": expanded_queries.copy(),
                "expansion_counter": expansion_counter + 1
                }
        
        except Exception as e:
            raise RAGException(e, sys)
                 
    def conversational_query(self, state:RAGState):
        try:
            return {"retrieved_docs": []}
        except Exception as e:
            raise RAGException(e, sys)
        

    def rate_answer(self, state:RAGState):
        
        try:        
            evaluator = self.llm.with_structured_output(Decision)
            
            context = "\n---\n".join([doc.page_content for doc in state['retrieved_docs']])
            answer = state['answer']
            question = state['question']
            
            given_rating = evaluator.invoke(
                [
                    SystemMessage(content=
        """
        You are an objective Answer Evaluator. Your sole task is to determine the quality and grounding of the provided 'FINAL ANSWER' by comparing it strictly against the 'RETRIEVED CONTEXT' and the original 'QUESTION'.

        --- CRITERIA FOR EVALUATION ---

        1.  **Relevance/Completeness:** Does the 'FINAL ANSWER' directly and fully address the user's 'QUESTION'?
        2.  **Grounding (Hallucination Check):** Is every factual claim in the 'FINAL ANSWER' directly supported by the text in the 'RETRIEVED CONTEXT'?
        3.  **Context Sufficiency:** Was the 'RETRIEVED CONTEXT' sufficient to generate a high-quality answer?

        --- DECISION LOGIC ---

        You MUST output your decision based on the following logic.

        1.  If the 'FINAL ANSWER' is **perfectly grounded** by the context AND fully answers the question, output: **'PASS'**.
        2.  If the 'FINAL ANSWER' contains **unsupported facts (hallucinations)**, fails to answer the question, OR the **RETRIEVED CONTEXT is clearly insufficient** to answer the question, output: **'FAIL'**.
        """
                                  ),
                    HumanMessage(content=f"Question: {question},\n\n Final answer: {answer}\n\n Retrieved context: {context}")
                ]
            )
            
            return {"rating": given_rating}
        except Exception as e:
            raise RAGException(e, sys)