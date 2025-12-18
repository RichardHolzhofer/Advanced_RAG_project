import os
import sys
from dotenv import load_dotenv

from src.logger.logger import logging
from src.exception.exception import RAGException
from src.config.config import Config
from src.state.state import RAGState, CustomAgentState, ExpandedQueries, Router, GraderDecision, HallucinationGrade
from src.vectorstore.vectorstore import RAGVectorStore
from src.utils.summarization import create_master_summary
from langchain_core.documents import Document
from typing import List
import uuid

from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool, tool
from langchain.agents import create_agent
from langgraph.prebuilt import ToolNode



load_dotenv()

class RAGNodes:
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.agent = None
        
        
    def _build_tools(self):
        
        try:
            logging.info("Building Tavily and Wikipedia tools")
            # Building Tavily Tool
            os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
            @tool
            def tavily_search(query: str):
                """Web search for up-to-date information."""
                tavily = TavilySearch()
                result = tavily.run(query)
                
                return result
            
            # Building Wikipedia Tool
            @tool
            def wikipedia_search(query: str):
                """Search Wikipedia for general knowledge."""
                api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=3, lang="en")
                wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
                result = wiki.run(query)
                
                return result
            
            logging.info("Tools have been built successfully.")
            
            return [tavily_search, wikipedia_search]
        
        except Exception as e:
            raise RAGException(e, sys)
    
    def _build_agent(self):
        
        try:
            
            logging.info("Building agent")
        
            tools = self._build_tools()
            
            
            agent_prompt =  """
                            You are an expert external knowledge assistant. 
                            Your task is to answer the user's question using ONLY external sources (Wikipedia, Search).

                            STRICT GUIDELINES:
                            1. Use tools to find factual information. 
                            2. If your search results return NO information about a specific company or entity, 
                            you MUST state: "I cannot find any information regarding [Entity Name] in external records." 
                            DO NOT invent details or assume a company exists based on its name.
                            3. If the user asks a follow-up, use the chat history to understand who they are talking about.
                            4. Keep answers concise and cite your sources if possible.
                            """

            agent = create_agent(
                model=self.llm,
                tools=tools,
                system_prompt=agent_prompt,
                state_schema=CustomAgentState
            )
            
            self.agent = agent
            
            logging.info("Agent has been built successfully.")
        
        except Exception as e:
            raise RAGException(e, sys)
        
        
    def invoke_agent(self, state:RAGState):
        try:
            logging.info("Invoking agent")
            
            # Only initializing agent if necessary
            if not self.agent:
                    self._build_agent()

            # Getting rewritten query
            rewritten_query = state["rewritten_query"]

            # Getting chat history
            messages = list(state.get("chat_history", []))
            

            # Adding question as last HumanMessage to chat history
            messages.append(
                HumanMessage(
                    content=rewritten_query
                )
            )

            # Invoking agent with messages only
            result = self.agent.invoke({
                "messages": messages
            })

            # Extracting final AI answer
            final_answer = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    final_answer = msg.content
                    break
                
            if not final_answer or "not find any information" in final_answer.lower():
                final_answer = f"I'm sorry, I couldn't find any verified information about '{rewritten_query}' in my external sources."
            
            logging.info("Agent answer has been generated successfully.")

            return {
                "answer": final_answer,
                "chat_history": result["messages"]
            }
                    
        except Exception as e:
            logging.info("Agent failed to generate an answer.")
            raise RAGException(e, sys)
    
    
    def rewrite_query(self, state:RAGState):
        
        try:
            logging.info("Rewriting query has been started.")
              
            rewriter_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", """
                                You are an expert question rewriter. Your task is to transform a user's question
                                into a highly effective, standalone search query, suitable for retrieval from a vector store or external search engine.

                                **INSTRUCTIONS:**
                                1.  **Contextualize:** If the current user query is ambiguous or relies on previous conversation history (provided in 'chat_history'), rewrite it to be fully self-contained.
                                2.  **Specific:** Ensure the rewritten query is maximally specific and detailed.
                                3.  **Default/Simple Queries:** If the user's query is already clear, self-contained, simple, or nonsensical, **return the word 'suitable'.** Do NOT inject any additional language or commentary.
                                """
                                    ),
                    MessagesPlaceholder(variable_name='chat_history'),
                    ("human", "Question: {question}")
                ]
            )
            
            formatted_rewriter_prompt = rewriter_prompt.format_messages(chat_history=state.get("chat_history",[]), question=state["question"])
            rewritten_query = self.llm.invoke(formatted_rewriter_prompt).content.strip()
            
            if not rewritten_query or rewritten_query == 'suitable':
                rewritten_query = state["question"]
        
            logging.info("Query has been rewritten successfully.")
            return {"rewritten_query": rewritten_query}
            
        except Exception as e:
            raise RAGException(e, sys)
        
        
    def route_query(self, state:RAGState):
        try:
        
            router = self.llm.with_structured_output(Router)
            router_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system",  """
                                You are a highly specialized RAG routing engine. Your task is to categorize the PROCESSED QUERY into one of three destinations.
                                
                                **CLASSIFICATION RULES (STRICT HIERARCHY):**
                                
                                1.  **rag**: Choose rag if the query asks for **ANY SPECIFIC FACT, DATA, or METRIC** related to internal company entities (e.g., 'NeuralFlow AI', 'DocFlow AI', QBRs, ROI, company policy). **RAG is the specialized default domain.**
                                    * *Example: "What is the Q3 sales figure for NeuralFlow AI?"*
                                
                                2.  **agent**: Choose agent if the query asks for **general, external, or current world knowledge** that is factual but NOT about the internal company (e.g., historical events, current world leaders, common science facts).
                                    * *Example: "Who is the Prime Minister of Canada?"*
                                
                                3.  **chat**: Choose chat **ONLY** if the query is strictly conversational, subjective, or a simple utility function. This includes: greetings, small talk, opinions, or simple arithmetic. **NEVER route a factual query to CHAT.**
                                    * *Example: "How are you today?" or "What is 10 plus 5?"*

                                **Your output must be ONLY the single word: rag, agent, or chat.**
                                """
                                ),
                    ("human", "Query: {rewritten_query}")
                ]
            )
            
            formatted_router_prompt = router_prompt.format_messages(rewritten_query=state["rewritten_query"])
            
            route = router.invoke(formatted_router_prompt).next_step
            
            return {"route": route}
        
        except Exception as e:
            raise RAGException(e, sys)
        
    def conversational_answer(self, state:RAGState):
        try:
            conv_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", """
                                You are a highly efficient and friendly general-purpose conversational assistant. 
                                Your sole purpose is to answer simple, non-factual questions.
                                
                                **Context:** Use the chat history to maintain the flow of conversation.
                                
                                Provide a short, concise answer, do not output anything else.
                                """       
                                ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{rewritten_query}")
                ]
            )
            
            formatted_conv_prompt = conv_prompt.format_messages(chat_history = state.get("chat_history", []), rewritten_query=state["rewritten_query"])
            

            response = self.llm.invoke(formatted_conv_prompt).content
            
            return {"answer": response}
        except Exception as e:
            raise RAGException(e, sys)

    def retrieve_documents(self, state:RAGState) -> RAGState:
        try:
            logging.info("Retrieving relevant documents")
            
            retrieved_doc_ids = []
            
            retrieved_docs = self.retriever.invoke(state["rewritten_query"])
            
            if not retrieved_docs:
                logging.warning(f"No documents found for query: {state['rewritten_query']}")
            
            retrieved_doc_ids = [doc.metadata["_id"] for doc in retrieved_docs]
                
                        
            logging.info("Documents have been retrieved successfully.")        
            return {
                "retrieved_docs":retrieved_docs,
                "retrieved_doc_ids": retrieved_doc_ids
                }
        
        except Exception as e:
            raise RAGException(e, sys)
        
    def grade_documents(self, state:RAGState):
        try:
            logging.info("Grading documents for relevance using LLM.")
            
            grader = self.llm.with_structured_output(GraderDecision)
            
            docs = state.get("retrieved_docs", [])
            
            if not docs:
                logging.info("No documents retrieved. Setting document_state to not_relevant.")
                return {"document_state": "not_relevant", "context": []}
            
            # Combining docs
            doc_contents = "\n\n---\n\n".join([f"Document ID: {doc.metadata.get('_id', f'Unknown_ID_{i+1}')}\n"
                                               f"Content:\n {doc.page_content}"
                                               for i, doc in enumerate(docs)])
            
            grader_prompt = PromptTemplate.from_template(
                                """
                                You are a highly reliable Document Grader for an internal knowledge retrieval system. 
                                Your sole purpose is to analyze the user's question against a set of retrieved documents and determine their relevance.

                                **QUESTION:** {rewritten_query}

                                **DOCUMENTS:**
                                ---
                                {doc_contents}
                                ---

                                **INSTRUCTIONS:**
                                1.  **Analyze Relevance:** Carefully examine the content of each document, using the unique 'Document ID' as the identifier.
                                2.  **Strict Filtering:** Only a document that contains specific, usable facts, figures, or details *directly* related to the question is considered 'relevant'. If the document is too vague, general, or unrelated, it is NOT relevant.
                                3.  **Strict ID Usage:** The values in the list MUST be the exact 'Document ID' values provided in the input. Do not include the word 'Document ID', only the ID itself.
                                
                                **LOGIC RULES:**
                                * If *at least one* document is relevant, list all relevant Document IDs.
                                * If *NO* document is relevant, the **"relevant_ids" list MUST be empty: [].
                                """
                                )
            
            formatted_grader_prompt= grader_prompt.format_prompt(rewritten_query=state["rewritten_query"], doc_contents=doc_contents)
            
            relevant_doc_ids = grader.invoke(formatted_grader_prompt).relevant_ids
            
            if relevant_doc_ids:
                logging.info("Documents are relevant, proceeding to generation")
                return {"relevant_ids": relevant_doc_ids}
            else:
                logging.info("Documents are NOT relevant, Falling back to Agent.")
                return {"relevant_ids": []}
        except Exception as e:
            raise RAGException(e, sys)
                
  
    def generate_answer(self, state:RAGState) -> RAGState:
        
        try:
            logging.info("Answer generation has started")
            
            relevant_ids = state.get("relevant_ids", [])
            
            filtered_context_chunks = [doc.page_content for doc in state.get("retrieved_docs", []) if doc.metadata.get("_id") in relevant_ids]
            
            clear_context_string = "\n---\n".join(filtered_context_chunks)
            
            prompt =  ChatPromptTemplate.from_messages(
                [

                    ("system", 
                        """
                        You are a RAG assistant that must answer the user's question based on the provided RAG CONTEXT.

                        ***INSTRUCTIONS FOR ANSWERING:***
                        1.  **Resolved Question:** The 'Question' provided has already been contextualized (pronouns resolved) by an upstream system. Use this as your primary focus.
                        2.  **RAG Context Use:** ONLY use the "RAG CONTEXT" for factual grounding. If the context does not contain the answer, you must state that the information is unavailable in the internal documents.
                        3.  **General Knowledge/Math:** If the question is simple math (e.g., "5+5") or common knowledge not found in the RAG CONTEXT, you may use your internal knowledge.
                        4.  **Chat History Use:** Use the 'Chat History' ONLY for tone and conversational flow, not for factual information.
                        5.  **Be Concise:** Provide a clear, direct, and concise final answer.
                        """
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("system", "RAG CONTEXT: \n {context}"),
                    ("human", "Question:\n {rewritten_query}")
                ]
            )
            
            final_messages = prompt.format_messages(
            chat_history=state["chat_history"],
            context=clear_context_string,
            rewritten_query=state["rewritten_query"]
            )
        
            
            response_message = self.llm.invoke(final_messages)
            
            logging.info("Answer has been generated successfully.")
        
            return {
                "answer": response_message.content,
                "context": clear_context_string
                }
        
        except Exception as e:
            raise RAGException(e, sys)
        
        
    def grade_hallucination(self, state:RAGState):
        try:
            logging.info("Starting hallucination grading (fact-check)")
            
            expansion_counter = state.get("expansion_counter", 0)
            
            hallucination_grader = self.llm.with_structured_output(HallucinationGrade)
            
            grader_prompt = PromptTemplate.from_template(
                            """
                            You are a fact-checker. Your task is to strictly compare the "GENERATED ANSWER" against the "RAG CONTEXT" and grade its factual grounding.

                            **RAG CONTEXT:**
                            ---
                            {context}
                            ---

                            **GENERATED ANSWER:**
                            ---
                            {answer}
                            ---

                            **GRADING INSTRUCTIONS:**
                            1.  **relevant:** If the ENTIRE answer is directly supported by the RAG CONTEXT.
                            2.  **partially_relevant:** If MOST of the answer is supported by the context, but some minor parts are missing, vague, or unsupported. This indicates insufficient context.
                            3.  **not_relevant:** If the answer contains major claims not found in the RAG CONTEXT, or if the context directly contradicts the answer. This indicates a hallucination or severe inaccuracy.
                        
                            """
                            )
        
            formatted_grader_prompt = grader_prompt.format_prompt(context=state["context"], answer=state["answer"])
            
            final_grade = hallucination_grader.invoke(formatted_grader_prompt).grade
            
            logging.info(f"Hallucination grade: {final_grade}")
            
            return {
                "final_grade": final_grade,
                "expansion_counter": expansion_counter
                }
        except Exception as e:
            raise RAGException(e, sys)
        
    def expand_query(self, state:RAGState):
        try:
            logging.info("Query expansion has been triggered.")
            expander = self.llm.with_structured_output(ExpandedQueries)
            expander_template ="""
            You are a Search Query Optimizer. The previous search for the question: 
            {rewritten_query}
            yielded incomplete results.
            
            Your task is to generate 3 DIFFERENT search queries that tackle the question from different angles to find the missing information.
            
            Guidelines:
            1.  **Specific:** Focus on keywords that might have been missed.
            2.  **Decomposition:** If the question is complex, break it into simpler sub-queries.
            3.  **Synonyms:** Use professional synonyms or related technical terms.
            
            """

            
            example_query = "What was the specific financial outcome detailed in the Q3 2024 quarterly earnings report for the 'Digital Transformation' division, and how did it affect the stock dividend paid to GlobalFinance Corp shareholders?"
            
            formatted_expander_template = expander_template.format(rewritten_query=state["rewritten_query"])
            
            new_queries = expander.invoke(formatted_expander_template).queries
            
            new_query_string = " ".join([query for query in new_queries])
            
            logging.info("Original query has been expanded.")
            
            current_count = state.get("expansion_counter", 0)
            
            
            return {
                "rewritten_query": new_query_string,
                "expanded_query_list": new_queries,
                "expansion_counter": current_count + 1
                }
        
        except Exception as e:
            raise RAGException(e, sys)
        