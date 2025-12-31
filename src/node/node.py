import os
import sys
from dotenv import load_dotenv

from src.logger.logger import logging
from src.exception.exception import RAGException
from src.state.state import RAGState, ExpandedQueries, Router, GraderDecision, HallucinationGrade

from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain.agents import create_agent



load_dotenv()

class RAGNodes:
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.external_agent = None
        self.rag_agent = None
        self.tools = self._build_tools()
        
        
    def reset_turn_state(self, state: RAGState) -> RAGState:
        """
        Resets state for every conversation turn.
        """
        logging.info("Resetting conversation state.")
        return {
            "answer": "",
            "answer_source": None,
            "frozen_rag_facts": None,
            "retrieved_docs": [],
            "retrieved_doc_ids": [],
            "relevant_ids": None,
            "context": None,
            "final_grade": None,
            "expanded_query_list": None,
            "expansion_counter": 0,
        }
            
        
    def _build_tools(self):
        """
        Builds Wikipedia and Tavily search for the agents to use later.
        """
        
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
    
    def _build_external_agent(self):
        """
        AI agent using external tools (Wikipedia + Tavily) for answering queries, it does NOT have access to the internal vector database.
        """
        
        
        try:
            
            logging.info("Building external agent")
            
            
            external_agent_prompt =  """
                            You are an External Knowledge Question-Answering Agent.

                            Your task is to answer the user’s question using ONLY external, public sources.
                            You have access to search tools (e.g., web search, Wikipedia).

                            STRICT RULES:
                            1. Use tools to find factual information before answering.
                            2. Do NOT use internal knowledge bases, private company data, or assumptions.
                            3. If the tools return no relevant information, explicitly say:
                            “I could not find reliable external information to answer this question.”
                            4. Do NOT hallucinate facts, names, metrics, or events.
                            5. Keep the answer concise, factual, and directly focused on the question.
                            6. If multiple sources disagree, state that uncertainty clearly.

                            You are the final authority for this response.
                            """

            external_agent = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=external_agent_prompt
            )
            
            self.external_agent = external_agent
            
            logging.info("External agent has been built successfully.")
        
        except Exception as e:
            raise RAGException(e, sys)
        
    def _build_rag_agent(self):
        """
        AI agent using external tools (Wikipedia + Tavily) for augmenting partially answered queries, it DOES have access to the internal vector database.
        """        
        try:
            
            logging.info("Building RAG agent")          
            
            rag_agent_prompt =  """
                            You are a RAG Augmentation Agent.

                            You are given:
                            - A trusted internal RAG answer (may be partial or incomplete)
                            - A user question
                            - Access to external search tools

                            YOUR ROLE:
                            Fill in missing information using external sources WITHOUT overwriting or contradicting the internal RAG answer.

                            STRICT RULES:
                            1. Treat the provided RAG answer as FACTUALLY CORRECT.
                            2. Use external tools ONLY to find information that is missing or incomplete.
                            3. NEVER change, reinterpret, or remove information from the RAG answer.
                            4. If external search succeeds:
                            - Clearly separate new information from existing RAG content.
                            5. If external search fails:
                            - Keep the RAG answer intact.
                            - Explicitly state that no additional external information was found.
                            6. Do NOT hallucinate or speculate.
                            7. Do NOT repeat the entire RAG answer verbatim unless necessary.

                            Your final response must be a clear, combined answer suitable for the end user.
                            You are NOT allowed to introduce new internal company metrics, KPIs, or performance figures that are not explicitly present in the provided RAG answer.

                            """

            rag_agent = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=rag_agent_prompt
            )
            
            self.rag_agent = rag_agent
            
            logging.info("RAG agent has been built successfully.")
        
        except Exception as e:
            raise RAGException(e, sys)    
        
        
        
    def invoke_external_agent(self, state:RAGState):
        """
        Invocation method of the external agent.
        """
        
        try:
            logging.info("Invoking external agent")
            
            #Only initializing agent if necessary
            if not self.external_agent:
                    self._build_external_agent()
                    
            #Getting original question
            og_question = state["question"]

            #Getting rewritten query
            rewritten_query = state["rewritten_query"]

            #Getting chat history
            messages = list(state.get("chat_history", []))
            
            external_agent_prompt = f"""
            You are an external information specialist. Your primary task is to use web search to find missing data.
    
            The question is: '{rewritten_query}'
            """

            #Adding question as last HumanMessage to chat history
            messages.append(
                HumanMessage(
                    content=external_agent_prompt.strip()
                )
            )

            #Invoking agent with messages only
            result = self.external_agent.invoke({
                "messages": messages
            })

            #Extracting final AI answer
            final_answer = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    final_answer = msg.content
                    break
                
            if final_answer and "not find any information" not in final_answer.lower():
                pass
            else:
                final_answer = f"I was unable to find the answer to your question ('{og_question}') after searching external knowledge sources."
            
            logging.info("Agent answer has been generated successfully.")

            return {
                "answer": final_answer.strip(),
                "answer_source": "external_agent",
                "chat_history": result["messages"]
            }
                    
        except Exception as e:
            logging.info("Agent failed to generate an answer.")
            raise RAGException(e, sys)
        
    def invoke_rag_agent(self, state:RAGState):
        """
        Invocation method of the RAG agent which tries to provide a complete answer for partially answered queries by using tools (Wikipedia + Tavily).
        """
        
        try:
            logging.info("Invoking agent")
            
            #Only initializing agent if necessary
            if not self.rag_agent:
                    self._build_rag_agent()
                    
            # Getting original question
            og_question = state["question"]

            #Getting rewritten query
            rewritten_query = state["rewritten_query"]

            # Getting chat history
            messages = list(state.get("chat_history", []))
            
            rag_answer = None
            
            frozen = state.get("frozen_rag_facts", "").strip()
            if frozen:
                rag_answer = frozen
            elif state.get("answer_source") == "rag":
                candidate = state.get("answer", "").strip()
                if candidate:
                    rag_answer = candidate
                    
            if rag_answer:
                rag_status = "A partial or complete internal RAG answer is available."
                rag_section = f"""
                INTERNAL RAG ANSWER (TRUSTED):
                {rag_answer}
                """
            else:
                rag_status = "No relevant internal documents were found for this question."
                rag_section = ""
                
            
            rag_agent_prompt = f"""
            USER QUESTION:
            {rewritten_query}

            INTERNAL RAG STATUS:
            {rag_status}

            {rag_section}

            TASK:
            Use external public sources to help answer the question.

            RULES:
            - If an INTERNAL RAG ANSWER is provided above, treat it as fact and ONLY add missing information.
            - If no INTERNAL RAG ANSWER is provided, answer the question using external sources only.
            - NEVER assume internal information exists unless explicitly provided.
            - NEVER use previous conversation answers as factual input.
            - If external sources do not provide reliable information, say so explicitly.

            Return a single, clear, user-facing answer.
            """

            #Adding question as last HumanMessage to chat history
            messages.append(
                HumanMessage(
                    content=rag_agent_prompt.strip()
                )
            )

            #Invoking agent with messages only
            result = self.rag_agent.invoke({
                "messages": messages
            })

            #Extracting final AI answer
            final_answer = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    final_answer = msg.content
                    break
                
            if final_answer and "not find any information" not in final_answer.lower():
                pass
            else:
                final_answer = f"I was unable to find the answer to your question ('{og_question}') after searching external knowledge sources."
            
            logging.info("Agent answer has been generated successfully.")

            return {
                "answer": final_answer.strip(),
                "answer_source": "rag_agent",
                "chat_history": result["messages"]
            }
                    
        except Exception as e:
            logging.info("Agent failed to generate an answer.")
            raise RAGException(e, sys)
    
    
    def rewrite_query(self, state:RAGState):
        """
        Docstring for rewrite_query
        
        :param self: Description
        :param state: Description
        :type state: RAGState
        """
        
        try:
            logging.info("Rewriting query has been started.")
              
            rewriter_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", """
                                You are an expert question rewriter. Your task is to transform a user's question
                                into a highly effective, standalone search query, suitable for retrieval from a vector store or external search engine.

                                **INSTRUCTIONS:**
                                1.  **Contextualize:** If the current query refers to entities or concepts from the 'chat_history' (e.g., "it", "them", "that philosophy", "he"), replace the pronoun with the specific entity name from history.
                                2.  **Neutral resolution:** Resolve references NEUTRALLY. If the user asks "Who created this?", rewrite to "Who created [Entity Name]". Do NOT assume titles like "Chief Architect" or specific roles unless explicitly mentioned.
                                3.  **Specific:** Ensure the rewritten query is maximally specific and detailed.
                                4.  **Default/Simple Queries:** If the user's query is already clear, self-contained, simple, or nonsensical, **return the word 'suitable'.** Do NOT inject any additional language or commentary.
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
        """
        Routes the query to the most relevant node for faster response time.
        
        Simple queries -> 'chat' -> Conversational node
        Internal knowledge related queries -> 'rag' -> Retriever node
        Queries not related to internal database but require tool usage -> 'agent' -> External agent node
        """
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
                                    * *Anti-Example: "When was this created?" (Route to RAG/Agent)*

                                **Your output must be ONLY the single word: rag, agent, or chat.**
                                """
                                ),
                    ("human", "Query: {rewritten_query}")
                ]
            )
            
            formatted_router_prompt = router_prompt.format_messages(rewritten_query=state["rewritten_query"])
            
            route = router.invoke(formatted_router_prompt).next_step
            
            return {
                "route": route
                }
        
        except Exception as e:
            raise RAGException(e, sys)
        
    def conversational_answer(self, state:RAGState) -> RAGState:
        """
        LLM which directly answers simple, non-factual questions, without accessing the vector database or external tools.
        """
        try:
            conv_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", """
                                You are a highly efficient and friendly general-purpose conversational assistant. 
                                Your sole purpose is to answer simple, non-factual questions such as greetings, small talk, opinions, or simple arithmetic.
                                
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
            
            return {"answer": response, "answer_source": "chat"}
        except Exception as e:
            raise RAGException(e, sys)

    def retrieve_documents(self, state:RAGState) -> RAGState:
        """
        Retrieves documents from the vector store.
        """
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
        
    def grade_documents(self, state:RAGState) -> RAGState:
        """
        Grades the retieved chunks and outputs a list of chunks ids which are relevant to the query.
        """
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
        """
        Based on the retrieved relevant chunk contexts, it formulates the answer to the query.
        """
        
        try:
            logging.info("Answer generation has started")
            
            relevant_ids = state.get("relevant_ids", [])
            
            filtered_context_chunks = [doc.page_content for doc in state.get("retrieved_docs", []) if doc.metadata.get("_id") in relevant_ids]
            
            clear_context_string = "\n---\n".join(filtered_context_chunks)
            
            prompt =  ChatPromptTemplate.from_messages(
                [

                    ("system", 
                        """
                        You are a factual, precise RAG assistant.

                        Your task is to answer the user's question using ONLY the information provided in the RAG CONTEXT.

                        ***INSTRUCTIONS FOR ANSWERING (STRICT):***

                        1. FACTUAL GROUNDING
                        - Use ONLY facts explicitly stated in the RAG CONTEXT.
                        - Do NOT infer, guess, extrapolate, or use external knowledge.

                        2. MULTI-PART QUESTIONS
                        - Carefully identify all parts of the user's question.
                        - Answer every part that is supported by the RAG CONTEXT.

                        3. MISSING INFORMATION DISCLOSURE
                        - If the question asks for information that is NOT present in the RAG CONTEXT:
                        - Clearly state that this information is not available.
                        - Phrase this naturally for the user.
                        - Do NOT mention “RAG”, “context”, “documents”, or retrieval mechanics.
                        - Example phrasing:
                            • “The available information does not specify …”
                            • “This aspect is not documented in the provided information.”

                        4. WHEN NOT TO DISCLOSE MISSING INFORMATION
                        - If all parts of the question are fully answered, do NOT mention missing information.
                        - Do NOT add disclaimers unless strictly necessary to avoid misleading the user.

                        5. ANSWER QUALITY
                        - Be concise, clear, and well-structured.
                        - Use bullet points or sections when multiple metrics or facts are involved.
                        - Do NOT restate the entire context.
                        - Do NOT explain your reasoning process.

                        6. OUTPUT RULES
                        - Produce a single, user-facing answer only.
                        - Do NOT include system notes, grading labels, or internal commentary.
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
                "answer_source": "rag",
                "context": clear_context_string
                }
        
        except Exception as e:
            raise RAGException(e, sys)
        
        
    def grade_hallucination(self, state: RAGState) -> RAGState:
        try:

            logging.info("Starting hallucination grading (fact-check)")
            
            expansion_counter = state.get("expansion_counter", 0)
            
            hallucination_grader = self.llm.with_structured_output(HallucinationGrade)
            
            grader_prompt = PromptTemplate.from_template(
                            """
                            You are a strict classifier. Your task is to compare the GENERATED ANSWER against the RAG CONTEXT and the USER QUESTION to assign ONE label.

                            **USER QUESTION:**
                            ---
                            {rewritten_query}
                            ---

                            **RAG CONTEXT:**
                            ---
                            {context}
                            ---

                            **GENERATED ANSWER:**
                            ---
                            {answer}
                            ---

                            Labels:

                            relevant:
                            - All claims are fully supported by the context.
                            - The answer completely addresses every part of the USER QUESTION.
                            - No subpart of the question is missing from the answer.

                            partially_relevant:
                            - The answer is factually correct based on context, BUT it fails to address the entire USER QUESTION.
                            - If the answer explicitly states that some information is not present in the context, choose this.
                            - If the question asks for multiple metrics and only one is provided, choose this.

                            not_relevant:
                            - Claims are unsupported, hallucinated, or contradicted by the context.

                            IMPORTANT PRIORITY RULES:
                            1. If the answer explicitly notes missing information required to fully answer the QUESTION, you MUST choose "partially_relevant".
                            2. Compare the ANSWER to the QUESTION strictly. If the Question asks for "Revenue and Net Profit" and the Answer only provides "Revenue", the grade MUST be "partially_relevant".

                            STRICT RULES:
                            - DO NOT explain your reasoning
                            - OUTPUT ONLY ONE WORD: relevant, partially_relevant, or not_relevant
                            """
                            )

            formatted_grader_prompt = grader_prompt.format_prompt(
                context=state["context"], 
                answer=state["answer"],
                rewritten_query=state["rewritten_query"] 
            )
            
            final_grade = hallucination_grader.invoke(formatted_grader_prompt).grade
            
            logging.info(f"Hallucination grade: {final_grade}")
            
            update = {
                "final_grade": final_grade,
                "expansion_counter": expansion_counter
            }
            
            if (
                expansion_counter == 0
                and state.get("answer_source") == "rag"
                and final_grade in ["relevant", "partially_relevant"]
                ):
            
                update["frozen_rag_facts"] = state["answer"]
                
            if not state.get("context"):
                update["final_grade"] = "not_relevant"
                
            logging.info(
                f"Grader decision: {final_grade} "
                f"Expansion_counter: {expansion_counter} "
                f"Answer_source: {state.get('answer_source')}"
            )
                
            return update

        except Exception as e:
            raise RAGException(e, sys)
        
    def increment_expansion_counter(self, state: RAGState) -> RAGState:
        """
        Increment expansion counter.
        """
        try:
            return {
                "expansion_counter": state.get("expansion_counter", 0) + 1
            }
            
        except Exception as e:
            raise RAGException(e, sys)
        
    def expand_query(self, state:RAGState) -> RAGState:
        """
        Expands the query to 3 different search queries to improve retrieved context.
        """
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
            
            formatted_expander_template = expander_template.format(rewritten_query=state["rewritten_query"])
            
            new_queries = expander.invoke(formatted_expander_template).queries
            
            new_query_string = " ".join([query for query in new_queries])
            
            logging.info("Original query has been expanded.")
             
            return {
                "rewritten_query": new_query_string,
                "expanded_query_list": new_queries
                }
        
        except Exception as e:
            raise RAGException(e, sys)
        