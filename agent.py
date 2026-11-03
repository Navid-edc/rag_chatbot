import sqlite3
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from retriever import AdvancedRetriever

# Initialize our Stage-1 & Stage-2 RAG Retriever once
rag_retriever = AdvancedRetriever()
DB_PATH = "company_data.db"

# --- TOOL 1: Unstructured Policy Search (RAG + Reranker) ---
@tool
def search_company_policy(query: str) -> str:
    """Search internal DBA company policies regarding annual leave rules, remote work schedules, IT security, and AI standards."""
    chunks = rag_retriever.retrieve_and_rerank(query, initial_k=4, top_n=2)
    return "\n\n".join([f"[Policy Excerpt]: {c['text']}" for c in chunks])

# --- TOOL 2: Structured SQL Database Query ---
@tool
def get_employee_record(employee_name: str) -> str:
    """Query the SQL database to get a specific employee's department, role, used_leave_days, and remote_days_per_week by their name."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Use parameterized SQL query (?) to prevent SQL Injection attacks
    cursor.execute("""
        SELECT emp_id, full_name, department, role, used_leave_days, remote_days_per_week 
        FROM employees 
        WHERE full_name LIKE ?
    """, (f"%{employee_name}%",))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return f"No employee found matching name '{employee_name}' in SQL database."
    
    return (
        f"SQL Record -> ID: {row[0]}, Name: {row[1]}, Department: {row[2]}, "
        f"Role: {row[3]}, Used Leave Days: {row[4]}, Remote Days/Week: {row[5]}"
    )

class EnterpriseAgent:
    def __init__(self):
        # Define available tools for the LLM
        self.tools = [search_company_policy, get_employee_record]
        self.tool_map = {t.name: t for t in self.tools}
        
        # Bind tools to local Llama 3.2 so it knows their schemas and arguments
        self.llm_with_tools = ChatOllama(
            model="llama3.2",
            temperature=0.0
        ).bind_tools(self.tools)
        
        # Conversation memory keyed by session_id to maintain multi-turn context
        self.memory = {}

    def chat(self, session_id: str, user_input: str) -> dict:
        if session_id not in self.memory:
            self.memory[session_id] = [
                SystemMessage(content=(
                    "You are an intelligent DBA Enterprise Agent with access to two tools:\n"
                    "1. 'search_company_policy' for general company rules (like total allowed annual leave).\n"
                    "2. 'get_employee_record' for specific employee stats from the SQL database.\n"
                    "Always call the appropriate tool(s) to gather facts before answering. Never guess numbers."
                ))
            ]
        
        # Append user's new question to session memory
        messages = self.memory[session_id]
        messages.append(HumanMessage(content=user_input))
        
        tools_called = []
        
        # Agent Execution Loop (Allow up to 3 sequential tool-calling steps)
        for step in range(3):
            ai_msg = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)
            
            # If the model decided it has enough info and didn't call any tool, break and return final answer
            if not ai_msg.tool_calls:
                break
                
            # Execute each tool requested by the LLM
            for tool_call in ai_msg.tool_calls:
                t_name = tool_call["name"]
                t_args = tool_call["args"]
                print(f"\n[AGENT ACTION] -> Calling Tool: '{t_name}' with args: {t_args}")
                
                # Run the actual Python function (RAG or SQL)
                tool_output = self.tool_map[t_name].invoke(t_args)
                tools_called.append({"tool": t_name, "args": t_args, "output": tool_output})
                
                # Feed the tool's output back to the LLM as a ToolMessage
                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
        
        return {
            "session_id": session_id,
            "question": user_input,
            "answer": messages[-1].content,
            "tools_used": tools_called
        }

if __name__ == "__main__":
    agent = EnterpriseAgent()
    session = "test_session_1"
    
    # Turn 1: Hybrid question requiring BOTH SQL lookup and Policy RAG lookup!
    q1 = "How many leave days has Navid Abbasi used so far, and based on the company annual leave policy, how many days does he have left?"
    print(f"\n==================== TURN 1 (SQL + RAG Hybrid) ====================")
    print(f"User: {q1}")
    res1 = agent.chat(session, q1)
    print(f"\n[AGENT FINAL ANSWER]:\n{res1['answer']}\n")
    
    # Turn 2: Follow-up question testing Conversation Memory (Notice we say 'she' instead of 'Sara Ahmadi'!)
    q2 = "What department does he work in and what is his role?"
    print(f"\n==================== TURN 2 (Memory Test) ====================")
    print(f"User: {q2}")
    res2 = agent.chat(session, q2)
    print(f"\n[AGENT FINAL ANSWER]:\n{res2['answer']}\n")
