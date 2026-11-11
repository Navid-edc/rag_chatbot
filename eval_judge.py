# Automated RAG Evaluation using the LLM-as-a-Judge pattern
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Initialize the Judge LLM with temperature=0 for strict, deterministic grading
judge_llm = ChatOllama(model="llama3.2", temperature=0.0, format="json")

# Define the grading rubric prompt
judge_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an impartial AI Judge evaluating a RAG system's answer.
Compare the [GENERATED ANSWER] against the [GROUND TRUTH CONTEXT].
Evaluate based on two criteria:
1. Hallucination (Faithfulness): Does the answer contain any fabricated facts not present in the context? (true/false)
2. Score: Rate the accuracy from 1 (completely wrong/hallucinated) to 5 (accurate and faithful).

Return ONLY a valid JSON object with keys: "score" (int), "hallucination" (bool), and "reason" (string)."""),
    ("human", """[QUESTION]: {question}
[GROUND TRUTH CONTEXT]: {context}
[GENERATED ANSWER]: {answer}""")
])

def evaluate_rag_response(question: str, context: str, answer: str) -> dict:
    # Invoke the chain to evaluate the answer
    chain = judge_prompt | judge_llm
    response = chain.invoke({
        "question": question,
        "context": context,
        "answer": answer
    })
    return json.loads(response.content)

if __name__ == "__main__":
    print("\033[96m" + "="*70)
    print(" ⚖️  RUNNING LLM-AS-A-JUDGE EVALUATION PIPELINE ⚖️")
    print("="*70 + "\033[0m\n")

    context_doc = "Full-time employees at DBA receive 26 days of paid annual leave per year. Max 9 days carry over."
    question = "How many days of annual leave do employees get?"

    # Test Case 1: Faithful Answer (No Hallucination)
    good_answer = "Employees receive 26 days of paid annual leave, and up to 9 days can be carried over."
    res1 = evaluate_rag_response(question, context_doc, good_answer)
    print(f"🟢 Test 1 (Good Answer):\n   Answer: {good_answer}\n   Judge Verdict: {res1}\n")

    # Test Case 2: Hallucinated Answer (Fabricated numbers!)
    bad_answer = "Employees receive 45 days of leave and get a $500 vacation bonus."
    res2 = evaluate_rag_response(question, context_doc, bad_answer)
    print(f"🔴 Test 2 (Hallucinated Answer):\n   Answer: {bad_answer}\n   Judge Verdict: {res2}")
    print("="*70)
