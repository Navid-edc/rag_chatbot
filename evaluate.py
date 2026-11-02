from rag_chain import EnterpriseRAGBot

# Manual evaluation dataset with reference keywords expected in correct answers
EVAL_DATASET = [
    {
        "question": "How many days of paid annual leave do full-time employees get?",
        "expected_keywords": ["26", "calendar year"]
    },
    {
        "question": "Which days of the week can engineers work remotely after probation?",
        "expected_keywords": ["Mondays", "Wednesdays", "2 days"]
    },
    {
        "question": "Does the company pay for gym memberships?",
        "expected_keywords": ["cannot find this information"]
    }
]

def run_evaluation():
    bot = EnterpriseRAGBot()
    passed = 0
    
    print("=== Running Automated RAG Evaluation Suite ===\n")
    for idx, item in enumerate(EVAL_DATASET, 1):
        res = bot.ask(item["question"])
        answer_lower = res["answer"].lower()
        
        # Verify if all expected reference keywords appear in the generated response
        is_correct = all(kw.lower() in answer_lower for kw in item["expected_keywords"])
        status = "PASS" if is_correct else "FAIL"
        if is_correct:
            passed += 1
            
        print(f"Test Case {idx}: [{status}]")
        print(f"  Q: {item['question']}")
        print(f"  A: {res['answer']}\n")
        
    accuracy = (passed / len(EVAL_DATASET)) * 100
    print(f"=== Final Evaluation Accuracy: {passed}/{len(EVAL_DATASET)} ({accuracy:.1f}%) ===")

if __name__ == "__main__":
    run_evaluation()
