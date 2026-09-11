"""
Interactive CLI for @AmazonHelp AI Customer Support Agent.
Run live queries to observe Intent Classification, Grounded Reply Generation, 
and Deterministic Safety Triage with explicit Stated Reasons.
"""
import sys
import json
from src.agent import AmazonSupportAgent

def main():
    agent = AmazonSupportAgent()
    print("\n" + "="*80)
    print("  @AmazonHelp AI SUPPORT AGENT - INTERACTIVE TEST CONSOLE")
    print("  Objective: Maximize Safe Resolution, Not Automation Rate")
    print("="*80)
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_query(agent, query)
        return

    sample_queries = [
        "My package was supposed to arrive today, where is it?",
        "Package says delivered on the app, but there is nothing on my porch! I think it was stolen.",
        "I was charged $14.99 for Amazon Prime on my credit card but I never signed up for it.",
        "How do I return a pair of shoes that are too small?",
        "Your customer service is utterly useless, I am calling my lawyer and contacting the consumer court!",
        "Shoutout to agent Sherry who helped me recover my package yesterday!"
    ]

    print("\nSelect a sample query to test or type your own custom message:")
    for idx, sq in enumerate(sample_queries, 1):
        print(f"  [{idx}] {sq}")
    print("  [0] Type custom query")
    print("  [q] Quit\n")

    while True:
        try:
            choice = input("Enter selection [1-6, 0, or q]: ").strip()
            if choice.lower() in ["q", "exit", "quit"]:
                print("Exiting console.")
                break
            if choice == "0":
                custom = input("Enter customer message: ").strip()
                if custom:
                    run_query(agent, custom)
            elif choice.isdigit() and 1 <= int(choice) <= len(sample_queries):
                run_query(agent, sample_queries[int(choice) - 1])
            else:
                print("Invalid choice, please try again.")
        except (KeyboardInterrupt, EOFError):
            break

def run_query(agent: AmazonSupportAgent, query: str):
    print("\n" + "-"*80)
    print(f"CUSTOMER MESSAGE: \"{query}\"")
    print("-"*80)
    
    res = agent.process(query)
    
    print(f"INTENT PREDICTED    : {res['intent']} (Confidence: {res['intent_confidence']:.2f})")
    print(f"TRIAGE DECISION     : [{res['decision']}]")
    print(f"ESCALATION CATEGORY : {res['escalation_category']}")
    print(f"STATED REASON       : {res['reason']}")
    print("\nDRAFT REPLY:")
    print(f"  \"{res['reply']}\"")
    
    print("\nHISTORICAL EVIDENCE GROUNDING:")
    if res["evidence"]:
        for i, ev in enumerate(res["evidence"], 1):
            print(f"  [{i}] Conv ID: {ev['conversation_id']} | Cosine Similarity: {ev['similarity']:.4f}")
            print(f"      Historical Reply Excerpt: \"{ev['support_reply'][:100]}...\"")
    else:
        print("  No historical match found.")
    print("-"*80 + "\n")

if __name__ == "__main__":
    main()
