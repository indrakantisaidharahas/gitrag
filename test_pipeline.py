import sys
import os
from src.pipeline import ChatGraphRAGPipeline

def main():
    print("="*60)
    print("Starting ChatGraph RAG Pipeline Test")
    print("="*60)
    
    pipeline = ChatGraphRAGPipeline()
    pipeline.build_indices()
    
    test_questions = [
        "What caused the PaymentService 504 error and who fixed it?",
        "What frontend framework did Eve decide on for the customer portal?",
        "Who is assigned to the Presidio PII redaction security ticket?"
    ]
    
    for q in test_questions:
        print("\n" + "="*60)
        print(f"QUESTION: {q}")
        print("="*60)
        
        result = pipeline.query(q)
        
        print("\nMATCHED GRAPH NODES:")
        print(result["matched_nodes"])
        
        print("\nRETRIEVED GRAPH TRIPLETS:")
        for t in result["graph_triplets"]:
            print(f"  ({t['subject']}, {t['relation']}, {t['object']})")
            
        print("\nRETRIEVED VECTOR CHUNKS:")
        for c in result["vector_chunks"]:
            print(f"  {c['text']}")
            
        print("\nSYNTHESIZED ANSWER:")
        print(result["answer"])

if __name__ == "__main__":
    main()
