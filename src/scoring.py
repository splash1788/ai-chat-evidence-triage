import pandas as pd
import re

# Dictionaries of investigative interest
ILLICIT_KEYWORDS = ['package', 'kilos', 'shipment', 'warehouse', 'payment', 'cash']
# Regex pattern for a standard Bitcoin wallet address
BTC_REGEX = r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b'

def calculate_relevance(text):
    """
    Calculates a relevance score based on heuristic keyword hits and Regex matches.
    """
    score = 0
    
    # 1. Check for standard keywords (1 point each)
    for word in ILLICIT_KEYWORDS:
        if word in text:
            score += 1
            
    # 2. Check for rigid data structures like Cryptocurrency wallets (5 points)
    if re.search(BTC_REGEX, text):
        score += 5
        
    return score

def run_triage_engine(df):
    """
    Applies the scoring logic to the entire dataset and sorts by highest priority.
    """
    print("[*] Running relevance scoring engine...")
    df['relevance_score'] = df['clean_text'].apply(calculate_relevance)
    
    # Sort messages by score (highest first) to simulate triage
    prioritized_df = df.sort_values(by='relevance_score', ascending=False)
    return prioritized_df

if __name__ == "__main__":
    from ingestion import load_chat_data, preprocess_text
    
    # Run the full pipeline
    df = load_chat_data("../data/synthetic_chat_v1.json")
    if df is not None:
        df = preprocess_text(df)
        triaged_data = run_triage_engine(df)
        
        print("\n=== TRIAGE OUTPUT (TOP 3 HIGHEST PRIORITY) ===")
        # Print the top 3 most relevant messages
        print(triaged_data[['timestamp', 'sender_id', 'message', 'relevance_score']].head(3))
        
        # Save output for investigator review
        output_file = "../data/triage_output_report.csv"
        triaged_data.to_csv(output_file, index=False)
        print(f"\n[+] Full triage report saved to {output_file}")
