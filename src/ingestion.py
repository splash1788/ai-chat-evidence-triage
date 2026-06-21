import json
import pandas as pd

def load_chat_data(filepath):
    """
    Ingests synthetic chat data from a JSON export and converts it into a Pandas DataFrame.
    """
    print(f"[*] Loading forensic chat export from {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        df = pd.DataFrame(data)
        print(f"[+] Successfully loaded {len(df)} messages.")
        return df
        
    except FileNotFoundError:
        print(f"[-] Error: File {filepath} not found.")
        return None

def preprocess_text(df):
    """
    Normalizes the chat text by converting to lowercase for consistent keyword matching.
    (Future iterations will include stop-word removal and advanced tokenization).
    """
    print("[*] Preprocessing text data...")
    df['clean_text'] = df['message'].str.lower()
    return df

if __name__ == "__main__":
    # Test the ingestion pipeline locally
    dataset_path = "../data/synthetic_chat_v1.json"
    chat_df = load_chat_data(dataset_path)
    
    if chat_df is not None:
        processed_df = preprocess_text(chat_df)
        print(processed_df.head())
