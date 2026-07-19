import os
import uuid
import chromadb
from chromadb.utils import embedding_functions
from src.helper_functions import normalize_name

def get_vector_collection():
    """Helper to safely resolve the database path and return the collection."""
    # Dynamically find project root so it runs perfectly from any folder!
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    db_path = os.path.join(base_dir, "data", "scout_cache", "vector_db")
    
    chroma_client = chromadb.PersistentClient(path=db_path)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    return chroma_client.get_or_create_collection(
        name="player_narratives", 
        embedding_function=embedding_fn
    )

def add_custom_scouting_report(player_name: str, report_text: str):
    """
    Saves a proprietary scouting report into the vector database.
    This makes it instantly searchable by your multi-agent supervisor.
    """
    collection = get_vector_collection()
    clean_name = normalize_name(player_name)
    doc_id = str(uuid.uuid4())
    
    collection.add(
        documents=[report_text],
        metadatas=[{"player_match_name": clean_name, "original_name": player_name}],
        ids=[doc_id]
    )
    print(f"Successfully filed proprietary report for {player_name} (ID: {doc_id})")

# ==========================================
# INTERACTIVE SCOUT TERMINAL ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    print("\nSMASHING MONKEYS - PROPRIETARY INTEL PORTAL")
    print("="*48)
    
    while True:
        player = input("\nEnter player name (or type 'exit' to quit): ").strip()
        if player.lower() == 'exit':
            print("Exiting Intel Portal. Scout safely!")
            break
            
        if not player:
            continue
            
        print(f"Enter scouting report for {player} (Press Enter twice when finished):")
        
        # Multi-line input parser for long-form scout notes
        lines = []
        while True:
            line = input()
            if line:
                lines.append(line)
            else:
                break
        report = " ".join(lines).strip()
        
        if not report:
            print("Empty report discarded.")
            continue
            
        add_custom_scouting_report(player, report)