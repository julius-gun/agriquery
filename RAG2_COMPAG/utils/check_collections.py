import chromadb
import os
from chromadb.config import Settings

# Disable Hugging Face telemetry
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

CHROMA_PERSIST_DIR = "chroma_db" # Make sure this matches your script

if not os.path.exists(CHROMA_PERSIST_DIR):
    print(f"Error: Directory '{CHROMA_PERSIST_DIR}' not found in current directory '{os.getcwd()}'")
else:
    try:
        print(f"Attempting to connect to ChromaDB at: {os.path.abspath(CHROMA_PERSIST_DIR)}")
        # Ensure you are using the correct client type if not default
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
        print("Client initialized.")
        collections = client.list_collections()
        print("\nCollections found in the database (inside chroma.sqlite3):")
        if collections:
            for collection in collections:
                # Collection objects have a 'name' attribute
                print(f"- {collection.name}")
        else:
            print("  (No collections found)")

        # Specific check:
        target_name = "german_manual_cs200_os100"
        print(f"\nChecking specifically for '{target_name}'...")
        try:
             # Use get_collection which raises an exception if not found
             collection_obj = client.get_collection(name=target_name)
             print(f"  SUCCESS: Collection '{target_name}' found!")
             print(f"  Number of items in collection: {collection_obj.count()}")
        except Exception as get_exc:
             # Catching generic Exception, but specific Chroma errors might be better
             # E.g. ValueError if collection doesn't exist in some versions/setups
             print(f"  FAILED: Collection '{target_name}' not found. Error details: {get_exc}")


    except Exception as e:
        print(f"\nAn error occurred while trying to access the database: {e}")
        import traceback
        traceback.print_exc()
