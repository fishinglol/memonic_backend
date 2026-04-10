from fastmcp import FastMCP
import chromadb
import ollama
import uuid
import time

try:
    from .config import CHROMA_PATH, EMBEDDING_MODEL, LLM_MODEL_SUMMARY
except Exception:
    from config import CHROMA_PATH, EMBEDDING_MODEL, LLM_MODEL_SUMMARY

# ==========================================
# SETUP: Connect to the SAME ChromaDB used by api.py / memory.py
# No more mem0 — everything reads/writes the same collection.
# ==========================================
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection("memonic_memory")

# ==========================================
# MCP SERVER
# ==========================================
mcp = FastMCP("Memonic Memory Server")


@mcp.tool()
def search_memory(query: str, user_id: str, limit: int = 5) -> str:
    """
    Semantic search through a user's long-term memory in ChromaDB.
    Use this when you want to find what Memonic remembers about a specific topic for a user.

    Args:
        query: What to search for (e.g. "what does user like to eat?")
        user_id: The user's ID (must match enrolled voice profile name)
        limit: Max number of results to return (default 5)
    """
    try:
        # Generate embedding using the same model as memory.py
        res = ollama.embed(model=EMBEDDING_MODEL, input=query)
        query_embedding = res["embeddings"][0]

        results = collection.query(
            query_embeddings=[query_embedding],
            where={"user_id": user_id},
            n_results=limit,
            include=["documents", "metadatas", "distances"]
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            return f"No memories found for user '{user_id}' matching '{query}'"

        formatted = []
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
            emotion = meta.get("emotion", "Unknown") if meta else "Unknown"
            # ChromaDB returns L2 distance; convert to a rough similarity score
            score = max(0, 1 - dist / 2)
            formatted.append(
                f"{i}. [{emotion}] (score: {score:.2f}) {doc}"
            )

        return f"Memories for '{user_id}' about '{query}':\n" + "\n".join(formatted)

    except Exception as e:
        return f"Error searching memory: {str(e)}"


@mcp.tool()
def get_all_memories(user_id: str) -> str:
    """
    Retrieve ALL stored memories for a specific user.
    Use this to get a full picture of what Memonic knows about someone.

    Args:
        user_id: The user's ID
    """
    try:
        results = collection.get(
            where={"user_id": user_id},
            include=["documents", "metadatas"]
        )

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        if not documents:
            return f"No memories found for user '{user_id}'"

        formatted = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
            emotion = meta.get("emotion", "Unknown") if meta else "Unknown"
            confidence = meta.get("speaker_confidence", 0) if meta else 0
            formatted.append(
                f"{i}. [emotion: {emotion}, confidence: {confidence:.2f}] {doc}"
            )

        return (
            f"All memories for '{user_id}' ({len(documents)} total):\n"
            + "\n".join(formatted)
        )

    except Exception as e:
        return f"Error retrieving memories: {str(e)}"


@mcp.tool()
def add_memory(text: str, user_id: str, emotion: str = "Neutral") -> str:
    """
    Manually add a memory for a user (useful for testing or manual corrections).
    Uses the same embedding model and ChromaDB collection as the main pipeline.

    Args:
        text: The content to remember
        user_id: The user's ID
        emotion: Emotional context (Angry/Happy/Sad/Neutral)
    """
    try:
        res = ollama.embed(model=EMBEDDING_MODEL, input=text)
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[res["embeddings"][0]],
            documents=[text],
            metadatas=[{
                "user_id": user_id,
                "memory": text,
                "emotion": emotion,
                "speaker_confidence": 1.0,
                "timestamp": time.time(),
                "source": "manual_mcp"
            }]
        )
        return f"Memory added for '{user_id}': {text}"

    except Exception as e:
        return f"Error adding memory: {str(e)}"


@mcp.tool()
def delete_memory(memory_id: str) -> str:
    """
    Delete a specific memory by its ID.
    Get the memory ID from search_memory or get_all_memories first.

    Args:
        memory_id: The UUID of the memory to delete
    """
    try:
        collection.delete(ids=[memory_id])
        return f"Memory '{memory_id}' deleted successfully."
    except Exception as e:
        return f"Error deleting memory: {str(e)}"


@mcp.tool()
def delete_all_memories(user_id: str) -> str:
    """
    Delete ALL memories for a user. Use with caution — this is irreversible.

    Args:
        user_id: The user's ID to wipe
    """
    try:
        collection.delete(where={"user_id": user_id})
        return f"All memories for '{user_id}' have been deleted."
    except Exception as e:
        return f"Error deleting memories: {str(e)}"


@mcp.tool()
def list_users() -> str:
    """
    List all user IDs that have memories stored in the system.
    Useful for seeing who Memonic knows about.
    """
    try:
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        user_ids = set()
        for meta in metadatas:
            if meta and "user_id" in meta:
                user_ids.add(meta["user_id"])

        if not user_ids:
            return "No users with stored memories found."

        return "Users with memories:\n" + "\n".join(f"- {uid}" for uid in sorted(user_ids))

    except Exception as e:
        return f"Error listing users: {str(e)}"


@mcp.tool()
def get_memory_stats() -> str:
    """
    Get overall stats about the memory database.
    Shows total memory count and breakdown per user.
    """
    try:
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])

        total = len(metadatas)
        user_counts: dict = {}
        emotion_counts: dict = {}

        for meta in metadatas:
            if not meta:
                continue
            uid = meta.get("user_id", "unknown")
            emotion = meta.get("emotion", "Unknown")
            user_counts[uid] = user_counts.get(uid, 0) + 1
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        lines = [f"Total memories: {total}", "", "Per user:"]
        for uid, count in sorted(user_counts.items()):
            lines.append(f"  {uid}: {count} memories")

        lines += ["", "Emotion breakdown:"]
        for emotion, count in sorted(emotion_counts.items()):
            lines.append(f"  {emotion}: {count}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error getting stats: {str(e)}"


# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    print("🧠 Memonic MCP Server starting...")
    print(f"   Connecting to ChromaDB at {CHROMA_PATH}")
    print(f"   Embedding model: {EMBEDDING_MODEL}")
    mcp.run()  # Default: stdio transport (works with Claude Desktop, Cursor, etc.)