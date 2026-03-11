from fastmcp import FastMCP
from mem0 import Memory
import chromadb

# ==========================================
# SETUP: Connect to existing ChromaDB
# (same config as api.py — no duplication)
# ==========================================
mem0_config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:1b",
            "temperature": 0.1,
            "base_url": "http://localhost:11434"
        }
    },
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "memonic_memory",
            "path": "./memonic_memory",  # Same path as api.py
        }
    }
}

memory = Memory.from_config(config_dict=mem0_config)

# Raw ChromaDB client for direct queries (faster, no LLM overhead)
chroma_client = chromadb.PersistentClient(path="./memonic_memory")
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
        results = memory.search(query, user_id=user_id, limit=limit)
        if not results:
            return f"No memories found for user '{user_id}' matching '{query}'"

        formatted = []
        for i, r in enumerate(results, 1):
            mem_text = r.get("memory", "")
            score = r.get("score", 0)
            metadata = r.get("metadata", {})
            emotion = metadata.get("emotion", "Unknown")
            formatted.append(
                f"{i}. [{emotion}] (score: {score:.2f}) {mem_text}"
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
        results = memory.get_all(user_id=user_id)
        if not results:
            return f"No memories found for user '{user_id}'"

        formatted = []
        for i, r in enumerate(results, 1):
            mem_text = r.get("memory", "")
            metadata = r.get("metadata", {})
            emotion = metadata.get("emotion", "Unknown")
            confidence = metadata.get("speaker_confidence", 0)
            formatted.append(
                f"{i}. [emotion: {emotion}, confidence: {confidence:.2f}] {mem_text}"
            )

        return (
            f"All memories for '{user_id}' ({len(results)} total):\n"
            + "\n".join(formatted)
        )

    except Exception as e:
        return f"Error retrieving memories: {str(e)}"


@mcp.tool()
def add_memory(text: str, user_id: str, emotion: str = "Neutral") -> str:
    """
    Manually add a memory for a user (useful for testing or manual corrections).

    Args:
        text: The content to remember
        user_id: The user's ID
        emotion: Emotional context (Angry/Happy/Sad/Neutral)
    """
    try:
        memory.add(
            messages=[{"role": "user", "content": text}],
            user_id=user_id,
            metadata={"emotion": emotion, "speaker_confidence": 1.0, "source": "manual"}
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
        memory.delete(memory_id)
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
        memory.delete_all(user_id=user_id)
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
        # Query ChromaDB directly — faster than going through mem0
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
    print("   Connecting to ChromaDB at ./memonic_memory")
    mcp.run()  # Default: stdio transport (works with Claude Desktop, Cursor, etc.)