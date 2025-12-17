chat_memory = {}

def get_memory(session_id: str):
    return chat_memory.get(session_id, [])

def add_message(session_id: str, role: str, content: str):
    chat_memory.setdefault(session_id, []).append(
        {"role": role, "content": content}
    )
    chat_memory[session_id] = chat_memory[session_id][-10:]
