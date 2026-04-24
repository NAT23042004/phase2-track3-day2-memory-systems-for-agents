from typing import Any, List

import tiktoken


class ContextManager:
    def __init__(self, model_name="gpt-3.5-turbo", max_tokens=2000):
        self.encoder = tiktoken.encoding_for_model(model_name)
        self.max_tokens = max_tokens
        # Levels: 1 (System), 2 (Prefs), 3 (History), 4 (Retrieved)

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def manage_context(self, system_prompt: str, prefs: str, history: List[Any], retrieved: List[str]) -> str:
        """
        Combines and trims context based on priority levels.
        """
        # Current token counts
        tokens_system = self.count_tokens(system_prompt)
        tokens_prefs = self.count_tokens(prefs)

        # We must keep Level 1 (System Prompt)
        current_total = tokens_system

        final_prefs = prefs
        if current_total + tokens_prefs > self.max_tokens:
            # This is unlikely for prefs, but let's be safe.
            # Truncate prefs if somehow they exceed the limit by themselves
            final_prefs = self.encoder.decode(self.encoder.encode(prefs)[: self.max_tokens - tokens_system])

        current_total += self.count_tokens(final_prefs)

        # Level 3: History (Add from newest to oldest until limit)
        final_history = []
        history_tokens = 0
        for msg in reversed(history):
            msg_text = str(msg)
            msg_tokens = self.count_tokens(msg_text)
            if (
                current_total + history_tokens + msg_tokens <= self.max_tokens * 0.8
            ):  # Reserve 20% for retrieved & buffer
                final_history.insert(0, msg_text)
                history_tokens += msg_tokens
            else:
                break

        current_total += history_tokens

        # Level 4: Retrieved (Add until limit)
        final_retrieved = []
        for doc in retrieved:
            doc_tokens = self.count_tokens(doc)
            if current_total + doc_tokens <= self.max_tokens:
                final_retrieved.append(doc)
                current_total += doc_tokens
            else:
                break

        # Construct final context string
        context = f"SYSTEM: {system_prompt}\n\nUSER PREFERENCES: {final_prefs}\n\n"
        if final_retrieved:
            context += "RETRIEVED KNOWLEDGE/EXPERIENCES:\n" + "\n".join(final_retrieved) + "\n\n"
        context += "CONVERSATION HISTORY:\n" + "\n".join(final_history)

        return context
