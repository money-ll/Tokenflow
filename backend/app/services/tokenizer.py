try:
    import tiktoken
except Exception:
    tiktoken = None

class TokenCounter:
    def __init__(self):
        self.encoding = None
        if tiktoken:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass

    def count(self, text: str) -> int:
        if self.encoding:
            return len(self.encoding.encode(text))
        # Stable approximation when tiktoken is unavailable.
        return max(0, int(len(text.split()) * 1.3))

    def reduction_rate(self, raw, optimized):
        if raw <= 0:
            return 0.0
        return round((raw - optimized) / raw * 100, 2)
