import json
from llama_cpp import Llama


class MistralGgufAssistant:
    def __init__(self, model: Llama):
        self.model = model
        self.linebreak_token = 13

        self.role_tokens = {
            "user": 2188,
            "bot": 12435,
            "system": 1587
        }

    def get_answer(self, prompt_json: str, top_k: int = 15, top_p: float = 0.8, temperature: float = 0.6,
                   repeat_penalty: float = 1.2):
        """{"messages": [{"role": "system", "message": "You are an assistant"}, {"role": "user", "message": "hi"},
        {"role": "bot", "message": "hello"}, {"role": "user", "message": "how are you?"}]}"""
        messages = json.loads(prompt_json)['messages']
        tokens = []
        for message in messages:
            tokens += self.__get_message_tokens(role=message['role'], content=message['message'])

        tokens += [self.model.token_bos(), self.role_tokens['bot'], self.linebreak_token]

        generator = self.model.generate(
            tokens,
            top_k=top_k,
            top_p=top_p,
            temp=temperature,
            repeat_penalty=repeat_penalty
        )

        answer = ""

        for token in generator:
            tokens.append(token)
            if token == self.model.token_eos():
                break
            answer += self.model.detokenize([token]).decode("utf-8", errors="ignore")

        return answer

    def __get_message_tokens(self, role, content):
        message_tokens = self.model.tokenize(content.encode("utf-8"))
        message_tokens.insert(1, self.role_tokens[role])
        message_tokens.insert(2, self.linebreak_token)
        message_tokens.append(self.model.token_eos())
        return message_tokens
