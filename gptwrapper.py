class GPT4Wrapper:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        openai.api_key = open("key.txt").read().strip()

    def make_query_args(self, user_str, n_query=1):
        query_args = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible.",
                },
                {"role": "user", "content": user_str},
            ],
            "n": n_query,
        }
        return query_args

    def compute_num_tokens(self, user_str: str) -> int:
        return len(self.tokenizer.encode(user_str))

    def send_query(self, user_str, n_query=1):
        print(f"# tokens sent to GPT: {self.compute_num_tokens(user_str)}")
        query_args = self.make_query_args(user_str, n_query)
        completion = openai.ChatCompletion.create(**query_args)
        result = completion.choices[0]["message"]["content"]
        return result