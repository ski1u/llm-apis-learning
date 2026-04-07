from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = "gemini-2.5-flash"
config = types.GenerateContentConfig(
    system_instruction="You are a helpful assistant that will aid in user's requests",
    max_output_tokens=1024,
    temperature=0.6,
    response_mime_type="text/plain",
    thinking_config=types.ThinkingConfig(thinking_budget=320),
)

# ---

def single_turn(
    user_msg: str,
    # system_instruction: str
):
    response = client.models.generate_content(
        model=model,
        contents=user_msg,
        config=config,
    )
    usage = response.usage_metadata
    print(f"""{response.text}
    -----
    input: {usage.prompt_token_count}
    output: {usage.candidates_token_count}
    """)

# single_turn(
#     user_msg: "Write a short story about a cat who can talk",
#     system_instruction: "You are a helpful assistant that writes short stories"
# )

# ---

class Conversation:
    def __init__(self, system: str | None = None):
        self.system = system.strip() if system and system.strip() else None
        self.history: list[types.Content] = []
        self.total_tokens = 0

    def append_history(self, role: str, content: str):
        self.history.append(
            types.Content(role=role, parts=[types.Part(text=content)])
        )

    def chat(self, user_msg: str,
        # system_instruction: str
    ) -> str:
        self.append_history(role="user", content=user_msg)

        response = client.models.generate_content(
            model=model, config=config,
            contents=self.history
        ); reply = response.text; usage = response.usuage_metadata
        turn_tokens = (usage.prompt_token_count or 0) + (usage.candidates_token_count or 0)

        self.append_history(role="model", content=reply)
        self.total_tokens += turn_tokens

        print(f"Tokens this turn: {turn_tokens}")
        return reply

    def show_history(self) -> None:
        print("History:")
        for msg in self.history:
            text = msg.parts[0].text
            preview = text[:40] + "..." if len(text) > 40 else text

            print(f"{msg.role.upper()}: {preview}")
        print()