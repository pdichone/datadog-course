from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": "say hello in 3 different languages."}],
)

print(response.choices[0].message.content)
