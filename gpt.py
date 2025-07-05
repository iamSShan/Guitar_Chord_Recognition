import os
import json
from openai import OpenAI, AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

CACHE_FILE = "chord_gpt_cache.json"


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


# Load existing cache at start
chord_cache = load_cache()

# Initialize client with your API key (set this env var securely)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# For Azure OpenAI
# client = AzureOpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),  # Your specified version
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
# )


def get_chord_info_from_gpt(chord_name: str) -> str:
    prompt = (
        f"I play the {chord_name} chord. Please list **all finger placements**, "
        "including the pinky if used, and keep it brief (1 sentence per finger)."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise guitar tutor."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.7,
    )
    # Extract and return the assistant’s reply
    return response.choices[0].message.content.strip()


def get_chord_info(chord_name):
    # Check file-based cache
    if chord_name in chord_cache:
        return chord_cache[chord_name]

    # Temp condition:
    # if chord_name in custom_chord_info:
    #     response = custom_chord_info[chord_name]

    # else:
    response = get_chord_info_from_gpt(chord_name)

    # Call GPT and save result
    chord_cache[chord_name] = response
    save_cache(chord_cache)
    return response


# Example usage
# info = get_chord_info("D major")
# print(info)
