"""List Gemini models available to the configured API key."""

import os

from dotenv import load_dotenv
from google import genai


def main() -> None:
    """Print models that support content generation."""

    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing from .env.")

    client = genai.Client(api_key=api_key)

    for model in client.models.list():
        actions = getattr(model, "supported_actions", []) or []

        if "generateContent" in actions:
            print(model.name)


if __name__ == "__main__":
    main()