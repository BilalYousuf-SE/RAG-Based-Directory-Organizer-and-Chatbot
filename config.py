"""
Central place for secrets/config.

Loads GROQ_API_KEY from a local .env file (see .env.example) or from the
real environment. Every module that needs the Groq client imports the key
from here instead of hardcoding it.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Copy .env.example to .env and fill in "
        "your key, or set the GROQ_API_KEY environment variable."
    )
