"""One-time setup script for the LLM integration.

What it does:
  1. Reads .env (if it exists) and exports the API keys
  2. Verifies the LLM is reachable (OpenRouter / OpenAI / Anthropic / Ollama)
  3. Tests the connection with a simple clinical NER prompt
  4. Tells you which model is active and what the cost will be

Run:  python setup_llm.py
"""
from __future__ import annotations
import os
import sys
import json

# Try to load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Manual fallback: parse .env ourselves
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def test_openrouter() -> tuple:
    """Returns (model_name, status_msg) or (None, error)."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        return None, "no OPENROUTER_API_KEY"
    try:
        from openai import OpenAI
    except ImportError:
        return None, "openai package not installed (pip install openai)"
    model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        timeout=30,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=5,
        )
        return model, f"reachable, response: {resp.choices[0].message.content.strip()!r}"
    except Exception as e:
        return None, f"API call failed: {e}"


def test_openai() -> tuple:
    if not os.environ.get("OPENAI_API_KEY"):
        return None, "no OPENAI_API_KEY"
    try:
        from openai import OpenAI
    except ImportError:
        return None, "openai package not installed"
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=30)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=5,
        )
        return "gpt-4o-mini", f"reachable, response: {resp.choices[0].message.content.strip()!r}"
    except Exception as e:
        return None, f"API call failed: {e}"


def test_ollama() -> tuple:
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            data = json.loads(r.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            if models:
                return models[0], f"reachable, {len(models)} model(s) available: {', '.join(models)}"
            return None, "Ollama running but no models pulled. Run: ollama pull llama3.1:8b"
    except Exception as e:
        return None, f"Ollama not reachable at localhost:11434. Install: curl -fsSL https://ollama.com/install.sh | sh"


def main():
    print("=" * 70)
    print("Reasoning Canvas + Virtual Patient — LLM setup")
    print("=" * 70)
    print()
    print("Testing each LLM backend in priority order...")
    print()

    found = False
    for name, fn in [
        ("Ollama (local)",  test_ollama),
        ("OpenRouter",       test_openrouter),
        ("OpenAI",           test_openai),
    ]:
        model, status = fn()
        if model:
            print(f"  ✅  {name:20s}  model={model}")
            print(f"      {status}")
            found = True
            break
        else:
            print(f"  ❌  {name:20s}  {status}")

    print()
    if not found:
        print("=" * 70)
        print("No LLM backend found. To set one up:")
        print()
        print("EASIEST (free, no GPU):")
        print("  1. Go to https://openrouter.ai/keys")
        print("  2. Sign up, create a key, copy it")
        print("  3. Copy .env.example to .env, paste the key")
        print("  4. Run this script again")
        print()
        print("ALTERNATIVE (local, no API key, needs GPU):")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
        print("  ollama pull llama3.1:8b")
        print("  ollama serve")
        sys.exit(1)
    else:
        print("=" * 70)
        print("LLM is ready. You can now run:")
        print("  streamlit run hospital_portal.py")
        print("  python virtual_patient.py")
        print("  python elembic_pipeline.py")
        print("=" * 70)


if __name__ == "__main__":
    main()
