"""
attractor.py — AI-to-AI conversation harness for replicating attractor state experiments.

Usage:
    python attractor.py configs/local_v1_1.yaml
    python attractor.py configs/opus_v1_1.yaml

Press Ctrl+C to interrupt a run gracefully.
"""

import argparse
import json
import os
import signal
import sys
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

import anthropic
import psycopg2
import psycopg2.extras
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model providers
# ---------------------------------------------------------------------------

class ModelProvider(ABC):
    @abstractmethod
    def send(self, system_prompt: str, history: list[dict], message: str, receiving_speaker: str = "A") -> tuple[str, Optional[int]]:
        """Send a message and return (response_text, token_count)."""


def build_messages(history: list[dict], receiving_speaker: str, new_message: str) -> list[dict]:
    """
    Build a message list from the perspective of the receiving model.
    That model's own prior turns are 'assistant'; the other model's turns are 'user'.
    The incoming message is always 'user'.
    """
    messages = []
    for turn in history:
        role = "assistant" if turn["speaker"] == receiving_speaker else "user"
        messages.append({"role": role, "content": turn["content"]["message"]})
    messages.append({"role": "user", "content": new_message})
    return messages


class OllamaProvider(ModelProvider):
    def __init__(self, model: str, endpoint: str):
        self.model = model
        self.endpoint = endpoint.rstrip("/")

    def send(self, system_prompt: str, history: list[dict], message: str, receiving_speaker: str = "A") -> tuple[str, Optional[int]]:
        messages = build_messages(history, receiving_speaker, message)
        # Ollama's /api/chat ignores a top-level "system" field; it must be a system-role message.
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        resp = requests.post(f"{self.endpoint}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        text = data["message"]["content"]
        tokens = data.get("eval_count")
        return text, tokens


class AnthropicProvider(ModelProvider):
    def __init__(self, model: str):
        self.model = model
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def send(self, system_prompt: str, history: list[dict], message: str, receiving_speaker: str = "A") -> tuple[str, Optional[int]]:
        messages = build_messages(history, receiving_speaker, message)

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
        )

        text = resp.content[0].text
        tokens = resp.usage.output_tokens if resp.usage else None
        return text, tokens


def build_provider(cfg: dict) -> ModelProvider:
    provider = cfg["provider"].lower()
    if provider == "ollama":
        return OllamaProvider(model=cfg["model"], endpoint=cfg["endpoint"])
    elif provider == "anthropic":
        return AnthropicProvider(model=cfg["model"])
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_connection() -> psycopg2.extensions.connection:
    dsn = os.environ.get("ATTRACTOR_DB_DSN")
    if not dsn:
        raise EnvironmentError("ATTRACTOR_DB_DSN environment variable not set.")
    conn = psycopg2.connect(dsn)
    psycopg2.extras.register_uuid()
    return conn


def insert_run(conn, config: dict) -> str:
    run_id = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO runs (
                run_id, model_a, model_b,
                system_prompt_version, system_prompt_text,
                instigating_prompt, hard_stop_limit,
                context_window_turns, turn_delay_seconds
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                run_id,
                config["model_a"]["model"],
                config["model_b"]["model"],
                config["system_prompt_version"],
                config["system_prompt_text"],
                config["instigating_prompt"],
                config["hard_stop_limit"],
                config.get("context_window_turns"),
                config.get("turn_delay_seconds", 1.0),
            ),
        )
    conn.commit()
    return run_id


def insert_turn(conn, run_id: str, turn_number: int, speaker: str, model_version: str, message: str, token_count: Optional[int]):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO turns (run_id, turn_number, speaker, model_version, content, token_count)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                run_id,
                turn_number,
                speaker,
                model_version,
                json.dumps({"message": message}),
                token_count,
            ),
        )
    conn.commit()


def close_run(conn, run_id: str, terminated_by: str, error_detail: Optional[str] = None):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE runs SET completed_at = %s, terminated_by = %s, error_detail = %s
            WHERE run_id = %s
            """,
            (datetime.now(timezone.utc), terminated_by, error_detail, run_id),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Conversation loop
# ---------------------------------------------------------------------------

STOP_TOKEN = "/stop"

interrupted = False

def handle_interrupt(signum, frame):
    global interrupted
    interrupted = True
    print("\n\n[Interrupt received — finishing current turn and closing run gracefully...]")


def run_experiment(config_path: str):
    global interrupted

    config = load_config(config_path)
    system_prompt = config["system_prompt_text"]
    instigating_prompt = config["instigating_prompt"]
    hard_stop = config["hard_stop_limit"]
    delay = config.get("turn_delay_seconds", 1.0)
    context_window = config.get("context_window_turns")

    provider_a = build_provider(config["model_a"])
    provider_b = build_provider(config["model_b"])

    conn = get_connection()
    run_id = insert_run(conn, config)

    print(f"\n{'='*60}")
    print(f"  Run ID:    {run_id}")
    print(f"  Model A:   {config['model_a']['model']} ({config['model_a']['provider']})")
    print(f"  Model B:   {config['model_b']['model']} ({config['model_b']['provider']})")
    print(f"  Prompt:    {config['system_prompt_version']}")
    print(f"  Hard stop: {hard_stop} turns")
    print(f"{'='*60}\n")

    signal.signal(signal.SIGINT, handle_interrupt)

    history = []
    turn_number = 0
    terminated_by = "hard_stop"
    error_detail = None

    # Model A receives the instigating prompt
    current_message = instigating_prompt
    current_speaker = "A"

    try:
        while turn_number < hard_stop and not interrupted:
            if current_speaker == "A":
                turn_number += 1
            provider = provider_a if current_speaker == "A" else provider_b
            model_label = config[f"model_{current_speaker.lower()}"]["model"]

            print(f"[Turn {turn_number} — Model {current_speaker} ({model_label})]")

            # Build windowed history if configured
            if context_window:
                windowed = history[-context_window:]
            else:
                windowed = history

            try:
                response, tokens = provider.send(system_prompt, windowed, current_message, receiving_speaker=current_speaker)
            except Exception as e:
                error_detail = str(e)
                terminated_by = "error"
                print(f"\n[ERROR on turn {turn_number}: {e}]")
                break

            # Check for stop token
            if response.strip() == STOP_TOKEN:
                print(f"[Model {current_speaker} issued /stop]")
                terminated_by = f"model_{current_speaker.lower()}"
                # Record the stop turn
                insert_turn(conn, run_id, turn_number, current_speaker, model_label, STOP_TOKEN, tokens)
                history.append({"speaker": current_speaker, "content": {"message": STOP_TOKEN}})
                break

            print(f"{response}\n")

            insert_turn(conn, run_id, turn_number, current_speaker, model_label, response, tokens)
            history.append({"speaker": current_speaker, "content": {"message": response}})

            # Pass response to the other model
            current_message = response
            current_speaker = "B" if current_speaker == "A" else "A"

            if turn_number < hard_stop and not interrupted:
                time.sleep(delay)

    except Exception as e:
        error_detail = str(e)
        terminated_by = "error"
        print(f"\n[Unexpected error: {e}]")

    if interrupted:
        terminated_by = "manual"

    close_run(conn, run_id, terminated_by, error_detail)
    conn.close()

    print(f"\n{'='*60}")
    print(f"  Run complete.")
    print(f"  Turns completed: {turn_number}")
    print(f"  Terminated by:   {terminated_by}")
    if error_detail:
        print(f"  Error:           {error_detail}")
    print(f"  Run ID:          {run_id}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-to-AI attractor state experiment harness")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()
    run_experiment(args.config)
