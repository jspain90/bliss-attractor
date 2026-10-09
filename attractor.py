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
    The instigating prompt is 'user' and is shown only to Model A.
    The incoming message is always 'user' and must not already be in history.
    """
    messages = []
    for turn in history:
        if turn["speaker"] == PROMPT_SPEAKER and receiving_speaker != "A":
            continue
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
PROMPT_SPEAKER = "prompt"

interrupted = False

def handle_interrupt(signum, frame):
    global interrupted
    interrupted = True
    print("\n\n[Interrupt received — finishing current turn and closing run gracefully...]")


def run_experiment(config_path: str) -> str:
    """Run a single experiment. Returns the terminated_by value."""
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

    history = [{"speaker": PROMPT_SPEAKER, "content": {"message": instigating_prompt}}]
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

            # history[-1] is current_message; build_messages appends it separately
            prior = history[:-1]
            # Build windowed history if configured
            if context_window:
                windowed = prior[-context_window:]
            else:
                windowed = prior

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

    return terminated_by


def run_batch(config_path: str, total: int, max_retries: int = 2):
    """Run `total` experiments sequentially, retrying failed runs up to max_retries times."""
    global interrupted

    completed = 0
    skipped = 0

    print(f"\n{'#'*60}")
    print(f"  Batch start: {total} run(s) from {config_path}")
    print(f"{'#'*60}")

    for run_index in range(1, total + 1):
        if interrupted:
            print(f"\n[Batch interrupted — stopping after run {run_index - 1} of {total}]")
            break

        print(f"\n[Batch {run_index}/{total}]")

        for attempt in range(1, max_retries + 2):  # 1 initial + max_retries retries
            if interrupted:
                break

            if attempt > 1:
                print(f"  [Retry {attempt - 1}/{max_retries}]")

            terminated_by = run_experiment(config_path)

            if interrupted:
                break

            if terminated_by == "error":
                if attempt <= max_retries:
                    print(f"  [Run failed — will retry ({attempt}/{max_retries})]")
                else:
                    print(f"  [Run failed after {max_retries} retr{'y' if max_retries == 1 else 'ies'} — skipping]")
                    skipped += 1
            else:
                completed += 1
                break

        if interrupted:
            break

    total_attempted = run_index if interrupted else total
    print(f"\n{'#'*60}")
    print(f"  Batch complete.")
    print(f"  Completed: {completed}  /  Skipped (all retries failed): {skipped}")
    if interrupted:
        print(f"  Aborted by interrupt after run {run_index - 1 if interrupted else total_attempted}.")
    print(f"{'#'*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-to-AI attractor state experiment harness")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--runs", type=int, default=1, metavar="N",
        help="Number of sequential runs to execute (default: 1)",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_interrupt)

    if args.runs == 1:
        run_experiment(args.config)
    else:
        run_batch(args.config, total=args.runs)
