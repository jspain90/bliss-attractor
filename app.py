"""
app.py — Transcript viewer for attractor state experiments.

Usage:
    python app.py

Then open http://localhost:5000 in your browser.
"""

import os
from datetime import datetime, timezone
from flask import Flask, jsonify, render_template
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


def get_connection():
    dsn = os.environ.get("ATTRACTOR_DB_DSN")
    if not dsn:
        raise EnvironmentError("ATTRACTOR_DB_DSN environment variable not set.")
    conn = psycopg2.connect(dsn)
    psycopg2.extras.register_uuid()
    return conn


def serialize(value):
    """Convert non-JSON-serializable types."""
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def row_to_dict(cursor, row):
    return {col.name: serialize(value) for col, value in zip(cursor.description, row)}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/runs")
def list_runs():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    r.run_id,
                    r.created_at,
                    r.completed_at,
                    r.model_a,
                    r.model_b,
                    r.system_prompt_version,
                    r.instigating_prompt,
                    r.terminated_by,
                    r.attractor_observed,
                    r.hard_stop_limit,
                    COUNT(t.turn_id) AS message_count,
                    MAX(t.turn_number) AS turn_count
                FROM runs r
                LEFT JOIN turns t ON t.run_id = r.run_id
                GROUP BY r.run_id
                ORDER BY r.created_at DESC
            """)
            rows = [row_to_dict(cur, row) for row in cur.fetchall()]
        return jsonify(rows)
    finally:
        conn.close()


@app.route("/api/runs/<run_id>")
def get_run(run_id):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Run metadata
            cur.execute("SELECT * FROM runs WHERE run_id = %s", (run_id,))
            run_row = cur.fetchone()
            if not run_row:
                return jsonify({"error": "Run not found"}), 404
            run = row_to_dict(cur, run_row)

            # All turns ordered
            cur.execute("""
                SELECT turn_number, speaker, model_version, content, token_count, created_at
                FROM turns
                WHERE run_id = %s
                ORDER BY turn_number ASC, speaker ASC
            """, (run_id,))
            turn_rows = [row_to_dict(cur, row) for row in cur.fetchall()]

        # Group individual DB rows into A+B exchange pairs.
        # 1 turn = 1 complete A+B exchange. A new turn starts each time
        # speaker A speaks; B's reply belongs to that same turn.
        grouped_turns = []
        current_exchange = None
        exchange_number = 0

        for msg in turn_rows:
            entry = {
                "speaker": msg["speaker"],
                "model_version": msg["model_version"],
                "message": msg["content"]["message"] if isinstance(msg["content"], dict) else msg["content"],
                "token_count": msg["token_count"],
                "created_at": msg["created_at"],
            }
            if msg["speaker"] == "A":
                exchange_number += 1
                current_exchange = {"turn_number": exchange_number, "messages": [entry]}
                grouped_turns.append(current_exchange)
            else:
                if current_exchange is None:
                    exchange_number += 1
                    current_exchange = {"turn_number": exchange_number, "messages": []}
                    grouped_turns.append(current_exchange)
                current_exchange["messages"].append(entry)

        return jsonify({"run": run, "turns": grouped_turns})
    finally:
        conn.close()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
