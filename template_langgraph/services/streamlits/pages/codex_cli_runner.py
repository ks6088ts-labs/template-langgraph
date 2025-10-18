"""Simple Streamlit runner for executing shell commands and tailing their output."""

from __future__ import annotations

import html
import subprocess
import threading
import time
from datetime import datetime
from queue import Empty, Queue

import streamlit as st

LogEntry = tuple[str, str]


LOG_COLORS = {
    "stdout": "#e8f5e9",
    "stderr": "#ff7961",
}
LOG_FONT_FAMILY = "SFMono-Regular,Consolas,Menlo,monospace"
MAX_LOG_LINES = 4000


def _init_state() -> None:
    if "cli_runner" not in st.session_state:
        st.session_state.cli_runner = {
            "command": "",
            "process": None,
            "queue": Queue(),
            "logs": [],
            "start_time": None,
            "returncode": None,
            "auto_refresh": True,
        }


def _stream_reader(stream, label: str, buffer: Queue) -> None:
    for raw_line in iter(stream.readline, ""):
        buffer.put((label, raw_line.rstrip("\n")))
    stream.close()


def _start_process(command: str) -> None:
    runner = st.session_state.cli_runner
    if runner.get("process") and runner["process"].poll() is None:
        st.sidebar.warning("他のプロセスが実行中です。停止してから再度実行してください。")
        return

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    except OSError as exc:
        st.sidebar.error(f"プロセスを起動できませんでした: {exc}")
        return

    runner.update(
        {
            "command": command,
            "process": process,
            "queue": Queue(),
            "logs": [],
            "start_time": datetime.now(),
            "returncode": None,
        }
    )

    if process.stdout:
        stdout_thread = threading.Thread(
            target=_stream_reader,
            args=(process.stdout, "stdout", runner["queue"]),
            daemon=True,
        )
        stdout_thread.start()

    if process.stderr:
        stderr_thread = threading.Thread(
            target=_stream_reader,
            args=(process.stderr, "stderr", runner["queue"]),
            daemon=True,
        )
        stderr_thread.start()


def _drain_queue() -> None:
    runner = st.session_state.cli_runner
    queue: Queue | None = runner.get("queue")
    if queue is None:
        return
    logs: list[LogEntry] = runner.get("logs", [])

    while True:
        try:
            entry = queue.get_nowait()
        except Empty:
            break
        logs.append(entry)
        if len(logs) > MAX_LOG_LINES:
            logs.pop(0)

    runner["logs"] = logs


def _terminate_process() -> None:
    runner = st.session_state.cli_runner
    process = runner.get("process")
    if not process or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    runner["returncode"] = process.poll()


def _render_logs(logs: list[LogEntry]) -> None:
    if not logs:
        st.info("まだ出力はありません。コマンドを実行してください。")
        return

    log_container = st.container()
    lines = []
    for stream_label, line in logs:
        color = LOG_COLORS.get(stream_label, "#e0e0e0")
        safe_line = html.escape(line)
        lines.append(
            f"<div style='color:{color};font-family:{LOG_FONT_FAMILY};white-space:pre-wrap;margin:0;'>{safe_line}</div>"
        )
    log_container.markdown("\n".join(lines), unsafe_allow_html=True)


_init_state()
runner = st.session_state.cli_runner
process = runner.get("process")
is_running = bool(process) and process.poll() is None

st.title("Codex CLI Runner")

with st.sidebar:
    st.header("Command Settings")
    st.text("CLIコマンドを指定して実行します。")
    st.session_state.cli_runner["command"] = st.text_input(
        label="Command",
        key="cli_runner_command_input",
        value=runner.get("command", ""),
        placeholder="e.g. ls -la",
    )

    run_clicked = st.button("Run", use_container_width=True)
    stop_clicked = st.button(
        "Stop",
        use_container_width=True,
        disabled=not is_running,
    )

    runner["auto_refresh"] = st.checkbox(
        label="Auto refresh (1s)",
        value=runner.get("auto_refresh", True),
    )

    if st.button("Clear Logs", use_container_width=True):
        runner["logs"] = []

if run_clicked:
    command = st.session_state.cli_runner.get("command", "").strip()
    if not command:
        st.sidebar.error("コマンドを入力してください。")
    else:
        _start_process(command)

if stop_clicked:
    _terminate_process()

_drain_queue()

process = runner.get("process")
is_running = bool(process) and process.poll() is None

if process and not is_running and runner.get("returncode") is None:
    runner["returncode"] = process.poll()

status_placeholder = st.empty()
if is_running:
    status_placeholder.info("プロセス実行中です…")
else:
    return_code = runner.get("returncode")
    if return_code is None:
        status_placeholder.info("プロセスを待機しています。")
    elif return_code == 0:
        status_placeholder.success("プロセスが正常終了しました。")
    else:
        status_placeholder.error(f"プロセスが終了コード {return_code} で終了しました。")

if runner.get("start_time"):
    if is_running:
        elapsed = datetime.now() - runner["start_time"]
        st.caption(f"Started at {runner['start_time'].strftime('%Y-%m-%d %H:%M:%S')} | Elapsed: {elapsed}")
    else:
        st.caption(f"Started at {runner['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")

_render_logs(runner.get("logs", []))

if is_running and runner.get("auto_refresh", True):
    time.sleep(1)
    st.experimental_rerun()
