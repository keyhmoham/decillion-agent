"""Coding / shell tools that operate on the session's sandbox."""

from __future__ import annotations

import difflib
import re
import shlex
import uuid

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .base import ToolError, make_tool, truncate_middle
from .context import ToolContext, current_call_id

MAX_READ_BYTES = 2_000_000
MAX_LINE_CHARS = 2000


# ---------------------------------------------------------------------------------- schemas
class BashArgs(BaseModel):
    command: str = Field(description="Shell command to run (bash). Chain with && for several steps.")
    timeout: int = Field(120, description="Max seconds before the command is killed (1-900).")
    background: bool = Field(False, description="true = start a long-running process (server, watcher) and return a job id.")
    workdir: str | None = Field(None, description="Directory to run in (default: the workspace).")


class ReadFileArgs(BaseModel):
    path: str = Field(description="File path (absolute, or relative to the workspace).")
    offset: int = Field(1, description="First line to read (1-based).")
    limit: int = Field(400, description="Maximum number of lines to return.")


class WriteFileArgs(BaseModel):
    path: str = Field(description="File path. Parent directories are created automatically.")
    content: str = Field(description="Complete new content of the file.")


class EditFileArgs(BaseModel):
    path: str = Field(description="File to modify.")
    old_string: str = Field(description="Exact existing text to replace (copy it from read_file, without line numbers).")
    new_string: str = Field(description="Replacement text.")
    replace_all: bool = Field(False, description="Replace every occurrence instead of requiring a unique match.")


class ListDirArgs(BaseModel):
    path: str = Field(".", description="Directory to list.")
    depth: int = Field(2, description="How many levels to show (1-4).")


class GlobArgs(BaseModel):
    pattern: str = Field(description="Glob such as '**/*.py' or 'src/**/*.ts'.")
    path: str = Field(".", description="Directory to search in.")


class GrepArgs(BaseModel):
    pattern: str = Field(description="Regular expression to search for.")
    path: str = Field(".", description="File or directory to search.")
    glob: str | None = Field(None, description="Only search files matching this glob, e.g. '*.py'.")
    ignore_case: bool = Field(False, description="Case-insensitive search.")
    context: int = Field(0, description="Lines of context around each match.")


class JobArgs(BaseModel):
    job_id: str = Field(description="Job id returned by bash(background=true).")
    action: str = Field("output", description="output = show recent log lines, kill = stop the job.")
    tail: int = Field(60, description="Log lines to show (output).")


# ---------------------------------------------------------------------------------- helpers
def _check_job(job_id: str) -> None:
    if not re.fullmatch(r"[a-f0-9]{4,12}", job_id or ""):
        raise ToolError(f"Invalid job id {job_id!r}.")


def _fmt_exec(res, limit: int) -> str:
    parts = []
    if res.stdout.strip():
        parts.append(res.stdout.rstrip("\n"))
    if res.stderr.strip():
        parts.append(("[stderr]\n" if res.stdout.strip() else "") + res.stderr.rstrip("\n"))
    body = "\n".join(parts) if parts else "(no output)"
    body = truncate_middle(body, limit, "narrow the command or redirect output to a file")
    footer = f"\n[exit code {res.exit_code}]"
    if res.timed_out:
        footer = "\n[TIMED OUT - command was killed after the timeout. Use background=true for long-running processes.]"
    if res.truncated:
        footer += " [output was truncated]"
    return body + footer


def build_sandbox_tools(ctx: ToolContext) -> list[BaseTool]:
    sb = ctx.sandbox
    out_limit = ctx.settings.tool_output_limit

    async def stream_cb(stream: str, text: str) -> None:
        await ctx.emit("tool_output", call_id=current_call_id.get(), stream=stream, text=text)

    # ------------------------------------------------------------------ bash
    async def bash(command: str, timeout: int = 120, background: bool = False, workdir: str | None = None) -> str:
        cwd = sb.resolve(workdir) if workdir else None
        if background:
            job = uuid.uuid4().hex[:6]
            q = shlex.quote
            launcher = (
                f"mkdir -p /tmp/a1jobs && cd {q(cwd or sb.workdir)} && "
                f"(nohup setsid bash -c {q(command)} > /tmp/a1jobs/{job}.log 2>&1 & echo $! > /tmp/a1jobs/{job}.pid); "
                f"sleep 1.5; tail -n 15 /tmp/a1jobs/{job}.log"
            )
            res = await sb.exec(launcher, timeout=30)
            return f"Started background job {job}. Use bash_job(job_id='{job}') to read its log, or action='kill' to stop it.\n--- first output ---\n{res.stdout.strip() or '(none yet)'}"
        res = await sb.exec(command, cwd=cwd, timeout=timeout, on_output=stream_cb)
        return _fmt_exec(res, out_limit)

    async def bash_job(job_id: str, action: str = "output", tail: int = 60) -> str:
        _check_job(job_id)
        if action == "kill":
            await sb.exec(f"p=$(cat /tmp/a1jobs/{job_id}.pid 2>/dev/null) && (kill -TERM -- -$p 2>/dev/null || kill -TERM $p 2>/dev/null); echo done", timeout=15)
            return f"Sent SIGTERM to job {job_id}."
        script = (
            f"f=/tmp/a1jobs/{job_id}.log; p=/tmp/a1jobs/{job_id}.pid; "
            f"[ -f \"$f\" ] || {{ echo 'unknown job'; exit 3; }}; "
            f"st=$(ps -o stat= -p \"$(cat $p 2>/dev/null)\" 2>/dev/null | cut -c1); "
            f"if [ -n \"$st\" ] && [ \"$st\" != Z ]; then echo '[job is still running]'; else echo '[job has exited]'; fi; "
            f"tail -n {int(tail)} \"$f\""
        )
        res = await sb.exec(script, timeout=15)
        if res.exit_code == 3:
            raise ToolError(f"No such background job: {job_id}")
        return truncate_middle(res.stdout, out_limit)

    # ------------------------------------------------------------------ files
    async def read_file(path: str, offset: int = 1, limit: int = 400) -> str:
        limit_lines = max(1, int(limit))
        full = sb.resolve(path)
        kind = await sb.exists(full)
        if kind is None:
            raise ToolError(f"File not found: {full}. Use list_dir or glob to find the right path.")
        if kind == "dir":
            raise ToolError(f"{full} is a directory. Use list_dir instead.")
        data = await sb.read_bytes(full, MAX_READ_BYTES)
        if b"\x00" in data[:8192]:
            raise ToolError(f"{full} looks like a binary file ({len(data)} bytes); not showing it.")
        lines = data.decode("utf-8", "replace").splitlines()
        total = len(lines)
        offset = max(1, int(offset))
        chunk = lines[offset - 1 : offset - 1 + int(limit_lines)]
        if not chunk and total:
            raise ToolError(f"offset {offset} is past the end of the file ({total} lines).")
        numbered = [f"{i:>6}\t{(l[:MAX_LINE_CHARS] + '…') if len(l) > MAX_LINE_CHARS else l}" for i, l in enumerate(chunk, start=offset)]
        header = f"{full} ({total} lines)"
        footer = ""
        if offset - 1 + len(chunk) < total:
            footer = f"\n[showing lines {offset}-{offset + len(chunk) - 1} of {total}; call read_file with offset={offset + len(chunk)} for more]"
        if not total:
            return f"{header}\n(empty file)"
        return f"{header}\n" + "\n".join(numbered) + footer

    async def write_file(path: str, content: str) -> str:
        full = sb.resolve(path)
        kind = await sb.exists(full)
        if kind == "dir":
            raise ToolError(f"{full} is a directory.")
        await sb.write_bytes(full, content.encode("utf-8"))
        await ctx.emit("file_changed", path=full, action="update" if kind else "create")
        n_lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        return f"{'Overwrote' if kind else 'Created'} {full} ({len(content.encode())} bytes, {n_lines} lines)."

    async def edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        full = sb.resolve(path)
        if await sb.exists(full) != "file":
            raise ToolError(f"File not found: {full}.")
        text = (await sb.read_bytes(full, MAX_READ_BYTES)).decode("utf-8", "replace")
        if old_string == new_string:
            raise ToolError("old_string and new_string are identical; nothing to change.")
        if not old_string:
            raise ToolError("old_string is empty. To create or replace a whole file use write_file.")
        count = text.count(old_string)
        if count == 0:
            # help the model recover: show the closest existing line
            first = old_string.strip().splitlines()[0] if old_string.strip() else ""
            close = difflib.get_close_matches(first, [l.strip() for l in text.splitlines()], n=1, cutoff=0.6)
            hint = f" Closest existing line: {close[0]!r}." if close else ""
            raise ToolError(f"old_string was not found in {full}. It must match the file exactly (whitespace included).{hint} Re-read the file with read_file.")
        if count > 1 and not replace_all:
            raise ToolError(f"old_string occurs {count} times in {full}. Add more surrounding context to make it unique, or pass replace_all=true.")
        new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        await sb.write_bytes(full, new_text.encode("utf-8"))
        await ctx.emit("file_changed", path=full, action="update")
        diff = "\n".join(list(difflib.unified_diff(text.splitlines(), new_text.splitlines(), "before", "after", lineterm="", n=2))[:40])
        return f"Edited {full} ({count if replace_all else 1} replacement{'s' if replace_all and count > 1 else ''}).\n{diff}"

    async def list_dir(path: str = ".", depth: int = 2) -> str:
        full = sb.resolve(path)
        depth = max(1, min(int(depth), 4))
        if await sb.exists(full) != "dir":
            raise ToolError(f"Not a directory: {full}")
        cmd = (
            f"cd {shlex.quote(full)} && find . -mindepth 1 -maxdepth {depth} "
            r"\( -name .git -o -name node_modules -o -name __pycache__ -o -name .venv -o -name .cache \) -prune -o "
            r"-printf '%y %P\n' 2>/dev/null | sort -k2 | head -n 300"
        )
        res = await sb.exec(cmd, timeout=20)
        lines = []
        for line in res.stdout.splitlines():
            kind, _, rel = line.partition(" ")
            if rel:
                lines.append(f"{rel}/" if kind == "d" else rel)
        if not lines:
            return f"{full}: (empty)"
        return f"{full}:\n" + "\n".join(lines) + ("\n[list truncated at 300 entries]" if len(lines) >= 300 else "")

    async def glob(pattern: str, path: str = ".") -> str:
        full = sb.resolve(path)
        pat = pattern if "/" in pattern or "**" in pattern else f"**/{pattern}"
        cmd = (
            f"cd {shlex.quote(full)} && shopt -s globstar nullglob dotglob && "
            f"for f in {pat}; do [ -f \"$f\" ] && case \"$f\" in .git/*|*/node_modules/*|node_modules/*|*/__pycache__/*) ;; *) echo \"$f\";; esac; done | head -n 200"
        )
        res = await sb.exec(cmd, timeout=30)
        return res.stdout.strip() or "(no matches)"

    async def grep(pattern: str, path: str = ".", glob: str | None = None, ignore_case: bool = False, context: int = 0) -> str:
        full = shlex.quote(sb.resolve(path))
        ic = "-i" if ignore_case else ""
        ctxf = f"-C {int(context)}" if context else ""
        g = f"-g {shlex.quote(glob)}" if glob else ""
        gi = f"--include={shlex.quote(glob)}" if glob else ""
        cmd = (
            f"if command -v rg >/dev/null 2>&1; then rg -n --no-heading {ic} {ctxf} {g} -e {shlex.quote(pattern)} {full}; "
            f"else grep -rnE {ic} {ctxf} {gi} --exclude-dir=.git --exclude-dir=node_modules -e {shlex.quote(pattern)} {full}; fi | head -n 200"
        )
        res = await sb.exec(cmd, timeout=40)
        return truncate_middle(res.stdout.strip() or "(no matches)", out_limit)

    tools = [
        make_tool(
            "bash",
            "Run a shell command in the sandbox (Linux, bash). Use it to run code, tests, git, package managers. "
            "Returns stdout, stderr and the exit code.",
            BashArgs, bash, mutating=True, category="sandbox",
        ),
        make_tool("bash_job", "Read the log of (or kill) a background job started with bash(background=true).", JobArgs, bash_job, category="sandbox"),
        make_tool(
            "read_file",
            "Read a text file with line numbers. Always read a file before editing it.",
            ReadFileArgs, read_file, category="sandbox",
        ),
        make_tool("write_file", "Create a file or overwrite it completely with the given content.", WriteFileArgs, write_file, mutating=True, category="sandbox"),
        make_tool(
            "edit_file",
            "Replace exact text in an existing file. Prefer this over write_file for small changes.",
            EditFileArgs, edit_file, mutating=True, category="sandbox",
        ),
        make_tool("list_dir", "List files and folders (tree view).", ListDirArgs, list_dir, category="sandbox"),
        make_tool("glob", "Find files by glob pattern.", GlobArgs, glob, category="sandbox"),
        make_tool("grep", "Search file contents with a regular expression (ripgrep).", GrepArgs, grep, category="sandbox"),
    ]
    return tools
