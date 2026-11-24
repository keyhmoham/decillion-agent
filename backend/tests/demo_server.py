"""Run the real API with a scripted "model" so the UI can be developed/screenshotted without a GPU.

    TEST_DATABASE_URL=postgresql+asyncpg://... python -m tests.demo_server --port 8010
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import uvicorn

from app.config import Settings
from app.main import create_app

from .helpers import call_text, turn
from .test_team import scripted as team_scripted

ANALYZE = '''import json, random
random.seed(7)
regions = ["North", "South", "East", "West"]
data = {r: [round(random.uniform(80, 140) * (1 + i * 0.03), 1) for i in range(6)] for r in regions}
print(json.dumps({r: round(sum(v), 1) for r, v in data.items()}))
'''


def demo_script(req, n):
    p = req.prompt
    if "<function=submit_plan>" in p or "The team finished working" in p or "Your task [" in p or "Write a compact plan" in p or "submit_verdict" in p:
        return team_scripted(req, n)
    k = p.rsplit("<|im_start|>user\n", 1)[-1].count("<tool_response>") if False else p.count("<tool_response>")
    steps = [
        turn("I'll plan the analysis first.", think="The user wants regional sales analysis with visuals. I should plan, write a script, run it, then present results as cards.",
             write_todos={"todos": [{"content": "Write analyze.py to generate and total the data", "status": "in_progress"}, {"content": "Run it and read the totals", "status": "pending"}, {"content": "Present stats, chart and table", "status": "pending"}]}),
        turn("Writing the analysis script.", think="Create the script first.", write_file={"path": "analyze.py", "content": ANALYZE}),
        turn("Running it.", think="Now execute and read the numbers.", bash={"command": "python3 analyze.py"}),
        turn("", think="Step 1 and 2 are done.", update_todo={"index": 2, "status": "completed"}),
        turn("Here are the headline numbers.", think="Show KPIs.", show_stats={"title": "Sales overview", "items": [
            {"label": "Total revenue", "value": "$2.87M", "delta": "+12.4%", "status": "good"}, {"label": "Best region", "value": "West", "delta": "+18%", "status": "good"},
            {"label": "Avg / month", "value": "$119.6k", "delta": "-2.1%", "status": "bad"}, {"label": "Records", "value": "200"}]}),
        turn("", think="Chart by region.", show_chart={"title": "Revenue by region (k$)", "type": "bar", "labels": ["North", "South", "East", "West"], "series": [{"name": "H1", "values": [655.2, 701.4, 688.9, 742.1]}, {"name": "H2", "values": [610.0, 690.3, 702.2, 799.5]}], "unit": "k$"}),
        turn("", think="Monthly trend.", show_chart={"title": "Monthly trend", "type": "line", "labels": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"], "series": [{"name": "Total", "values": [410, 432, 455, 449, 480, 503]}]}),
        turn("", think="Table.", show_table={"title": "Totals per region", "columns": ["Region", "Revenue (k$)", "Growth", "On target"], "rows": [["North", 1265.2, "+4.1%", True], ["South", 1391.7, "+6.0%", True], ["East", 1391.1, "+2.2%", False], ["West", 1541.6, "+18%", True]], "caption": "Targets: 1300k$ per region"}),
        turn("", think="A diagram of the pipeline.", show_diagram={"title": "Pipeline", "mermaid": "graph LR; A[Generate CSV]-->B[Aggregate]-->C[Chart]-->D[Report]"}),
        turn("", think="Summary callout.", show_callout={"title": "Key takeaway", "body": "**West** is pulling ahead while **East** misses its target.", "level": "success", "items": ["West +18% YoY", "East is 0.1% below target", "Recommend shifting budget to West"]}),
        turn("I analysed the sales data and presented the results above.\n\n- **West** leads with $1.54M\n- **East** is the only region below target\n\n```python\nprint('totals per region')\n```\n\nFiles: `analyze.py` in the workspace.", think="Wrap up."),
    ]
    return steps[min(k, len(steps) - 1)]


def main() -> None:
    from .conftest import ScriptedBackend

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    a = ap.parse_args()
    root = tempfile.mkdtemp(prefix="demo-sbx-")
    s = Settings(database_url=os.environ["TEST_DATABASE_URL"], sandbox_backend="local", sandbox_local_root=root, llm_base_url="http://demo", enable_web_tools=False, llm_model="demo")
    import app.main as main_mod

    main_mod.make_backend = lambda _s: ScriptedBackend(fn=demo_script, chunk=5)
    app = create_app(s)

    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
