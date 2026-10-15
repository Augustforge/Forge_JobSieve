"""Write examples/vacancies.jsonl and examples/labels.jsonl: synthetic data for the README demo."""
import json
from pathlib import Path

FILLER = (" The team is remote-first and values clear written communication, ownership of tasks and steady"
          " delivery. Flexible schedule, monthly salary, paid time off.")
ROLES = [  # run, title, company, body, salary, label, track
    ("demo-2026-10-01", "Python Backend Developer", "Northwind Apps", "Build REST APIs in Python, write tests, deploy services with Docker and PostgreSQL.", "$2500 per month", "fit", "A-backend"),
    ("demo-2026-10-01", "Telegram Bot Developer", "Chatly", "Develop Telegram bots in Python that take orders and notify owners, with monitoring.", "$1800 per month", "fit", "A-backend"),
    ("demo-2026-10-01", "UI Designer", "Blue Pixel", "Design landing pages and mobile app screens in Figma and build a component library.", "$2000 per month", "fit", "B-design"),
    ("demo-2026-10-01", "Sales Manager", "Acme Retail", "Cold calls, client meetings, CRM pipeline and monthly sales targets by phone.", "$900 per month", "no_fit", ""),
    ("demo-2026-10-01", "Accountant", "Ledger & Co", "Bookkeeping, tax reports, payroll and reconciliation of bank statements.", "", "no_fit", ""),
    ("demo-2026-10-01", "Warehouse Operator", "Boxes Ltd", "Receive goods, pack orders, operate the scanner and keep the stock tidy.", "", "no_fit", ""),
    ("demo-2026-10-02", "Data Engineer", "Metricly", "Build data pipelines that pull from public APIs, clean records and feed dashboards with SQL.", "$3000 per month", "fit", "A-backend"),
    ("demo-2026-10-02", "Brand Designer", "Studio North", "Create brand systems, illustrations and landing page layouts in Figma.", "$1500 per month", "fit", "B-design"),
    ("demo-2026-10-02", "Automation Engineer", "OpsFlow", "Automate business processes with Python scripts, scheduled jobs and API integrations; our tools are open source.", "$2200 per month", "fit", "A-backend"),
    ("demo-2026-10-02", "Call Center Agent", "VoiceHub", "Answer inbound calls, follow scripts and log tickets, fluent German required.", "$700 per month", "no_fit", ""),
    ("demo-2026-10-02", "Marketing Specialist", "Growthy", "Run ad campaigns, write social posts and report weekly marketing results.", "", "no_fit", ""),
    ("demo-2026-10-02", "Remote Data Entry Assistant", "QuickStart Jobs", "Type documents from home and earn fast. Pay a registration fee for your starter kit before you begin.", "", "no_fit", ""),
    # Same skills, wrong terms: stage 1 alone ranks these next to the fits, stage 2 (rules) sends them down.
    ("demo-2026-10-02", "Python Developer (client-facing)", "CallStack", "Build REST APIs in Python and data pipelines for a Berlin agency; daily client meetings, fluent German required.", "$2600 per month", "no_fit", ""),
    ("demo-2026-10-02", "Junior Python Scripter", "PennyWorks", "Write Python scripts and small API integrations for internal tools.", "$600 per month", "no_fit", ""),
]


def main() -> None:
    here = Path(__file__).parent
    vac_lines, label_lines = [], []
    for n, (run, title, company, body, salary, label, track) in enumerate(ROLES, 1):
        url = f"https://jobs.example.com/{n}"
        vac_lines.append(json.dumps({"source": "example", "url": url, "title": title, "company": company,
                                     "text": body + FILLER, "salary_text": salary, "run_id": run}))
        label_lines.append(json.dumps({"url": url, "label": label, "track": track,
                                       "actor": "human" if n % 2 else "agent"}))
    (here / "vacancies.jsonl").write_text("\n".join(vac_lines) + "\n", encoding="utf-8")
    (here / "labels.jsonl").write_text("\n".join(label_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
