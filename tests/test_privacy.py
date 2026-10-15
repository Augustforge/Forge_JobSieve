import importlib.util
import subprocess
from pathlib import Path

spec = importlib.util.spec_from_file_location("privacy_check", Path(__file__).parents[1] / "scripts" / "privacy_check.py")
privacy_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(privacy_check)


def repo(tmp_path, files):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for name, text in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return tmp_path


def test_clean_repo_passes(tmp_path):
    root = repo(tmp_path, {"README.md": "Contact: dev@example.com", "src/a.py": "@dataclass\nclass A: pass\n"})
    assert privacy_check.check(root, []) == []


def test_flags_data_files_emails_phones_telegram(tmp_path):
    # literals are split so this test file itself passes the privacy check of the real repo
    email = "someone" + "@" + "gmail.com"
    phone = "+7 (495) " + "215-00-82"
    root = repo(tmp_path, {"data.jsonl": "{}", "x.db": "", "notes.md": f"mail me: {email}\n{phone}\n",
                           "c.md": "Tele" + "gram: @somebody\nhttps://t." + "me/somebody"})
    problems = "\n".join(privacy_check.check(root, []))
    for needle in ["data.jsonl", "x.db", f"email {email}", "phone", "telegram"]:
        assert needle in problems


def test_examples_may_hold_jsonl(tmp_path):
    root = repo(tmp_path, {"examples/vacancies.jsonl": "{}"})
    assert privacy_check.check(root, []) == []


def test_private_terms_reported_without_revealing_them(tmp_path):
    root = repo(tmp_path, {"README.md": "Built by Secretname Lastname"})
    problems = privacy_check.check(root, ["secretname"])
    assert problems == ["README.md: private term #1"]


def test_binary_and_unknown_files_fail(tmp_path):
    root = repo(tmp_path, {"README.md": "ok", "examples/notes.xlsx": "a,b", "tracker.csv": "a,b",
                           "examples/tracker.example.csv": "company,sent"})
    (root / "examples" / "resume.pdf").write_bytes(b"%PDF-1.4 \xff\xfe binary")
    problems = "\n".join(privacy_check.check(root, []))
    assert "examples/resume.pdf" in problems and "examples/notes.xlsx" in problems
    assert "tracker.csv: file type" in problems and "examples/tracker.example.csv" not in problems  # data only in examples/


def test_known_text_files_pass(tmp_path):
    root = repo(tmp_path, {"pyproject.toml": "x", "uv.lock": "x", ".gitignore": "x", "scripts/a.sh": "x",
                           ".github/workflows/ci.yml": "x", "examples/rules.example.yaml": "x", "LICENSE": "x"})
    assert privacy_check.check(root, []) == []


def test_staged_content_is_checked(tmp_path):
    root = repo(tmp_path, {"README.md": "Built by Secretname"})
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    (root / "README.md").write_text("clean now", encoding="utf-8")  # working copy fixed, index still dirty
    assert privacy_check.check(root, ["secretname"], staged=True) == ["README.md: private term #1"]


def test_cyrillic_names_and_deleted_files(tmp_path):
    root = repo(tmp_path, {"заметки.md": "ok", "gone.md": "x"})
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    (root / "gone.md").unlink()
    assert privacy_check.check(root, []) == []
