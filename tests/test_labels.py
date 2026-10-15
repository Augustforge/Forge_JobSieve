import pytest

from vacancy_match.labels import label_from_dict, load_label_file
from vacancy_match.models import make_id


def test_label_from_url_and_reason_string():
    lab = label_from_dict({"url": "https://hh.kz/vacancy/5", "label": "no_fit", "reasons": "language, geo",
                           "actor": "human"})
    assert lab.vacancy_id == make_id("hh.ru/vacancy/5")
    assert lab.reasons == ("language", "geo")


def test_label_needs_an_identifier():
    with pytest.raises(ValueError, match="vacancy_id or url"):
        label_from_dict({"label": "fit"})


def test_load_label_file_reports_line_number(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text('{"vacancy_id": "a", "label": "fit"}\n\n{"vacancy_id": "b", "label": "meh"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"l\.jsonl:3"):
        load_label_file(p)


def test_load_label_file_ok(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text('{"vacancy_id": "a", "label": "fit", "track": "C"}\n{"url": "x.com/1", "label": "no_fit"}\n',
                 encoding="utf-8")
    labels = load_label_file(p)
    assert [l.label for l in labels] == ["fit", "no_fit"] and labels[0].track == "C"


def test_non_object_line_names_the_line(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text('{"vacancy_id": "a", "label": "fit"}\n[]\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"l\.jsonl:2"):
        load_label_file(p)


def test_bom_is_accepted(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text('\ufeff{"vacancy_id": "a", "label": "fit"}\n', encoding="utf-8")
    assert load_label_file(p)[0].vacancy_id == "a"
