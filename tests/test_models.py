import pytest

from vacancy_match.models import Label, Vacancy, detect_lang, make_id, normalize_url


def test_normalize_url_strips_query_fragment_www_and_slash():
    assert normalize_url("https://www.Example.com/jobs/1/?utm=x#top") == "example.com/jobs/1"


def test_regional_hh_urls_share_id():
    a = make_id("https://krasnoyarsk.hh.ru/vacancy/100200300?hhtmFrom=vacancy")
    b = make_id("https://hh.kz/vacancy/100200300")
    c = make_id("hh.ru/vacancy/100200300")
    assert a == b == c
    assert len(a) == 16


def test_detect_lang():
    assert detect_lang("Python-разработчик, удалённо") == "ru"
    assert detect_lang("Python developer, remote") == "en"
    assert detect_lang("12345") == ""


def test_vacancy_from_dict_fills_derived_fields():
    v = Vacancy.from_dict(
        {"source": "hh", "url": "https://hh.kz/vacancy/1", "title": " Аналитик ", "text": "Ищем аналитика данных"},
        run_id="2026-10-14",
    )
    assert v.id == make_id("hh.ru/vacancy/1")
    assert v.title == "Аналитик"
    assert v.lang == "ru"
    assert v.run_id == "2026-10-14"
    assert v.collected_at
    assert v.embed_text() == "Аналитик\nИщем аналитика данных"


def test_vacancy_record_run_id_wins_over_default():
    v = Vacancy.from_dict({"source": "s", "url": "u.com/1", "title": "t", "run_id": "r-own"}, run_id="r-default")
    assert v.run_id == "r-own"


def test_vacancy_from_dict_requires_source_url_title():
    with pytest.raises(ValueError, match="title"):
        Vacancy.from_dict({"source": "hh", "url": "https://hh.kz/vacancy/1"})


def test_label_validates_values():
    Label(vacancy_id="x", label="fit", actor="human", reasons=("domain",))
    with pytest.raises(ValueError):
        Label(vacancy_id="x", label="maybe")
    with pytest.raises(ValueError):
        Label(vacancy_id="x", label="fit", actor="boss")
    with pytest.raises(ValueError, match="reason codes"):
        Label(vacancy_id="x", label="no_fit", reasons=("too far",))


def test_reason_codes_are_the_candidates_own():
    lab = Label(vacancy_id="x", label="no_fit", reasons=("onsite", "low_pay", "language2"))
    assert lab.reasons == ("onsite", "low_pay", "language2")


def test_id_like_query_params_survive_normalization():
    a = make_id("https://acme.com/careers?gh_jid=1&utm_source=x")
    b = make_id("https://acme.com/careers?gh_jid=2")
    assert a != b
    assert normalize_url("https://acme.com/careers?utm_source=x&gh_jid=1") == "acme.com/careers?gh_jid=1"


def test_run_date():
    from vacancy_match.models import run_date
    assert run_date("links-2026-10-12") == "2026-10-12"
    assert run_date("2026-10-14") == "2026-10-14"
    assert run_date("results-2026-08-10-backend") == "2026-08-10"
    assert run_date("history") == ""


def test_id_param_keys_are_case_insensitive():
    assert make_id("https://acme.com/job?ID=5") == make_id("https://acme.com/job?id=5")


def test_requirements_come_before_the_company_story():
    text = ("О компании\nМы молодая быстрорастущая компания с офисами в пяти странах и большими планами.\n\n"
            "Требования:\n- Python от 1 года\n- опыт с API\n\n"
            "Условия:\n- удалённо, оплата в евро\n\n"
            "О нас\nДружная команда, корпоративы.")
    v = Vacancy.from_dict({"source": "s", "url": "x.com/1", "title": "Python-разработчик", "text": text})
    e = v.embed_text()
    assert e.startswith("Python-разработчик\nТребования:")
    assert e.index("Условия:") < e.index("О компании") and e.index("О компании") < e.index("О нас")
    assert sorted(e.split()) == sorted(f"{v.title}\n{text}".split())  # nothing lost, only reordered


def test_text_without_headings_is_unchanged():
    v = Vacancy.from_dict({"source": "s", "url": "x.com/1", "title": "T", "text": "просто текст вакансии"})
    assert v.embed_text() == "T\nпросто текст вакансии"


def test_bullets_are_never_headings():
    text = ("Требования:\n- Python\n- Knowledge about REST APIs\n- Docker\n\n"
            "О компании\nМы студия.\n- Решать задачи клиентов вместе")
    e = Vacancy.from_dict({"source": "s", "url": "x.com/1", "title": "T", "text": text}).embed_text()
    assert e.index("Docker") < e.index("О компании")  # "about" in a bullet does not split the requirements
    assert e.index("Решать задачи") > e.index("О компании")  # a bullet about tasks stays with its block
