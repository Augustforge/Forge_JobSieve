import pytest

from vacancy_match.embed import DEFAULT_MODEL, get_embedder


@pytest.mark.model
def test_real_model_puts_bot_developer_above_salesman():
    e = get_embedder(DEFAULT_MODEL)
    profile = e.encode(["Пишу ботов и скрипты автоматизации на Python, Telegram API"], "passage")[0]
    vacs = e.encode(["Разработчик Telegram-ботов на питоне", "Кассир в продуктовый магазин"], "query")
    bot, sales = vacs @ profile
    assert bot > sales
