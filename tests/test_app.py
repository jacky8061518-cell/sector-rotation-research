import os

from streamlit.testing.v1 import AppTest


def test_app_renders_lightweight_broker_page():
    os.environ["SECTOR_ROTATION_TEST_MODE"] = "1"
    app = AppTest.from_file("app.py", default_timeout=30)
    app.run()

    assert not app.exception
    assert app.title[0].value == "台股資金流與券商分點研究系統"
    assert any("當週法人流入前 10 檔" in item.value for item in app.markdown)
    assert len(app.metric) >= 4
    del os.environ["SECTOR_ROTATION_TEST_MODE"]
