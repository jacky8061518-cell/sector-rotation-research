import os

from streamlit.testing.v1 import AppTest


def test_app_renders_lightweight_broker_page():
    os.environ["SECTOR_ROTATION_TEST_MODE"] = "1"
    app = AppTest.from_file("app.py", default_timeout=30)
    app.run()

    assert not app.exception
    assert app.title[0].value == "台股資金流與券商分點研究系統"
    assert app.segmented_control[0].options == [
        "資金流與輪動回測",
        "券商分點日週月",
        "因子研究 Phase 1",
    ]
    assert any("當日法人流入前 10 檔" in item.value for item in app.markdown)
    assert len(app.metric) >= 4
    del os.environ["SECTOR_ROTATION_TEST_MODE"]


def test_app_renders_factor_explorer_page():
    os.environ["SECTOR_ROTATION_TEST_MODE"] = "1"
    app = AppTest.from_file("app.py", default_timeout=60)
    app.run()
    app.segmented_control[0].set_value("因子研究 Phase 1").run(timeout=60)

    assert not app.exception
    assert any("Factor Explorer" in item.value for item in app.subheader)
    assert any("存活者偏誤" in item.value for item in app.warning)
    assert len(app.dataframe) >= 1
    del os.environ["SECTOR_ROTATION_TEST_MODE"]
