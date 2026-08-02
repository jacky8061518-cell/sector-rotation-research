import os

from streamlit.testing.v1 import AppTest


def test_app_renders_with_offline_data():
    os.environ["SECTOR_ROTATION_TEST_MODE"] = "1"
    app = AppTest.from_file("app.py", default_timeout=30)
    app.run()

    assert not app.exception
    assert app.title[0].value == "Institutional Fund Flow & Rotation Research Lab"
    assert any("最新模型訊號" in item.value for item in app.subheader)
    assert any("哪些產業正在轉強" in item.value for item in app.markdown)
    assert any(
        button.label == "下載產業輪動與原因 CSV"
        for button in app.download_button
    )
    assert len(app.metric) >= 9
    del os.environ["SECTOR_ROTATION_TEST_MODE"]
