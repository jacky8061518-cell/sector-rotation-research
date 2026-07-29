import os

from streamlit.testing.v1 import AppTest


def test_app_renders_with_offline_data():
    os.environ["SECTOR_ROTATION_TEST_MODE"] = "1"
    app = AppTest.from_file("app.py", default_timeout=30)
    app.run()

    assert not app.exception
    assert app.title[0].value == "U.S. & Taiwan Multi-Layer Rotation Lab"
    assert any("最新模型訊號" in item.value for item in app.subheader)
    assert len(app.metric) >= 9
    del os.environ["SECTOR_ROTATION_TEST_MODE"]
