from streamlit.testing.v1 import AppTest


def test_app_renders_with_offline_data():
    app = AppTest.from_file("app.py", default_timeout=30)
    app.run()
    app.radio[0].set_value("Offline demo").run()

    assert not app.exception
    assert app.title[0].value == "Sector Rotation Research Lab"
    assert any("Latest model signal" in item.value for item in app.subheader)
    assert len(app.metric) >= 5
