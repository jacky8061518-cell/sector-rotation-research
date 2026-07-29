import pandas as pd

from sector_rotation.taiwan import (
    assets_from_official_industries,
    assets_from_taiwan_themes,
    custom_taiwan_assets,
)


def sample_master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Code": ["2330", "5347", "2454"],
            "Yahoo ticker": ["2330.TW", "5347.TWO", "2454.TW"],
            "Name": ["台積電", "世界", "聯發科"],
            "Full name": ["台灣積體電路", "世界先進", "聯發科技"],
            "Industry code": ["24", "24", "24"],
            "Issued shares": [1000, 500, 800],
            "Market": ["上市", "上櫃", "上市"],
            "Industry": ["半導體業", "半導體業", "半導體業"],
        }
    )


def test_official_industry_assets_keep_twse_and_tpex_suffixes():
    assets = assets_from_official_industries(sample_master(), ["半導體業"])
    assert assets["2330.TW"].name == "台積電 (上市)"
    assert assets["5347.TWO"].name == "世界 (上櫃)"


def test_taiwan_theme_assets_resolve_codes_from_official_master():
    assets = assets_from_taiwan_themes(sample_master(), ["晶圓製造", "IC設計"])
    assert "2330.TW" in assets
    assert "5347.TWO" in assets
    assert "2454.TW" in assets


def test_custom_taiwan_codes_are_normalized():
    assets = custom_taiwan_assets(["2330", "5347.TWO"], sample_master())
    assert set(assets) == {"2330.TW", "5347.TWO"}
