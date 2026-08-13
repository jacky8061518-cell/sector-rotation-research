from factors.registry import get_factor, list_factors


def test_library_auto_registers_price_factors() -> None:
    names = {spec.name for spec in list_factors(market="TW")}

    assert names == {"mom_12_1", "mom_6m", "vol_60d", "beta_252d", "max_ret_5d"}
    assert get_factor("vol_60d").spec.direction == -1


def test_market_filter_is_applied() -> None:
    assert all("US" in spec.markets for spec in list_factors(market="US"))
