"""Research universes ranging from broad sectors to thematic stock baskets."""

from __future__ import annotations

from dataclasses import dataclass

from .config import SECTOR_ETFS


@dataclass(frozen=True)
class AssetInfo:
    ticker: str
    name: str
    group: str


UNIVERSE_GROUPS: dict[str, dict[str, dict[str, str]]] = {
    "Broad sectors — ETFs": {
        "All 11 GICS sectors": SECTOR_ETFS,
    },
    "Detailed industries — ETFs": {
        "Technology": {
            "SOXX": "Semiconductors",
            "SMH": "Semiconductors",
            "XSD": "Equal-weight semiconductors",
            "IGV": "Software",
            "SKYY": "Cloud computing",
            "CIBR": "Cybersecurity",
            "FDN": "Internet companies",
        },
        "Financials": {
            "KRE": "Regional banks",
            "KBE": "Banks",
            "KIE": "Insurance",
            "IAI": "Broker-dealers and exchanges",
            "FINX": "FinTech",
        },
        "Health care": {
            "IBB": "Large-cap biotechnology",
            "XBI": "Equal-weight biotechnology",
            "IHI": "Medical devices",
            "IHF": "Health care providers",
            "IHE": "Pharmaceuticals",
            "ARKG": "Genomic innovation",
        },
        "Industrials and transport": {
            "ITA": "Aerospace and defense",
            "XAR": "Equal-weight aerospace and defense",
            "IYT": "Transportation",
            "JETS": "Airlines",
            "PAVE": "U.S. infrastructure",
            "XHB": "Homebuilders",
        },
        "Consumer": {
            "XRT": "Retail",
            "IBUY": "Online retail",
            "PEJ": "Leisure and entertainment",
            "PBJ": "Food and beverage",
        },
        "Energy": {
            "XOP": "Oil and gas exploration",
            "OIH": "Oil services",
            "AMLP": "Energy infrastructure and MLPs",
            "TAN": "Solar energy",
        },
        "Materials and mining": {
            "GDX": "Gold miners",
            "GDXJ": "Junior gold miners",
            "COPX": "Copper miners",
            "SLX": "Steel",
            "WOOD": "Timber and forestry",
        },
        "Real estate and infrastructure": {
            "VNQ": "U.S. REITs",
            "REZ": "Residential and specialized REITs",
            "REM": "Mortgage REITs",
            "SRVR": "Data center and digital infrastructure",
            "IGF": "Global infrastructure",
        },
    },
    "Structural themes — ETFs": {
        "AI and robotics": {
            "AIQ": "Artificial intelligence and technology",
            "BOTZ": "Robotics and artificial intelligence",
            "ROBO": "Robotics and automation",
            "IRBO": "Robotics and AI",
            "QTUM": "Quantum computing and machine learning",
        },
        "Cloud and cybersecurity": {
            "CLOU": "Cloud computing",
            "SKYY": "Cloud computing",
            "CIBR": "Cybersecurity",
            "HACK": "Cybersecurity",
            "BUG": "Cybersecurity",
        },
        "Clean energy and electrification": {
            "ICLN": "Global clean energy",
            "TAN": "Solar energy",
            "FAN": "Wind energy",
            "QCLN": "Clean energy",
            "GRID": "Smart grid infrastructure",
        },
        "Nuclear and uranium": {
            "URA": "Uranium and nuclear components",
            "URNM": "Uranium miners",
            "NLR": "Uranium and nuclear energy",
        },
        "Electric vehicles and batteries": {
            "DRIV": "Autonomous and electric vehicles",
            "IDRV": "Self-driving EV technology",
            "LIT": "Lithium and battery technology",
        },
        "Space and defense technology": {
            "ARKX": "Space and defense innovation",
            "UFO": "Space economy",
            "ITA": "Aerospace and defense",
            "SHLD": "Defense technology",
        },
        "Genomics and longevity": {
            "ARKG": "Genomic innovation",
            "GNOM": "Genomics and biotechnology",
            "IDNA": "Genomics, immunology and health care",
        },
        "FinTech and blockchain": {
            "FINX": "FinTech",
            "ARKF": "Blockchain and FinTech innovation",
            "BLOK": "Blockchain transformation",
            "BKCH": "Blockchain",
        },
        "Digital consumer": {
            "HERO": "Video games and esports",
            "ESPO": "Video gaming and esports",
            "SOCL": "Social media",
            "METV": "Metaverse and immersive technology",
        },
        "Resources and sustainability": {
            "PHO": "Water resources",
            "FIW": "Water",
            "MOO": "Agribusiness",
            "VEGI": "Global agriculture producers",
            "COPX": "Copper miners",
        },
    },
    "Styles and asset types — ETFs": {
        "Growth": {"VUG": "U.S. large-cap growth", "IWF": "Russell 1000 Growth", "QQQ": "Nasdaq-100"},
        "Value": {"VTV": "U.S. large-cap value", "IWD": "Russell 1000 Value", "RSP": "S&P 500 equal weight"},
        "Dividend": {"SCHD": "Dividend equity", "VYM": "High dividend yield", "DGRO": "Dividend growth"},
        "Quality": {"QUAL": "U.S. quality factor", "SPHQ": "S&P 500 quality"},
        "Low volatility": {"USMV": "Minimum volatility", "SPLV": "S&P 500 low volatility"},
        "Company size": {"IWM": "Russell 2000", "IJR": "S&P SmallCap 600", "MDY": "S&P MidCap 400"},
        "International": {"EFA": "Developed markets ex-U.S.", "EEM": "Emerging markets", "EWJ": "Japan"},
        "Real assets": {"GLD": "Gold", "SLV": "Silver", "DBC": "Broad commodities", "VNQ": "U.S. REITs"},
        "Rates and defensive": {"SHY": "1–3 year Treasuries", "IEF": "7–10 year Treasuries", "TLT": "20+ year Treasuries"},
    },
    "Thematic stocks — research baskets": {
        "AI compute": {
            "NVDA": "NVIDIA",
            "AMD": "Advanced Micro Devices",
            "AVGO": "Broadcom",
            "TSM": "Taiwan Semiconductor",
            "ASML": "ASML",
            "MU": "Micron Technology",
            "ARM": "Arm Holdings",
        },
        "Cloud platforms and software": {
            "MSFT": "Microsoft",
            "AMZN": "Amazon",
            "GOOGL": "Alphabet",
            "ORCL": "Oracle",
            "CRM": "Salesforce",
            "NOW": "ServiceNow",
            "SNOW": "Snowflake",
        },
        "Cybersecurity": {
            "PANW": "Palo Alto Networks",
            "CRWD": "CrowdStrike",
            "FTNT": "Fortinet",
            "ZS": "Zscaler",
            "OKTA": "Okta",
        },
        "Semiconductor equipment": {
            "ASML": "ASML",
            "AMAT": "Applied Materials",
            "LRCX": "Lam Research",
            "KLAC": "KLA",
            "TER": "Teradyne",
        },
        "Data centers and power": {
            "VRT": "Vertiv",
            "ETN": "Eaton",
            "CEG": "Constellation Energy",
            "VST": "Vistra",
            "DLR": "Digital Realty",
            "EQIX": "Equinix",
        },
        "Nuclear energy": {
            "CCJ": "Cameco",
            "CEG": "Constellation Energy",
            "VST": "Vistra",
            "LEU": "Centrus Energy",
            "BWXT": "BWX Technologies",
            "SMR": "NuScale Power",
            "OKLO": "Oklo",
        },
        "Space and defense": {
            "RKLB": "Rocket Lab",
            "LMT": "Lockheed Martin",
            "NOC": "Northrop Grumman",
            "RTX": "RTX",
            "GD": "General Dynamics",
            "PLTR": "Palantir",
        },
        "Electric vehicles and batteries": {
            "TSLA": "Tesla",
            "RIVN": "Rivian",
            "LI": "Li Auto",
            "NIO": "NIO",
            "ALB": "Albemarle",
            "QS": "QuantumScape",
        },
        "Metabolic health and GLP-1": {
            "LLY": "Eli Lilly",
            "NVO": "Novo Nordisk",
            "AMGN": "Amgen",
            "VKTX": "Viking Therapeutics",
        },
        "FinTech and exchanges": {
            "COIN": "Coinbase",
            "HOOD": "Robinhood",
            "SOFI": "SoFi",
            "AFRM": "Affirm",
            "CME": "CME Group",
            "ICE": "Intercontinental Exchange",
            "IBKR": "Interactive Brokers",
        },
        "Genomics and computational biology": {
            "CRSP": "CRISPR Therapeutics",
            "NTLA": "Intellia Therapeutics",
            "BEAM": "Beam Therapeutics",
            "RXRX": "Recursion Pharmaceuticals",
            "TWST": "Twist Bioscience",
        },
    },
}


def groups_for(universe_name: str) -> list[str]:
    return list(UNIVERSE_GROUPS[universe_name])


def assets_for(universe_name: str, groups: list[str]) -> dict[str, AssetInfo]:
    """Flatten selected groups while retaining a primary group for each ticker."""
    assets: dict[str, AssetInfo] = {}
    universe = UNIVERSE_GROUPS[universe_name]
    for group in groups:
        for ticker, name in universe[group].items():
            assets.setdefault(ticker, AssetInfo(ticker=ticker, name=name, group=group))
    return assets


def custom_assets(tickers: list[str]) -> dict[str, AssetInfo]:
    return {
        ticker: AssetInfo(ticker=ticker, name=ticker, group="Custom")
        for ticker in dict.fromkeys(tickers)
    }
