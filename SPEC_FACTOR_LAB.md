# Factor Research Lab 建置規格書

給 AI Agent 的執行用 prompt。建議存成 repo 根目錄的 `SPEC_FACTOR_LAB.md`，並在每次開新 session 時要求 agent 先讀這份文件。

---

## 0. 開場訊息（直接貼給 agent）

> 你要在我現有的 repo 上新增一個因子研究模組。請先完整讀 `SPEC_FACTOR_LAB.md`，然後執行第 14 節的 Phase 1。
>
> 硬性規則有三條。第一，不要動 `src/sector_rotation/` 既有模組的公開介面。第二，任何因子計算都不得取用 asof 之後的資料，這點必須用測試證明。第三，在寫任何實作之前，先把第 17 節要你回報的三件事回報給我，等我確認後才動手。
>
> 我要的是可審查的研究工具，不是能跑就好的 demo。回測結果如果太漂亮，優先懷疑資料而不是慶祝。

---

## 1. 角色設定

- 你是一位同時具備量化研究與 Python 工程能力的開發者
- 你的產出會被用來做真實的投資決策參考，因此方法論正確性優先於功能數量
- 遇到方法論上的取捨（例如中性化方式、成本假設）時，先提出兩到三個選項與各自代價，讓我決定，不要自己選一個然後往下寫
- 不確定既有程式碼行為時，先讀原始碼再動作，不要憑檔名猜測

---

## 2. 專案目標

- 建一個橫斷面因子研究工作台，涵蓋從原始資料到因子建構、因子評估、多空組合回測的完整流程
- 核心設計要求是因子邏輯與應用層完全解耦，新增一個因子只需在因子庫新增一個檔案，不需要改動任何評估模組或前端
- 服務對象是研究員，不是交易員。輸出重點在為什麼這個訊號有效與在哪些條件下失效，不是最大化回測績效
- 支援美股與台股雙市場，兩邊共用同一套因子介面與評估邏輯

---

## 3. 既有環境（必須沿用）

- Streamlit 應用，部署於 Streamlit Community Cloud
- 既有 Python 套件 `src/sector_rotation/`，內含 `config`、`data`、`fund_flow`、`holdings`、`metrics`、`rrg`、`snapshots`、`strategy`
- 進入點 `app.py`，排程腳本 `scripts/daily_update.py`，測試以 pytest 撰寫
- 資料源為 Yahoo Finance 價格、TWSE 與 TPEx 公司主檔與三大法人買賣超
- 資料以 Parquet 儲存，美股與台股分離
- 基準與防禦資產設定，美股為 SPY 與 SHY，台股為 0050.TW 與 00679B.TWO
- 自動更新排程於平日美東時間 18:00 執行

新模組必須重用既有的資料讀取與快取層，不要另建一套平行的資料下載邏輯。若既有介面不足，用擴充而非取代的方式處理。

---

## 4. 新增目錄結構

```
src/
  factors/
    __init__.py
    spec.py            # FactorSpec 與 Factor protocol
    registry.py        # 註冊與查詢機制
    context.py         # DataContext，唯一的資料存取入口
    pipeline.py        # winsorize、標準化、中性化、缺值處理
    evaluation.py      # IC、分位數、衰減、穩定性
    portfolio.py       # 權重建構與限制
    backtest.py        # 再平衡迴圈、成本、績效指標
    costs.py           # 交易成本模型
    library/
      __init__.py      # 匯入所有因子檔案以觸發註冊
      momentum.py
      risk.py
      liquidity.py
      flow_tw.py
      revenue_tw.py
      value.py         # Phase 4
      quality.py       # Phase 4
pages/                 # 或既有的多頁機制，依 app.py 現況決定
tests/
  factors/
```

---

## 5. 資料層規格

### 5.1 已可使用

- 日頻價格，需為除權息與分割調整後，並保留未調整收盤價供成交金額計算
- 台股與美股股票主檔，含產業分類與上市市場
- 台股三大法人每日買賣超金額與股數

### 5.2 需新增

- 台股月營收，含年增率與月增率
- 市值與流通股數的歷史序列，用於市值中性化與規模因子
- 財報基本面，Phase 4 才需要
- 每日股票池快照，從建置日起每天存一份當日有效清單

### 5.3 資料源決策（需要我確認）

- 台股月營收，候選為公開資訊觀測站與 FinMind。關鍵不是取得資料，是取得公布日。上月營收於次月十日前公布，因子的 point-in-time 標記必須用實際公布日，用營收月份會造成十天左右的前視偏誤
- 財報基本面，優先順序為 TEJ、FinMind、yfinance。yfinance 的財報沒有公布日且會被回溯修改，只可用於探索性分析，不得用於任何正式回測結論
- 存活者偏誤，TWSE 與 TPEx 只提供現存清單。歷史上已下市的公司無法回補，這是本專案的已知限制，必須在應用介面明確揭露，不要假裝不存在。從現在開始每日快照，讓未來的回測逐步乾淨

---

## 6. 因子層架構

### 6.1 因子規格

```python
from dataclasses import dataclass
from typing import Protocol, Literal
import pandas as pd

Category = Literal["momentum", "value", "quality", "growth", "risk", "liquidity", "flow"]

@dataclass(frozen=True)
class FactorSpec:
    name: str                      # 唯一鍵，例如 "mom_12_1"
    label: str                     # 前端顯示名稱
    category: Category
    direction: int                 # +1 數值越大越看多，-1 相反
    lookback_days: int             # 計算所需最少歷史天數
    requires: tuple[str, ...]      # 依賴的資料表，例如 ("prices", "inst_flow")
    markets: tuple[str, ...]       # ("US", "TW") 或單一市場
    description: str               # 一句話說明經濟直覺
    reference: str | None = None   # 文獻或來源

class Factor(Protocol):
    spec: FactorSpec
    def compute(self, ctx: "DataContext", asof: pd.Timestamp) -> pd.Series:
        """回傳以 ticker 為 index 的原始因子值，未標準化。缺值以 NaN 保留。"""
```

### 6.2 註冊機制

- 以 decorator 註冊，`@register_factor`
- `registry.list_factors(market=..., category=...)` 回傳可用因子清單
- `library/__init__.py` 負責匯入所有因子檔案以觸發註冊
- 前端與評估模組只能透過 registry 取得因子，不得 import 任何具體因子類別

### 6.3 DataContext 是防前視偏誤的關鍵

- 所有資料存取一律經由 `DataContext`，因子內部禁止直接讀檔或呼叫網路
- `DataContext` 建構時綁定 asof 日期，內部對每張表做硬性時間截斷，讓因子作者在技術上不可能取得未來資料
- 有公布日概念的資料表（營收、財報），截斷條件用公布日而非期間結束日
- 提供的方法至少包含 `prices(fields, window)`、`market_cap(window)`、`inst_flow(window)`、`revenue(window)`、`industry_map()`、`universe()`

### 6.4 第一批因子（只需價量與法人資料）

- `mom_12_1` 十二個月動能剔除最近一個月，direction +1
- `mom_6m` 六個月動能，+1
- `resid_mom_12m` 對市場與產業回歸後的殘差動能，+1
- `vol_60d` 六十日已實現波動，-1
- `beta_252d` 對市場基準的 beta，-1
- `max_ret_5d` 近五日最大單日報酬，捕捉彩券效應，-1
- `turnover_20d` 二十日日均成交金額除以市值，-1，同時作為流動性篩選欄位
- `size_ln_mcap` 市值自然對數，-1
- `flow_inst_20d` 三大法人二十日累計買超金額佔成交金額比，台股專屬，+1
- `flow_foreign_persist` 外資連續買超天數，台股專屬，+1
- `rev_yoy` 月營收年增率，台股專屬，+1
- `rev_mom_3m` 三個月營收累計動能，台股專屬，+1

### 6.5 第二批因子（Phase 4，需財報）

- 價值 `ep`、`bp`、`fcf_yield`、`ev_ebitda_inv`
- 品質 `roe`、`gross_profitability`（毛利除以總資產）、`accruals`（-1）、`debt_to_equity`（-1）
- 成長 `eps_growth_yoy`、`sales_growth_yoy`

---

## 7. 前處理管線

依序執行，每一步都可在 config 開關

- 覆蓋率檢查，單一橫斷面若有效值低於門檻（預設 60%）則該日不產生訊號並記錄
- 極值處理，預設 winsorize 至 1% 與 99% 分位，可選 MAD 法
- 標準化，預設橫斷面 z-score，可選 rank 轉常態
- 中性化，以產業虛擬變數與市值對數做橫斷面迴歸取殘差，兩者可獨立開關
- 缺值處理，禁止用平均值或前值填補，缺值即排除該股當期，並記錄排除數量
- 方向調整，統一乘上 `spec.direction`，讓所有因子的高分等於看多

---

## 8. 因子評估層

- IC 採用 Spearman rank correlation，計算因子分數對未來報酬的相關性
- 報告項目為 IC 平均、IC 標準差、IC IR（平均除以標準差）、Newey-West 調整後 t 值、正 IC 比率
- 多期間評估，未來 5、20、60 個交易日，用於觀察因子衰減曲線
- 分位數分析，預設五分位，輸出各分位年化報酬、累積報酬曲線、單調性檢定
- 多空組合，最高分位減最低分位，輸出年化報酬、年化波動、Sharpe、最大回撤、月勝率
- 換手率，逐期計算並換算成本後淨績效
- 穩定性切割，分市場、分產業、分市值三分位重算 IC，用於判斷因子是否只在小型股或單一產業有效
- 因子間相關矩陣與 IC 相關矩陣，供合成階段去重

---

## 9. 組合建構與回測

- 再平衡頻率，月頻為預設，可選週頻與雙週頻
- 訊號到成交的時序，asof 日收盤產生訊號，下一個交易日收盤成交，不得同日成交
- 權重方式，等權、因子分數加權、逆波動加權
- 約束條件，單股權重上限、最少持股檔數、產業權重上限、日均成交金額下限
- 多因子合成，標準化後等權加總為預設，可選 IC 加權，並提供合成前後的相關性檢查
- 回測輸出，淨值曲線、逐期持股明細、逐期換手、成本明細，全部可下載為 CSV

---

## 10. 交易成本假設

- 台股，手續費 0.1425% 單邊並可設券商折扣，證券交易稅 0.3% 僅賣出，滑價預設 0.1%
- 美股，手續費預設 0，滑價以價差百分比估計，預設 0.05%
- 所有數值放 config，不得寫死在計算邏輯內
- 回測必須同時輸出無成本與含成本兩條淨值，方便判斷訊號是否只是被成本吃掉

---

## 11. Streamlit 前端規格

- 側邊欄沿用既有的市場切換（美股與台股），不要另做一套
- 頁面一 Factor Explorer，單因子診斷，含 IC 時序圖、分位數累積報酬、因子值分布、產業暴露、覆蓋率時序
- 頁面二 Factor Zoo，所有因子的 IC IR 排序表與相關矩陣熱圖
- 頁面三 Portfolio Builder，選因子、設權重與約束、跑回測、看績效
- 頁面四 Cross-section Snapshot，當日全市場因子排名表，可篩選產業並下載
- 圖表統一用 plotly，配色與既有頁面一致
- 重運算一律加快取，冷啟動不得超過十秒

---

## 12. 前視偏誤檢查清單

每一項都要有對應的 pytest 測試

- 財報與月營收使用實際公布日而非期間結束日
- 因子計算只取用 asof 當日收盤及以前的資料
- 交易發生在訊號日之後，不得同日成交
- 除權息與分割調整後價格與報酬序列一致
- 產業分類使用當期而非最新版本，若無歷史版本須標註為已知限制
- 股票池使用當期有效清單，若使用現存清單須在介面揭露存活者偏誤
- 標準化與中性化只在單一橫斷面內進行，不得跨期使用全樣本統計量

---

## 13. 工程規範

- Python 3.11 以上，全面型別註記，ruff 與 mypy 需通過
- 計算函式純函式化，不在函式內讀檔或改動全域狀態
- 允許的依賴為 pandas、numpy、scipy、statsmodels、pyarrow、plotly、streamlit，其他一律先問過
- 效能目標，台股全市場約一千八百檔加標普五百成分股，單因子全歷史計算在數十秒內完成，禁止使用 iterrows 做橫斷面運算
- 測試至少涵蓋每個因子的小樣本已知答案測試、DataContext 時間截斷測試、回測引擎的零成本恆等測試、管線各步驟的單元測試
- 每個 Phase 結束時更新 README 的中文說明

---

## 14. 交付階段

- Phase 1 骨架。`spec.py`、`registry.py`、`context.py`、`pipeline.py`、五個價量因子、IC 評估、Factor Explorer 頁面
- Phase 2 回測。分位數與多空回測、成本模型、績效指標、Portfolio Builder 頁面
- Phase 3 台股特色因子。月營收與三大法人資金流因子，並與既有 `fund_flow` 模組整合，讓產業層資金流與個股層因子能互相對照
- Phase 4 財報因子。接入確認後的財報資料源，補上價值與品質因子
- Phase 5 研究輸出。Factor Zoo、每日快照自動產生、併入 `scripts/daily_update.py` 排程

每個 Phase 結束後停下來，回報完成內容、已知問題與下一階段計畫，等我確認再繼續。

---

## 15. 驗收標準

- 新增一個因子只需在 `library/` 新增一個檔案，不需修改 `app.py` 或任何評估模組
- 十二減一動能在美股樣本上應呈現正 IC 與多空正報酬，若結果相反，先檢查報酬對齊與方向設定，不要直接接受
- 低波與低 beta 因子在方向設為 -1 後應為正 IC，若相反須明確報告並附上診斷
- 任一因子的 IC IR 若高於 1.0，視為疑似資料問題，必須主動查核並回報，不得直接呈現
- 所有測試通過，mypy 無錯誤，Streamlit 可正常部署

---

## 16. 明確禁止事項

- 不得重寫或破壞 `src/sector_rotation/` 既有模組的公開介面
- 不得用 yfinance 財報資料產生正式回測結論
- 不得在因子內部直接讀檔或呼叫網路
- 不得為了績效好看而調整樣本期間、排除個股或更換基準，任何篩選都必須寫在 config 並可追溯
- 不得以平均值、中位數或前值填補缺值
- 不得產生任何合成或模擬資料充當真實資料
- 不得在沒有測試的情況下宣稱某個 Phase 完成

---

## 17. 開始前必須回報的三件事

1. 現有 Parquet 檔案的實際 schema，列出每張表的欄位名稱、型別與日期欄位語意，並指出哪些欄位可直接支援因子計算、哪些缺
2. 財報與月營收資料源的建議選擇，說明各選項在公布日完整度、歷史長度與取得成本上的差異
3. Phase 1 的完整檔案清單與主要函式簽名，包含 `DataContext` 的方法設計，等我確認後才開始寫實作

回報時不要附上任何實作程式碼，只給設計。
