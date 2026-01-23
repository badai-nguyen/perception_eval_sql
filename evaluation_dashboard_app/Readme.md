# 評価ダッシュボード

## 必須インストール

本ダッシュボード・評価ツールの動作には、以下の前提と Python パッケージが必要です。  


### Python パッケージ（基本機能）
```sh
pip install \
  streamlit pandas plotly duckdb numpy \
  requests pyyaml matplotlib shapely
```

### Python パッケージ（ダウンロード機能）
```sh
# Install authentication library (Download/Scenario API)
pip install git+https://github.com/tier4/webauto-auth-py.git
```

```sh
# Install CLI tool (評価実行コマンド生成で利用する場合)
pipx install git+ssh://git@github.com/tier4/v_and_v_util.git
```

### pilot-auto / perception_eval（Summary/Score 生成時のみ）
- `perception_eval` が使える pilot-auto 環境が必要です（下記「使い方」参照）
- `perception_eval` の import が失敗すると `Summary.csv` / `Score.csv` 生成が停止します

### 設定ファイル
- `configs/autoware_evaluator_dl_config.json` に入力値を保存します（自動生成/更新）

## 概要
Streamlit で動作する評価ダッシュボードです。`data/` 配下の評価結果（`Summary.csv`、`Score.csv`、`.parquet`）を読み込み、複数ページで可視化できます。さらに、`pages/6_Download.py` では評価結果（`result.txt` など）の一括集計や `Summary.csv` / `Score.csv` の自動生成、結果ディレクトリの検索・ダウンロード管理も可能です。

## 主な機能
- 概要ページで Run の選択、単体/比較モードの切替、全体指標を表示
- TP/位置/速度の統計ビューア（散布図・分布）
- Criteria-based の評価ビューア（指標分布・平均・箱ひげ）
- 検出統計の比較ビューア（TP/FP の距離ビン比較など）
- BEV のバウンディングボックス可視化
- 評価実行コマンドの生成ツール

## ディレクトリ構成
```
evaluation_dashboard_app/
  Overview.py
  pages/
    1_TP_Summary.py
    2_Criteria_Based_Score.py
    3_Detection_Stats.py
    4_Bounding_Box_Viewer.py
    5_Tools.py
    6_Download.py
  lib/
    
  configs/
    autoware_evaluator_dl_config.json
  data/
    <run_id>/
      Summary.csv
      Score.csv
    *.parquet
```

## 使い方

1. サマリーやスコア生成（`pages/6_Download.py` の「Summary.csv / Score.csv を生成」）を実行するには、**事前に下記コマンドで pilot-auto（ROS 2）環境を有効化する必要があります**:
   ```
   source path_to_pilot/install/setup.zsh
   ```
   ※ この作業は `pages/6_Download.py` の「Summary/Score CSV 生成」で必要です。

2. `evaluation_dashboard_app/` で Streamlit を起動します。
   ```
   streamlit run Overview.py
   ```

3. サイドバーからページやフィルタを選択して可視化します。

## ページ説明
### `Overview.py`
- 全体のエントリーポイント
- Single/Compare モードの切替
- Perception/Product ラベルの共通フィルタ
- 各ページに渡す `st.session_state` を構築

### `pages/1_TP_Summary.py`
- TP/位置/速度の統計ビューア
- `TP` の範囲フィルタ、速度の外れ値クリップ
- 散布図（`xrms` vs `yrms` / `vx` vs `vy`）と分布ヒストグラム

### `pages/2_Criteria_Based_Score.py`
- Criteria-based 評価ビューア
- Criteria を選択して指標分布・平均・箱ひげを表示

### `pages/3_Detection_Stats.py`
- `.parquet` を DuckDB で集計し、TP/FP などの距離ビン比較を可視化
- 検出対象/トピック/ラベル/visibility などのフィルタを提供

### `pages/4_Bounding_Box_Viewer.py`
- `.parquet` の BEV バウンディングボックス表示
- t4dataset/topic/label/visibility で絞り込み

### `pages/5_Tools.py`
- 評価実行コマンド生成ツール
- Report/Suite URL から Job ID / Suite ID を抽出

### `pages/6_Download.py`
- Evaluator の結果ダウンロードと評価実行
- `result.txt` / `score.json` から `Summary.csv` / `Score.csv` を生成

## データ形式（概略）
- `Summary.csv`: `id`, `TP`, `xstd`, `xrms`, `ystd`, `yrms`, `vx`, `vy`, `perception_label`, `product_label`
- `Score.csv`: Criteria ごとの評価指標ブロック（`Scenario`, `Option`, `GT_OBJ`, 以降は criteria0..n）
- `.parquet`: 検出統計/BB 表示に必要な `x`, `y`, `length`, `width`, `yaw`, `label`, `source`, `status` など

## 補足
- 最初に `Overview` を開いて Run を読み込む必要があります（各ページは `st.session_state` 前提）。
- 画像は本ダッシュボードの表示例です。