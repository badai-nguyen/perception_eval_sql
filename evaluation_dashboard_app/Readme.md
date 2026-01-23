# 評価ダッシュボード

## 概要
Streamlit で動作する評価ダッシュボードです。`data/` 配下の評価結果（`Summary.csv` / `Score.csv` / `.parquet`）を読み込み、複数のページで可視化します。画像は現在の UI イメージです。

## 主な機能
- 概要ページで Run の選択、単体/比較モードの切替、全体指標を表示
- TP/位置/速度の統計ビューア（散布図・分布）
- Criteria-based の評価ビューア（指標分布・平均・箱ひげ）
- 検出統計の比較ビューア（TP/FP の距離ビン比較など）
- BEV のバウンディングボックス可視化
- 評価実行コマンドの生成ツール

## ディレクトリ構成
```
script/
  Overview.py
  pages/
    1_TP_Summary.py
    2_Criteria_Based_Score.py
    3_Detection_Stats.py
    4_Bounding_Box_Viewer.py
    5_Tools.py
    6_Download.py
  lib/
    run_loader.py
  data/
    <run_id>/
      Summary.csv
      Score.csv
    *.parquet
```

## 使い方
1. `data/` に評価結果を配置します。
   - `Summary.csv` / `Score.csv` を `data/<run_id>/` に配置
   - 検出統計/BB ビュー用の `.parquet` を `data/` 直下に配置
2. Streamlit を起動します。
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

## ライブラリ
### `lib/run_loader.py`
- `data/<run_id>/Summary.csv` と `Score.csv` を読み込むローダ
- Overview から利用され、各ページに渡される

## データ形式（概略）
- `Summary.csv`: `id`, `TP`, `xstd`, `xrms`, `ystd`, `yrms`, `vx`, `vy`, `perception_label`, `product_label`
- `Score.csv`: Criteria ごとの評価指標ブロック（`Scenario`, `Option`, `GT_OBJ`, 以降は criteria0..n）
- `.parquet`: 検出統計/BB 表示に必要な `x`, `y`, `length`, `width`, `yaw`, `label`, `source`, `status` など

## 補足
- 最初に `Overview` を開いて Run を読み込む必要があります（各ページは `st.session_state` 前提）。
- 画像は本ダッシュボードの表示例です。