# x265 アダプタ設計メモ（M-9L、取得ブロック）

## 状態

- `forks/x265/` は**未取得**。
- `git clone https://bitbucket.org/multicoreware/x265_git.git` は
  **外部ネットワーク取得の権限拒否**で実行できず。
- したがって以下は**未実測**であり、ソース確認前の設計案として扱う。

## 1. ビルド手順（取得後の予定）

x265 3.6 を想定:
```
cd forks/x265/source
cmake -G "MinGW Makefiles" \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DENABLE_ASSEMBLY=OFF -DENABLE_SHARED=OFF \
  -DCMAKE_BUILD_TYPE=Release ..
mingw32-make -j4
```
- 32bit gcc / nasm 無しを想定し、`ENABLE_ASSEMBLY=OFF`。
- `AR=ar rc ` が必要になる可能性。
- `x265.exe` が生成できたら `README_build_ja.md` に確定手順を記録。

## 2. `--analysis-save` / `--analysis-load`（未検証）

既知の x265 3.x 仕様:
- `--analysis-save <prefix>` はフレームごとのバイナリファイル
  `<prefix>_<poc>.bin` を書く（テキストではない）。
- `--analysis-load <prefix>` は同じ命名規則で読む。
- 内容は `x265_analysis_data`（picPar、slice type、depth、MV、
  intra/inter cost 等）で、**構造体レイアウトはビルド依存**。
- `--analysis-load` と `--analysis-save` を同一設定で往復すると
  ビットストリームは一致する、というのが期待値（未検証）。
- `--refine-inter 0` は解析 MV をそのまま使うモード。
  `refine-inter 1..3` は周辺探索を追加する。

### 外部生成ファイルの可否

判定: **b) 形式変換が必要**（推定）。
理由:
- テキストではないバイナリで、構造体のフィールド順・パディング・
  バージョン番号を持つ。
- GPU 生成の `(frame, mbx, mby, mvx, mvy)` をそのまま食わせるには、
  `x265_analysis_data` の該当フィールドを埋めたバイナリを
  書き出す変換器が必要。
- ファイル名も `prefix_<poc>.bin` に合わせる必要がある。

## 3. GPU が供給すべき値

- lowres MV（16x16 / 8x8）
- intra / inter cost
- `refine-inter` の対象 MB
- slice type（I/P/B）は x265 側の判定を尊重

消費点:
- `--analysis-load` の `x265_analysis_data`
- `--refine-inter` の MV 初期値

## 4. ビット一致

- `--refine-inter 0` では解析 MV がそのまま使われるため、
  MV が 1 quarter-pel でも違えばビットストリームは変わる。
- x264 版と同様、**ビット一致は必須**。
- x265 の analysis はバージョン・ビルド依存なので、
  同一バイナリで save/load の往復一致を先に確認する必要がある。

## 5. 実現案

- 1 フレーム 1 起動 + 1 同期。
- GPU 出力: MB ごとの MV（int16×2）＋ cost（int32）。
  - 1080p: 8,160 MB × 8 B = 65 KB/frame。
  - 640p: 920 MB × 8 B = 7.4 KB/frame。
- 変換器: GPU バイナリ → `x265_analysis_data` バイナリ。
- 工数: 取得・ビルド 2 日、形式解析 3–5 日、変換器 3–4 日、
  一致検証 3 日、合計 **11–14 日**。

## 6. x264 版との比較

- x264 版は 640p slow/limit:4 で **4/4・BD-rate -0.12%** を実測済み。
- x265 版は**未取得・未ビルド・未検証**。
- x265 の `--analysis-load` が外部ファイルを受け入れるなら
  同程度の見込みはあるが、**現時点では数値的根拠なし**。
- 形式変換が必要なため、x264 版より工数が大きい。

## 7. 判断

- 現状は **x265 アダプタに進む根拠が未確定**。
- 先に `forks/x265` の取得許可が必要。
- 取得できない場合、x264 版の 4/4 を維持し、
  medium 向けの別施策（先読み GPU 化は保留）を検討する。
