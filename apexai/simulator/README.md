# ApexAI Real-time Inference Simulator

リアルタイム推論シミュレーター用Streamlitアプリケーション

## 概要

レーステレメトリデータ（CSV）を10Hzでストリーミングし、深層学習モデルによる10秒ごとの推論をシミュレートします。

### 主な機能

- **データストリーミング**: 10Hz（0.1秒間隔）でCSVデータを読み込み
- **スライディングウィンドウ**: 100サンプル（10秒分）をバッファリング
- **リアルタイム推論**: 10秒ごとにドライバー分類を実行
- **Test Time Augmentation (TTA)**: 6推論分（60秒）の最頻値投票
- **Min-Max正規化**: [0, 1]範囲へのクリッピング正規化

## アーキテクチャ

```
simulator/
├── backend/
│   ├── __init__.py
│   └── engine.py          # コアロジック
│       ├── DataStreamer       # CSVストリーミング (10Hz)
│       ├── SlidingWindowBuffer # バッファ管理 (seq_len=100)
│       ├── InferenceEngine    # 推論エンジン (正規化 + モデル)
│       └── TestTimeAugmentation # TTA投票
├── frontend/
│   ├── __init__.py
│   └── app.py             # Streamlit UI
├── model_repository/
│   └── prodmodel.yaml     # 本番モデル設定
├── requirements.txt
└── README.md
```

## 使用方法

### 1. 依存パッケージインストール

```bash
cd apexai/simulator
pip install -r requirements.txt
```

### 2. モデル準備

`model_repository/prodmodel.yaml`に本番モデルの設定を記載：

```yaml
model_ID: m-xxxxxxxx
model_path: path/to/model.pth
name: "Transformer"
type: "Transformer"

preprocessing:
  features:
    - "pbrake_f"
    - "pbrake_r"
    - "Steering_Angle"
    - "accx_can"
    - "accy_can"
    - "ath"
    - "gear"
    - "nmot"
    - "speed"

  normalization:
    enabled: true
    method: "minmax"

    min_max:
      min: [...]  # 9次元の最小値
      max: [...]  # 9次元の最大値
```

### 3. Streamlitアプリ起動

```bash
cd apexai/simulator
streamlit run frontend/app.py
```

### 4. シミュレーション実行

1. **CSVアップロード**: テレメトリデータをアップロード
2. **初期化**: "Initialize Simulator"をクリック
3. **シミュレーション開始**: "Start Simulation"をクリック
4. **リアルタイム確認**: 推論結果とTTA結果を確認

## データ要件

### CSVフォーマット

| 列名 | 説明 |
|------|------|
| pbrake_f | 前輪ブレーキ圧 |
| pbrake_r | 後輪ブレーキ圧 |
| Steering_Angle | ステアリング角度 |
| accx_can | X軸加速度 |
| accy_can | Y軸加速度 |
| ath | スロットル開度 |
| gear | ギアポジション |
| nmot | エンジン回転数 |
| speed | 車速 |

### サンプリング仕様

- **サンプリング周波数**: 10Hz（0.1秒間隔）
- **シーケンス長**: 100サンプル（10秒分）
- **推論間隔**: 10秒（100サンプル取得後）
- **TTA期間**: 60秒（6推論分）

## 技術詳細

### バッファ戦略

**独立ウィンドウアプローチ**: 推論後にバッファをクリア

```
[10秒] → 推論1 → [クリア] → [10秒] → 推論2 → [クリア] → ...
```

**利点**:
- 統計的独立性の確保
- データ汚染の防止
- シンプルな実装

**トレードオフ**:
- 推論中の約0.5秒データロス（許容範囲）

### 正規化処理

**Min-Max正規化 [0, 1]**:

```python
normalized = (x - min) / (max - min + 1e-8)
normalized = clamp(normalized, 0.0, 1.0)  # [0, 1]にクリッピング
```

### Test Time Augmentation (TTA)

**最頻値投票**:

```
推論結果: [Class 5, Class 5, Class 3, Class 5, Class 3, Class 5]
投票数: {5: 4, 3: 2}
最終予測: Class 5
```

## パフォーマンス

### 推論タイミング

```
0秒           10秒          20秒          30秒
├─────────────┼─────────────┼─────────────┼
│ バッファ蓄積 │ 推論1        │ バッファ蓄積 │ 推論2
│ (100サンプル)│ (~0.5秒)     │ (100サンプル)│ (~0.5秒)
└─────────────┴─────────────┴─────────────┴
```

### デバイス対応

- **CUDA (GPU)**: RTX 3070推奨（高速推論）
- **CPU**: 低速だが動作可能

## 今後の拡張

MVP実装後の拡張案：

- [ ] 複数モデル比較機能
- [ ] リアルタイムグラフ可視化（Plotly）
- [ ] 推論結果のCSVエクスポート
- [ ] 信頼度しきい値による警告
- [ ] モデルパフォーマンスメトリクス表示

## トラブルシューティング

### モデルロードエラー

```
FileNotFoundError: Model file not found
```

**解決策**: `prodmodel.yaml`の`model_path`を確認

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**解決策**: デバイスを"cpu"に変更

### 正規化エラー

```
ValueError: Min-Max normalization parameters not found
```

**解決策**: `prodmodel.yaml`に`min_max.min`と`min_max.max`を追加

## ライセンス

ApexAI Project (MIT License)

## お問い合わせ

問題や改善案は [GitHub Issues](https://github.com/yourusername/apexai/issues) へ
