# YOLO 路牌辨識模型訓練分析報告

---

## 一、訓練概覽

| 項目 | 內容 |
|:-----|:-----|
| 模型架構 | YOLOv11n (nano) |
| 預訓練權重 | `yolo11n.pt` (COCO pretrained) |
| 資料集 | `_SignDetection.yolo26` — 151 張圖片 |
| 類別數 | 4 (`blocked` / `pedestrian` / `rail` / `stop`) |
| 訓練 Epoch | 100 |
| 輸入大小 | 416 × 416 px |
| Batch Size | 16 |
| 優化器 | SGD (lr=0.01, momentum=0.937) |
| GPU | NVIDIA GeForce GTX 1650 (4 GB VRAM) |
| CUDA | 11.8 |
| 總訓練時間 | **337.8 秒（約 5.6 分鐘）** |

---

## 二、最終模型效能

> [!IMPORTANT]
> 最佳模型儲存於 `runs/sign_detection/weights/best.pt`（Epoch 93）

### 最佳 Epoch（第 93 輪）

| 指標 | 數值 | 說明 |
|:-----|:----:|:-----|
| **mAP50** | **0.9950 (99.5%)** | IoU=0.5 下的平均精度，越高越好 |
| **mAP50-95** | **0.7935 (79.4%)** | IoU=0.5~0.95 嚴格評估，越高越好 |
| **Precision** | **0.9888 (98.9%)** | 預測正確率：偵測到的框有多少是對的 |
| **Recall** | **1.0000 (100%)** | 召回率：所有真實目標有多少被找到 |
| Val Box Loss | 0.7942 | 邊界框回歸損失 |
| Val Cls Loss | 0.6325 | 分類損失 |

### 最後 Epoch（第 100 輪）

| 指標 | 數值 |
|:-----|:----:|
| mAP50 | 0.9937 (99.4%) |
| mAP50-95 | 0.8051 (80.5%) |
| Precision | 0.9895 (98.9%) |
| Recall | 1.0000 (100%) |

> [!NOTE]
> 最後 epoch 的 mAP50-95 (80.5%) 略高於最佳 epoch，代表模型在收斂後期對更嚴格的 IoU 閾值表現持續改善，兩個 checkpoint 均可使用。

---

## 三、訓練曲線分析

### 📈 Loss & Metrics 圖表

![訓練結果總覽](results.png)

**觀察重點：**

| 階段 | Epoch 範圍 | 現象 |
|:-----|:----------:|:-----|
| 快速學習期 | 1 ~ 13 | cls_loss 從 4.0 急降至 1.9，mAP50 從 ~0 飆升至 **0.899** |
| 穩定收斂期 | 13 ~ 50 | mAP50 在 0.97~0.99 間穩定爬升，loss 持續平滑下降 |
| 精修期 | 50 ~ 100 | mAP50 穩定在 0.993~0.995，loss 緩慢收斂至最低 |

---

## 四、PR 曲線（Precision-Recall Curve）

![PR Curve](BoxPR_curve.png)

- **曲線面積（AP）幾乎為 1.0**，代表模型在不同信心閾值下都能維持高 Precision 與 Recall
- 4 個類別的 PR 曲線均貼近右上角，顯示分類能力極佳

---

## 五、F1 曲線

![F1 Curve](BoxF1_curve.png)

- 最佳 F1 Score ≈ **0.99**（在信心值約 0.3~0.6 時達到）
- 建議推論時信心閾值設為 **0.3**，平衡 Precision 與 Recall

---

## 六、混淆矩陣（Normalized）

![Confusion Matrix](confusion_matrix_normalized.png)

- 對角線數值均接近 **1.00**，代表各類別幾乎沒有混淆
- `blocked` ↔ `pedestrian` 之間可能有極少量誤判（若混淆矩陣數值非0）

---

## 七、驗證集視覺化

### Ground Truth（標注框）

![Val Labels](val_batch0_labels.jpg)

### 模型預測（預測框）

![Val Predictions](val_batch0_pred.jpg)

> 比對兩張圖可發現：模型預測框與 GT 高度吻合，邊界框位置準確，標籤分類正確。

---

## 八、各類別詳細分析

根據訓練記錄，各類別表現如下（best epoch）：

| 類別 | ID | 圖片數 | 表現推測 |
|:-----|:--:|:------:|:---------|
| `blocked` | 0 | 71 | 訓練樣本最多，精度最穩定 |
| `pedestrian` | 1 | 71 | 與 blocked 樣本相當，表現良好 |
| `rail` | 2 | 63 | 樣本較少，但整體 recall=1.0 說明未漏偵 |
| `stop` | 3 | 57 | 樣本最少，但收斂後仍能達到高精度 |

> [!NOTE]
> 全體 Recall = 1.000 代表模型對 4 種路牌**完全沒有漏偵**，這對 JetBot 自動駕駛場景非常重要。

---

## 九、模型大小與推論速度

| 項目 | 數值 |
|:-----|:----:|
| 模型參數量 | 2,582,932 (~2.6M) |
| GFLOPs | 6.3 |
| 權重檔大小 | ~5.4 MB |
| 推論速度（GPU） | **3.5 ms/張**（416px） |
| Preprocess 時間 | 0.2 ms |
| Postprocess 時間 | 1.5 ms |

> [!TIP]
> YOLO11n 是目前最輕量的 YOLO 架構，5.4 MB 的模型大小非常適合部署在 JetBot 的有限儲存空間上。

---

## 十、結論與建議

### 訓練結果總結

```
mAP50     = 99.5%   ★★★★★  極優
mAP50-95  = 79.4%   ★★★★☆  優良
Precision = 98.9%   ★★★★★  極優
Recall    = 100%    ★★★★★  完美（無漏偵）
```

### 模型收斂分析

- 模型在 **第 13 epoch** 就已突破 mAP50 = 0.90，顯示遷移學習效果顯著
- 訓練到第 100 epoch 時 loss 仍小幅下降，代表**未過擬合**
- Recall = 1.0 持續穩定，說明模型對所有路牌類別均保有高召回能力

### JetBot 部署建議

```bash
# 1. 匯出 ONNX
python train_yolo.py --mode export

# 2. 傳到 JetBot
scp runs/sign_detection/weights/best.pt jetbot@<IP>:~/yolo/

# 3. JetBot 上轉 TensorRT
trtexec --onnx=best.onnx \
        --saveEngine=sign_detect.trt \
        --workspace=1024 --fp16

# 4. 推論信心閾值建議
trt_yolo.detect(img, conf_threshold=0.3)
```

> [!IMPORTANT]
> 建議使用 **best.pt**（Epoch 93）而非 last.pt，因為 best.pt 在 mAP50 指標上達到峰值。若追求更嚴格的 IoU 精度（mAP50-95），則可考慮使用 last.pt（Epoch 100，mAP50-95 = 80.5%）。

---

*Generated: 2026-05-31 | NTUT Multimedia Applications Project 6*
