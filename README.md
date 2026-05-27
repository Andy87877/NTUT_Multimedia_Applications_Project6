<div align="center">

# 🏎️ JetBot AI 路牌辨識專案

**多媒體技術與應用 Project 6 — Jetbot Road Sign Recognition**

<img src="https://img.shields.io/badge/Python-3.6-blue?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Model-YOLOv4--tiny-EE4C2C?style=for-the-badge" />
<img src="https://img.shields.io/badge/NVIDIA-TensorRT-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
<img src="https://img.shields.io/badge/Hardware-Jetson_Nano-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
<img src="https://img.shields.io/badge/Training-Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" />

<br /><br />

本專案利用 NVIDIA Jetson Nano（JetBot）進行道路路牌辨識，結合 **YOLOv4-tiny** 模型與 **TensorRT** 加速推論，實現即時自走車路牌識別與自動反應行為。

</div>

---

<details open>
<summary><b>🧑‍🎓 專案團隊 & 課程資訊</b></summary>

<br>

| 項目 | 內容 |
|------|------|
| **指導教授** | 陳彥霖 (Yen-Lin Chen), Ph.D. |
| **課程單位** | 國立臺北科技大學 電資學士班 / Spring 2026 |
| **組員** | 113820033 謝奕宏 ／ 113820020 林政德 ／ 112820034 呂伊茹 |
| **Demo 截止** | 115/6/10 |
| **報告截止** | 115/6/12 23:59 |

</details>

---

## 📋 目錄

| # | 步驟 | 工具 |
|---|------|------|
| [1](#step-1--收集路牌影像資料) | 收集路牌影像資料 | Jetbot / OpenCV |
| [2](#step-2--用-labelimg-標記資料) | 用 LabelImg 標記資料 | LabelImg |
| [3](#step-3--準備訓練所需檔案) | 準備訓練所需檔案 | 手動建立 |
| [4](#step-4--設定-google-colab-並-clone-darknet) | 設定 Google Colab & Clone Darknet | Google Colab |
| [5](#step-5--訓練-yolov4-tiny-模型) | 訓練 YOLOv4-tiny 模型 | Darknet |
| [6](#step-6--設定-jetbot-推論環境) | 設定 Jetbot 推論環境 | Jetbot Terminal |
| [7](#step-7--編譯-yolo-tensorrt-plugin) | 編譯 YOLO TensorRT Plugin | Jetbot Terminal |
| [8](#step-8--轉換模型為-onnx) | 轉換模型為 ONNX | yolo_to_onnx.py |
| [9](#step-9--轉換-onnx-為-tensorrt) | 轉換 ONNX 為 TensorRT | onnx_to_tensorrt.py |
| [10](#step-10--部署-project06ipynb-到-jetbot) | 部署到 Jetbot 執行 | JupyterLab |

---

## 🔧 環境需求

| 環境 | 規格 |
|------|------|
| Jetbot | Jetson Nano, Ubuntu, Python 3.6, CUDA 10.2, TensorRT 7.1.3.4 |
| 訓練平台 | Google Colab（免費 GPU） |
| 標記工具 | LabelImg（Windows / Linux） |

---

## 📷 Step 1 — 收集路牌影像資料

1. 在 Jetbot 桌面右鍵 → **Open Terminal**。

2. 安裝 jetcam：

   ```bash
   git clone https://github.com/NVIDIA-AI-IOT/jetcam
   cd jetcam
   sudo python3 setup.py install
   ```

3. 開啟瀏覽器，連至 `http://<Jetbot_IP>:8888`（密碼：`jetbot`）。

4. 在 Jupyter 新增 Notebook，以 OpenCV 批次拍攝：

   ```python
   import cv2, time
   webcam = cv2.VideoCapture(0)
   for i in range(1, 101):
       _, image = webcam.read()
       cv2.imwrite(f"pics/{i}.png", image)
       time.sleep(0.5)
   del(webcam)
   ```

5. 右鍵下載照片備份至本機。

> **Tip**：建議每類路牌至少收集 **30~50** 張，涵蓋不同角度、距離與光線條件，訓練效果更佳。

---

## 🏷️ Step 2 — 用 LabelImg 標記資料

1. 開啟 `labelImg.exe`，進入 `data/predefined_classes.txt`，確認內容：

   ```
   stop
   pedestrian
   rail
   blocked
   ```

2. 在 LabelImg 左側欄切換格式為 **YOLO**（非 PascalVOC）。

3. 開啟圖片資料夾，對每張圖：
   - 按 **W** → 拖曳框選路牌 → 選擇對應類別
   - 按 **Ctrl+S** 儲存（每張都要存）

4. 完成後，每張 `.jpg` 應有同名 `.txt`（YOLO 格式座標）。

---

## 📁 Step 3 — 準備訓練所需檔案

建立以下 5 個檔案，上傳至 Google 雲端 `FinalProject/` 資料夾：

| 檔案 | 內容重點 |
|------|----------|
| `obj.zip` | 所有影像 + 標記 `.txt` 打包 |
| `obj.names` | `stop` / `pedestrian` / `rail` / `blocked`（各一行） |
| `obj.data` | `classes=4`，訓練/驗證路徑，backup 路徑 |
| `yolov4-tiny-custom.cfg` | 修改 batch=64, subdivisions=16, max_batches=8000, steps=6400,7200, filters=27（兩處）, classes=4（兩處） |
| `generate_train.py` | 掃描 `obj/` 資料夾，產生 `train.txt` |

---

## ☁️ Step 4 — 設定 Google Colab 並 Clone Darknet

1. 開啟 Google Colab，切換 GPU：Edit → Notebook settings → **GPU** → Save。

2. 掛載雲端硬碟：

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !ln -s /content/drive/MyDrive /mydrive
   ```

3. Clone Darknet：

   ```bash
   %cd /mydrive/FinalProject/
   !git clone https://github.com/AlexeyAB/darknet
   ```

4. 修改 Makefile 開啟 GPU 支援：

   ```bash
   %cd /mydrive/FinalProject/darknet
   !sed -i "s/GPU=0/GPU=1/" Makefile
   !sed -i "s/OPENCV=0/OPENCV=1/" Makefile
   !sed -i "s/CUDNN=0/CUDNN=1/" Makefile
   !sed -i "s/CUDNN_HALF=0/CUDNN_HALF=1/" Makefile
   ```

5. 安裝 OpenCV 並編譯：

   ```bash
   !sudo apt install -y libopencv-dev
   !make clean && make
   !chmod 755 ./darknet
   ```

---

## 🧠 Step 5 — 訓練 YOLOv4-tiny 模型

1. 下載預訓練權重：

   ```bash
   !wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.conv.29
   ```

2. 解壓資料集、產生 train.txt：

   ```bash
   %cd ..
   !unzip obj.zip
   !python generate_train.py
   %cd darknet/
   ```

3. 在 `FinalProject/` 下建立 `backup/` 資料夾，然後開始訓練：

   ```bash
   !./darknet detector train data/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -dont_show
   ```

4. 中斷後可接續（使用 `_last.weights`）：

   ```bash
   !./darknet detector train data/obj.data cfg/yolov4-tiny-custom.cfg backup/yolov4-tiny_custom_last.weights -dont_show
   ```

5. 訓練完成後，備份最佳權重：

   ```bash
   !cp backup/yolov4-tiny_custom_best.weights /mydrive/FinalProject/
   ```

---

## ⚙️ Step 6 — 設定 Jetbot 推論環境

回到 Jetbot Terminal，依序執行：

```bash
sudo pip install protobuf==3.8.0
sudo apt-get install protobuf-compiler libprotoc-dev
sudo pip install onnx==1.4.1
sudo pip install --upgrade pip
sudo pip uninstall enum34
```

設定 CUDA 環境變數（編輯 `~/.bashrc`，在末尾加入）：

```bash
export PATH=/usr/local/cuda-10.2/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:${LD_LIBRARY_PATH}
```

重開 Terminal，再執行：

```bash
export CUDA_ROOT=/usr/local/cuda
pip3 install pycuda --verbose
```

---

## 🔨 Step 7 — 編譯 YOLO TensorRT Plugin

1. 下載並解壓縮 `trt_yolv4-tiny-master.zip`。
2. 進入 `plugins/` 資料夾，開啟 Terminal：

   ```bash
   make -j4
   ```

3. 確認資料夾中出現 `libyolo_layer.so`。

---

## 🔄 Step 8 — 轉換模型為 ONNX

1. 將訓練結果的兩個檔案放入 `trt_yolv4-tiny-master/yolo/`，並重命名：

   | 原始名稱 | 重命名 |
   |----------|--------|
   | `yolov4-tiny-custom.cfg` | `yolov4-tiny-416.cfg` |
   | `yolov4-tiny_custom_best.weights` | `yolov4-tiny-416.weights` |

2. 在 `yolo/` 資料夾 Terminal 中執行：

   ```bash
   python3 yolo_to_onnx.py -c 4 -m yolov4-tiny-416
   ```

3. 確認出現 `yolov4-tiny-416.onnx`。

---

## ⚡ Step 9 — 轉換 ONNX 為 TensorRT

```bash
python3 onnx_to_tensorrt.py -c 4 -m yolov4-tiny-416
```

確認輸出：`Serialized the TensorRT engine to file: yolov4-tiny-416.trt`

---

## 🚗 Step 10 — 部署 Project06.ipynb 到 Jetbot

1. 開啟 Jetbot Jupyter（`http://<Jetbot_IP>:8888`），上傳 `Project06.ipynb` 及 `yolov4-tiny-416.trt`。

2. 按順序執行以下 Cell：

   | Cell | 動作 |
   |------|------|
   | Cell 1 | 載入 TRT 模型 |
   | Cell 2 | 靜態影像測試（確認偵測正常） |
   | Cell 3 | 初始化相機與 Robot |
   | Cell 4–6 | 載入車道偵測函式 |
   | Cell 7 | 載入速度修正函式 |
   | Cell 8–9 | 載入路牌判斷與主更新函式 |
   | Cell 10 | 確認鏡頭畫面（除錯用） |
   | Cell 11 | **啟動自走車**（開始循環偵測） |

3. 停車：先 **Interrupt Kernel**，再執行 Cell 12（`robot.stop()`）。

4. 釋放資源：Cell 13（`camera_link.unlink()`）→ Cell 14（`camera.stop()`）。

> **Note**：Cell 11 啟動時，程式會先以 `0.37 / 0.41` 給予起步推力克服靜摩擦，0.3 秒後恢復巡航速度 `0.35`。右輪 speed 略高是因為馬達個體差異，請依實際車況調整。

---

## 🚦 路牌辨識動作對照

| 路牌 | Class ID | 偵測條件 | 觸發動作 |
|------|:--------:|----------|----------|
| 🛑 `stop` | 0 | bbox 寬 > 50 px | 停止 **3 秒**後繼續行駛 |
| 🚂 `rail` | 1 | bbox 寬 > 30 px | 停止 **5 秒**後繼續行駛 |
| 🚶 `pedestrian` | 2 | bbox 寬 > 50 px | 速度 × **0.7** 減速 |
| 🚧 `blocked` | 3 | bbox 寬 > 50 px | **立即停止**，不再繼續行駛 |

> **冷卻機制**：`stop` 與 `rail` 觸發後，各設定 **10 秒冷卻**，避免同一標誌重複觸發。
