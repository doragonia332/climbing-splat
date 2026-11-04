# 攀岩牆 3D 重建 × Gaussian Splatting（瀏覽器內全管線原型）

在岩場離線連拍一組多視角照片，直接在瀏覽器裡跑 SfM 三維重建，再把點雲轉成 Gaussian splats 即時渲染——全程不上傳、不經任何伺服器。

**線上版（GitHub Pages）**：https://doragonia332.github.io/climbing-splat/

## 怎麼用

1. 打開頁面，上方檢視器會先顯示參考成果（COLMAP 以 128 張照片離線重建的建築模型），可以拖曳旋轉、滾輪縮放。
2. 第 2 節把照片加入佇列：可以現拍、選圖，或按「載入範例照片組（15 張）」用內建範例試跑。照片只存在你裝置的 IndexedDB。
3. 第 3 節按「開始重建」：OpenCV.js（WASM）在本機擷取特徵、估計相機位姿、三角化並做 bundle adjustment，完成後成果會自動載入檢視器，也能下載 `.splat` 檔。
4. 第 4 節是拍攝要領：15 張適用一顆抱石或一小面牆（約 3–6 m），沿牆連拍成弧、相鄰重疊至少 2/3、再補一排不同高度；整面大牆或長路線請拍 25–60 張。

## 技術組成

| 階段 | 做法 |
| --- | --- |
| 拍攝 / 離線佇列 | 靜態頁面 + IndexedDB 存原圖；讀 EXIF 的 35mm 等效焦距當相機內參 |
| SfM 重建 | OpenCV.js（WASM）AKAZE 特徵 + 純 JS 幾何層（八點法 + LO-RANSAC、DLT 三角化、自寫 LM bundle adjustment）；焦距不準時自動掃描倍率驗證 |
| 點雲 → Gaussians | 稀疏點直接轉 gaussians（PCA 方向正規化、kNN 距離決定大小、觀測平均色）；參考模型另備抽稀 lite 版先出圖、完整版背景換入 |
| 渲染 | 改自 [antimatter15/splat](https://github.com/antimatter15/splat)（MIT）的 WebGL 渲染器，輸出標準 `.splat` |

## 誠實的限制

- 本機重建是「稀疏 SfM + 點 splat」：可辨識路線與岩點位置，不是照片級的 3DGS 訓練成果。照片級畫質需要真正的 3DGS 訓練（可微渲染迭代），建議正式版採雙軌：本機快速預覽 + 雲端 GPU 精修。
- 室內岩場若牆面太素、岩點太少，特徵會抓不到——拍攝時讓每張照片帶到足夠彩色岩點最重要。
- 範例照片為 COLMAP 官方 South Building 資料集的連續 15 幀（模擬沿牆連拍情境）；參考模型為同資料集 128 張的 COLMAP 重建（77,438 點）。

## 發布版與原始建置的差異

- 參考模型資料（lite / 完整版）以 base64 內嵌於程式檔內（`assets/splat-viewer.js`、`assets/app.js`），不另外 fetch `.splat` 檔。
- 範例照片以 base64 內嵌於 `assets/examples-data.js`（發布管道無法推送二進位檔案）；`app.js` 優先讀內嵌資料，無內嵌時才 fetch `assets/examples/`。
- OpenCV.js 因發布管道的請求大小上限拆成 `assets/opencv-part1/2/3.js` 三段供應，`app.js` 載入時依序 fetch 後合併注入；合併結果與原始整檔逐位元組相同（SHA-256 一致）。
- 其餘行為與建置版本一致。

## 授權 / 出處

- 檢視器基於 antimatter15/splat（MIT License）
- OpenCV.js（Apache 2.0）
- 範例影像：COLMAP 官方範例資料集（South Building）
