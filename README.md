# RSS_positioning

1. 從網路上下載 UJIIndoorLoc dataset，免除 offline 資料收集步驟。
http://indoorlocplatform.uji.es/databases/get/1/
https://www.kaggle.com/giantuji/UjiIndoorLoc
TrainingData.csv: 用來訓練之 radio database。
ValidationData.csv: 用來驗證效果之 radio database。
說明文件可以參考：https://ieeexplore.ieee.org/document/7275492

2. 使用 Building 2 Floor 3 的資料做訓練與測試。

3. 撰寫程式從 TrainingData.csv 讀入 radio database，以自訂的資料結構
儲存。

4. 採用 Euclidean distance 作為 distance 計算之演算法，撰寫對應之程
式。某筆資料中未偵測到的 AP 訊號可定為 100 或是-104。

5. 讀入 ValiationData.csv，可一次讀入也可一次讀取一行。一筆資料做一
次測試，將新的 fingerprint 與 radio database 的每項計算 distance，
以距離最近的 fingerprint 其代表的位置作為預估的位置。計算真實座標
與預估座標的誤差。最後算出 ValidationData 中的各自距離誤差與平均
值，並繪製成直方圖。

6. 重複步驟四，但是改採用 KNN (K = 3 或 5)的方法，挑選最近 K 筆資料
將其 K 個座標值平均作為預估位置。同樣計算 ValidationData 中距離誤
差與平均值，並繪製成直方圖。

7. 在步驟六計算平均時，改成下面方法參考網址中權重計算的作法。
方法參考網址：http://indoorlocplatform.uji.es/methods/knn/
結果參考網址：http://indoorlocplatform.uji.es/dashboard/ujiindoorloc/
