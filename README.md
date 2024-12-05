# TM_Instructor

## 安裝指南

### 1. 下載專案
首先，將專案從 GitHub 下載下來：
```bash
git clone https://github.com/eric-yhlu/TM_Instructor.git
```
```bash
cd TM_Instructor
```
### 2. 設置環境變數
請在 programe/.env 文件中添加 OpenAI API 金鑰，格式如下：
```bash
OPENAI_API_KEY=your_openai_api_key
```
### 3. 建立 Docker 映像檔
執行以下命令來建構 Docker 映像檔：
```bash
docker build -t my-flask-app .
```
### 4. 啟動 Docker 容器
```bash
docker run -p 5000:5000 my-flask-app
```
### 5. 使用應用程式
在瀏覽器中輸入主機位址來訪問應用程式，例如： http://192.168.X.X:5000

