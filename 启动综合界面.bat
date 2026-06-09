@echo off
chcp 65001 >nul
echo 正在启动古诗词情感分析综合可视化界面...
echo 请稍候，浏览器将自动打开...
echo.

python -m streamlit run poem_analysis_train_gui.py --server.port=8501 --server.headless=false

pause

