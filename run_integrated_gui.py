"""
启动古诗词情感分析综合可视化界面
"""
import subprocess
import sys

if __name__ == "__main__":
    print("正在启动古诗词情感分析综合可视化界面...")
    print("请稍候，浏览器将自动打开...")
    print()
    
    try:
        # 启动 Streamlit 应用
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "poem_analysis_train_gui.py",
            "--server.port=8501",
            "--server.headless=false"
        ])
    except KeyboardInterrupt:
        print("\n程序已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        print("\n请确保已安装所有依赖:")
        print("pip install -r requirements.txt")
        print("\n或者手动启动:")
        print("streamlit run poem_analysis_train_gui.py")

