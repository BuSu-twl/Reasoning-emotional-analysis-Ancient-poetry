"""
BERT模型下载脚本
支持自动重试和进度显示
"""
import requests
from pathlib import Path
import sys
import time

def download_file_with_retry(url: str, output_path: str, max_retries: int = 5, timeout: int = 60):
    """
    下载文件，支持重试和进度显示
    
    Args:
        url: 下载链接
        output_path: 输出文件路径
        max_retries: 最大重试次数
        timeout: 超时时间（秒）
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"开始下载: {url}")
    print(f"保存到: {output_path}")
    
    for attempt in range(max_retries):
        try:
            print(f"\n尝试 {attempt + 1}/{max_retries}...")
            
            # 发送HEAD请求获取文件大小
            try:
                head_response = requests.head(url, timeout=timeout)
                total_size = int(head_response.headers.get('content-length', 0))
                if total_size > 0:
                    print(f"文件大小: {total_size / (1024*1024):.2f} MB")
            except:
                total_size = 0
                print("无法获取文件大小，开始下载...")
            
            # 下载文件
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            downloaded = 0
            chunk_size = 8192
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 显示进度
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            print(f"\r进度: {percent:.1f}% ({mb_downloaded:.2f}/{mb_total:.2f} MB)", end='', flush=True)
                        else:
                            mb_downloaded = downloaded / (1024 * 1024)
                            print(f"\r已下载: {mb_downloaded:.2f} MB", end='', flush=True)
            
            print(f"\n✅ 下载完成: {output_path}")
            print(f"文件大小: {output_path.stat().st_size / (1024*1024):.2f} MB")
            return True
            
        except requests.exceptions.Timeout:
            print(f"❌ 超时错误 (尝试 {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print("❌ 下载失败：已达到最大重试次数")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求错误: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print("❌ 下载失败：已达到最大重试次数")
                return False
                
        except Exception as e:
            print(f"❌ 未知错误: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print("❌ 下载失败：已达到最大重试次数")
                return False
    
    return False


def main():
    """主函数"""
    print("=" * 60)
    print("BERT模型下载工具")
    print("=" * 60)
    
    model_dir = Path("models/bert-base-chinese")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 询问是否使用镜像站点
    print("\n选择下载源:")
    print("1. Hugging Face 官方源 (huggingface.co)")
    print("2. HF-Mirror 镜像站点 (hf-mirror.com) - 推荐，速度更快")
    
    source_choice = input("\n请选择下载源 (1/2，默认2): ").strip() or "2"
    
    if source_choice == "2":
        base_url = "https://hf-mirror.com"
        print("✅ 使用 HF-Mirror 镜像站点")
    else:
        base_url = "https://huggingface.co"
        print("✅ 使用 Hugging Face 官方源")
    
    # 模型文件URL
    model_urls = [
        f"{base_url}/bert-base-chinese/resolve/main/pytorch_model.bin",
        f"{base_url}/bert-base-chinese/resolve/main/model.safetensors",
    ]
    
    output_files = [
        model_dir / "pytorch_model.bin",
        model_dir / "model.safetensors",
    ]
    
    print("\n可用的下载选项:")
    print("1. pytorch_model.bin (传统格式，兼容性好)")
    print("2. model.safetensors (新格式，更安全)")
    
    choice = input("\n请选择要下载的文件 (1/2，默认1): ").strip() or "1"
    
    if choice == "1":
        url = model_urls[0]
        output_path = output_files[0]
    elif choice == "2":
        url = model_urls[1]
        output_path = output_files[1]
    else:
        print("无效选择，使用默认选项1")
        url = model_urls[0]
        output_path = output_files[0]
    
    # 检查文件是否已存在
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n⚠️  文件已存在: {output_path}")
        print(f"文件大小: {file_size_mb:.2f} MB")
        overwrite = input("是否覆盖? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("跳过下载")
            return
        output_path.unlink()
    
    # 开始下载
    print(f"\n{'=' * 60}")
    success = download_file_with_retry(
        url=url,
        output_path=str(output_path),
        max_retries=5,
        timeout=120  # 120秒超时
    )
    
    if success:
        print(f"\n{'=' * 60}")
        print("✅ 下载成功！")
        print(f"模型文件位置: {output_path}")
        print("\n现在可以运行训练脚本:")
        print("  python train_poem_sentiment.py")
    else:
        print(f"\n{'=' * 60}")
        print("❌ 下载失败")
        print("\n建议:")
        print("1. 检查网络连接")
        print("2. 使用代理或VPN")
        print("3. 手动下载:")
        print(f"   {url}")
        print(f"   保存到: {output_path}")
        print("\n4. 或使用其他下载工具（如IDM、迅雷等）")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n下载已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

