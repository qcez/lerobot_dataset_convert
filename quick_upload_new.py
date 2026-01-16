from huggingface_hub import HfApi, create_repo, login, logout
import os
import sys
import subprocess

# === 关键修改：启用 hf-mirror 镜像（中国大陆加速） ===
# 在任何 huggingface_hub 操作之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"  # 启用高性能 Xet 上传（chunk级加速，断点续传更好）

# 配置
DATASET_PATH = "./folding_clothes_50_12_13_lerobot"
DATASET_NAME = "folding_clothes_50_12_13_lerobot"  # 数据集名称

def quick_upload():
    """快速上传 lerobot 格式数据集到 Hugging Face（使用镜像加速）"""
    
    # 打印当前端点，方便确认是否用了镜像
    print("当前 HF_ENDPOINT:", os.environ.get("HF_ENDPOINT", "未设置（默认 huggingface.co）"))
    print("HF_XET_HIGH_PERFORMANCE:", os.environ.get("HF_XET_HIGH_PERFORMANCE", "未设置"))

    # 检查是否需要强制重新登录
    force_login = "--relogin" in sys.argv or "-r" in sys.argv
    
    if force_login:
        print("强制重新登录...")
        try:
            logout()
        except:
            pass

    # 登录
    api = HfApi()
    try:
        user_info = api.whoami()
        username = user_info['name']
        print(f"✓ 已登录为: {username}")
        
        if force_login:
            confirm = input("\n是否继续使用此账号？(y/n): ").strip().lower()
            if confirm != 'y':
                logout()
                raise Exception("需要重新登录")
    except:
        print("\n请输入新的 Hugging Face Token:")
        print("(从 https://huggingface.co/settings/tokens 获取，需有 write 权限)")
        token = input("Token: ").strip()
        if not token:
            print("❌ Token 不能为空")
            sys.exit(1)
        login(token=token, add_to_git_credential=True)
        api = HfApi()
        user_info = api.whoami()
        username = user_info['name']
        print(f"✓ 成功登录为: {username}")

    repo_id = f"{username}/{DATASET_NAME}"
    print(f"\n目标仓库: {repo_id}")
    
    # 创建仓库（如果不存在）
    print("检查/创建仓库...")
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
    print("仓库已就绪")

    # 优先使用 huggingface-cli upload（更稳定，支持更好断点续传、分批 commit）
    print("\n开始上传（使用 huggingface-cli + Xet 高性能模式）...")
    print("建议在 tmux/screen 中运行，以防断开")
    
    cmd = [
        "huggingface-cli", "upload",
        "--repo-type", "dataset",
        repo_id,
        DATASET_PATH,
        ".",  # 上传整个文件夹到仓库根目录
        "--commit-message", "Upload lerobot folding clothes dataset (50 episodes)"
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print("\n✅ CLI 上传成功完成！")
    except subprocess.CalledProcessError as e:
        print("\nCLI 上传失败:", e)
        print("stderr:", e.stderr)
        print("\n回退到 Python API upload_folder...")
        
        # 回退方案：使用 api.upload_folder
        api.upload_folder(
            folder_path=DATASET_PATH,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload lerobot folding clothes dataset (fallback)",
            # 可选：忽略临时文件/缓存，提高稳定性
            ignore_patterns=["*.tmp", "__pycache__/*", ".git/*", "*.lock"]
        )
        print("\n✅ API 上传完成（fallback 模式）")

    print(f"\n上传完成！")
    print(f"🔗 数据集链接: https://huggingface.co/datasets/{repo_id}")
    print(f"   （若网络慢，可访问镜像站查看: https://hf-mirror.com/datasets/{repo_id}）")

if __name__ == "__main__":
    quick_upload()
