#!/usr/bin/env python3
"""
快速上传脚本 - 自动检测用户名
"""

from huggingface_hub import HfApi, create_repo, login, logout
import os
import sys

# 配置
DATASET_PATH = "./folding_clothes_50_12_13_lerobot"
DATASET_NAME = "folding_clothes_50_12_13_lerobot"  # 数据集名称

def quick_upload():
    """快速上传"""
    
    # 检查是否需要重新登录
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
        print("(从 https://huggingface.co/settings/tokens 获取)")
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
    print(f"\n上传到: {repo_id}")
    
    # 创建仓库
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)
    
    # 上传
    print("\n开始上传...")
    api.upload_folder(
        folder_path=DATASET_PATH,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload lerobot_folding dataset"
    )
    
    print(f"\n✅ 完成！")
    print(f"🔗 https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    quick_upload()
