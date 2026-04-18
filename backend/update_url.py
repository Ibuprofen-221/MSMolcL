import requests
import json

# 配置项
GIST_ID = "1982a9bc1904fe82f7549dd57ccc88f6"
TOKEN = "your_github_token_here"
# 当前后端服务器运行端口的映射地址()
CURRENT_ADDR = "https://uu742891-9109-afb63769.bjb2.seetacloud.com:8443" 

def update_gist(new_url):
    headers = {
        "Authorization": f"token {TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {
        "files": {
            "server_config.json": {
                "content": json.dumps({"url": new_url})
            }
        }
    }
    response = requests.patch(
        f"https://api.github.com/gists/{GIST_ID}",
        headers=headers,
        data=json.dumps(data)
    )
    if response.status_code == 200:
        print("Successfully updated the address to GitHub!")
    else:
        print(f"Failed to update: {response.text}")

if __name__ == "__main__":
    update_gist(CURRENT_ADDR)