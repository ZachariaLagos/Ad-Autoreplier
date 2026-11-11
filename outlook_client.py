"""通过 Microsoft Graph API 读写 Outlook 邮件（Outlook 已不支持密码直接登录 IMAP，需要 OAuth）。"""
import os
import msal
import requests
import config

GRAPH = "https://graph.microsoft.com/v1.0"
TEXT_BODY = {"Prefer": 'outlook.body-content-type="text"'}  # 让 Graph 直接返回纯文本正文


class OutlookClient:
    def __init__(self):
        self.cache = msal.SerializableTokenCache()
        if os.path.exists(config.TOKEN_CACHE_FILE):
            with open(config.TOKEN_CACHE_FILE, encoding="utf-8") as f:
                self.cache.deserialize(f.read())
        self.app = msal.PublicClientApplication(
            config.CLIENT_ID, authority=config.AUTHORITY, token_cache=self.cache)

    def _token(self):
        result = None
        accounts = self.app.get_accounts()
        if accounts:
            result = self.app.acquire_token_silent(config.SCOPES, account=accounts[0])
        if not result:
            # 设备码登录：终端会提示你打开网址并输入一个代码，只需登录一次
            flow = self.app.initiate_device_flow(scopes=config.SCOPES)
            if "user_code" not in flow:
                raise RuntimeError(f"无法开始登录：{flow.get('error_description')}")
            print(flow["message"])
            result = self.app.acquire_token_by_device_flow(flow)
        if self.cache.has_state_changed:
            with open(config.TOKEN_CACHE_FILE, "w", encoding="utf-8") as f:
                f.write(self.cache.serialize())
        if "access_token" not in result:
            raise RuntimeError(f"登录失败：{result.get('error_description')}")
        return result["access_token"]

    def _request(self, method, path, params=None, json=None, headers=None):
        h = {"Authorization": f"Bearer {self._token()}", **(headers or {})}
        url = path if path.startswith("http") else GRAPH + path
        r = requests.request(method, url, params=params, json=json, headers=h, timeout=30)
        r.raise_for_status()
        return r.json() if r.content else None

    def iter_messages(self, folder_id="inbox", filter=None, limit=None):
        """逐封返回某个文件夹里的邮件，自动翻页。"""
        params = {"$top": 50, "$select": "id,subject,body,from,receivedDateTime"}
        if filter:
            params["$filter"] = filter
        url, count = f"/me/mailFolders/{folder_id}/messages", 0
        while url:
            data = self._request("GET", url, params=params, headers=TEXT_BODY)
            params = None  # 下一页的链接里已经带了参数
            for m in data.get("value", []):
                yield m
                count += 1
                if limit and count >= limit:
                    return
            url = data.get("@odata.nextLink")

    def find_folder(self, name):
        data = self._request("GET", "/me/mailFolders",
                             params={"$filter": f"displayName eq '{name}'"})
        items = data.get("value", [])
        return items[0]["id"] if items else None

    def get_or_create_folder(self, name):
        return self.find_folder(name) or self._request(
            "POST", "/me/mailFolders", json={"displayName": name})["id"]

    def reply(self, message_id, text):
        self._request("POST", f"/me/messages/{message_id}/reply", json={"comment": text})

    def move(self, message_id, folder_id):
        self._request("POST", f"/me/messages/{message_id}/move", json={"destinationId": folder_id})
