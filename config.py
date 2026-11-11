# ===== Outlook / Microsoft Graph 设置 =====
CLIENT_ID = "在这里填你在 Azure 注册的应用程序(客户端) ID"
# 个人账号(outlook.com / hotmail.com)用 consumers；工作或学校账号改成 common
AUTHORITY = "https://login.microsoftonline.com/consumers"
SCOPES = ["Mail.ReadWrite", "Mail.Send"]
TOKEN_CACHE_FILE = "token_cache.json"

# ===== 模型设置 =====
MODEL_PATH = "ad_classifier.pt"
MAX_LEN = 256            # 每封邮件最多取多少个 token
AD_THRESHOLD = 0.9       # 广告概率高于这个值才算广告，宁可漏判也别误判

# ===== 行为设置 =====
DRY_RUN = True           # True：只打印判断结果，不动任何邮件。确认准确后再改成 False
SEND_REPLY = False       # True：给广告邮件回感谢信（注意：会让对方确认你的邮箱是活跃的）
MOVE_TO_FOLDER = True    # True：把广告邮件移到下面的文件夹
AD_FOLDER_NAME = "广告"
CHECK_INTERVAL = 300     # 每隔多少秒检查一次新邮件
MAX_REPLIES_PER_RUN = 10 # 每轮最多回复几封，防止意外群发

# 发件地址包含这些词时不回复（通常没人看，或可能引发自动回复循环）
SKIP_SENDER_KEYWORDS = ["noreply", "no-reply", "donotreply", "do-not-reply",
                        "mailer-daemon", "postmaster", "notification", "bounce"]

REPLY_TEMPLATES = [
    "您好，\n\n感谢您发来的信息，我已收到。目前暂无相关需求，如有需要会主动联系您。\n\n祝好！",
    "您好，\n\n谢谢您的推荐，我已了解相关内容。暂时不需要，感谢您的关注。\n\n此致\n敬礼",
]
