"""主程序：定时检查收件箱，识别广告邮件，按设置回复感谢信并移到“广告”文件夹。"""
import json
import os
import random
import time
import config
from outlook_client import OutlookClient
from model import load_classifier, ad_probability
from collect_data import email_text

STATE_FILE = "state.json"


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            s = json.load(f)
        return set(s["processed"]), set(s["replied_senders"])
    return set(), set()


def save_state(processed, replied):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed": list(processed)[-5000:], "replied_senders": list(replied)},
                  f, ensure_ascii=False)


def sender_of(m):
    return (m.get("from", {}).get("emailAddress", {}).get("address") or "").lower()


def run_once(client, model, tok, ad_folder_id, processed, replied):
    replies = 0
    for m in client.iter_messages("inbox", filter="isRead eq false", limit=100):
        if m["id"] in processed:
            continue
        prob = ad_probability(model, tok, email_text(m), config.MAX_LEN)
        addr, subject = sender_of(m), m.get("subject") or "(无标题)"
        is_ad = prob >= config.AD_THRESHOLD
        print(f"[{'广告' if is_ad else '正常'} {prob:.2f}] {addr} | {subject}")

        if is_ad and not config.DRY_RUN:
            can_reply = (config.SEND_REPLY and replies < config.MAX_REPLIES_PER_RUN
                         and addr and addr not in replied
                         and not any(k in addr for k in config.SKIP_SENDER_KEYWORDS))
            if can_reply:
                client.reply(m["id"], random.choice(config.REPLY_TEMPLATES))  # 先回复，移动后 id 会变
                replied.add(addr)  # 同一个发件人只回复一次
                replies += 1
                print("    -> 已回复感谢信")
            if config.MOVE_TO_FOLDER:
                client.move(m["id"], ad_folder_id)
                print(f"    -> 已移到“{config.AD_FOLDER_NAME}”")
        processed.add(m["id"])


def main():
    model, tok = load_classifier(config.MODEL_PATH)
    client = OutlookClient()

    if config.DRY_RUN:
        print("当前是试运行模式，只显示判断结果，不会改动任何邮件。\n")
        run_once(client, model, tok, None, set(), set())
        return

    ad_folder_id = client.get_or_create_folder(config.AD_FOLDER_NAME)
    processed, replied = load_state()
    while True:
        try:
            run_once(client, model, tok, ad_folder_id, processed, replied)
            save_state(processed, replied)
        except Exception as e:
            print(f"出错了，稍后重试：{e}")
        time.sleep(config.CHECK_INTERVAL)


if __name__ == "__main__":
    main()
