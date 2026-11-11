"""从你自己的 Outlook 导出训练数据。
先在 Outlook 里建一个“广告”文件夹，手动拖几百封广告邮件进去；收件箱里剩下的当作正常邮件。"""
import argparse
import csv
import config
from outlook_client import OutlookClient


def email_text(m):
    return (m.get("subject") or "") + "\n" + (m.get("body", {}).get("content") or "")[:3000]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ad-folder", default=config.AD_FOLDER_NAME)
    p.add_argument("--normal-folder", default="inbox")
    p.add_argument("--limit", type=int, default=2000, help="每类最多导出多少封")
    p.add_argument("--out", default="data.csv")
    args = p.parse_args()

    client = OutlookClient()
    ad_id = client.find_folder(args.ad_folder)
    if not ad_id:
        raise SystemExit(f"没找到文件夹“{args.ad_folder}”，请先在 Outlook 里建好并放入广告邮件。")
    normal_id = args.normal_folder if args.normal_folder == "inbox" else client.find_folder(args.normal_folder)

    rows = []
    for folder_id, label in [(ad_id, 1), (normal_id, 0)]:
        n = 0
        for m in client.iter_messages(folder_id, limit=args.limit):
            rows.append({"text": email_text(m), "label": label})
            n += 1
        print(f"{'广告' if label else '正常'}邮件：{n} 封")

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"已保存到 {args.out}")


if __name__ == "__main__":
    main()
