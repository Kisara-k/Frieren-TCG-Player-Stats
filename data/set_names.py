import csv
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not TOKEN:
    raise ValueError("DISCORD_BOT_TOKEN environment variable is not set")

CSV_PATH = os.path.join(os.path.dirname(__file__), "Player.csv")
MAX_WORKERS = 1 # Discord API rate limit is 50 requests per second, so we use 1 worker to be safe


def fetch_username(user_id):
    url = f"https://discord.com/api/v10/users/{user_id}"
    headers = {
        "Authorization": f"Bot {TOKEN}"
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"[ERROR] {user_id}: {response.status_code} {response.text}")
        return user_id, None

    data = response.json()
    return user_id, data.get("username")


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    id_to_row = {}
    for row in rows:
        discord_id = row.get("discordId", "").strip()
        if discord_id and not row.get("discordName", "").strip():
            id_to_row[discord_id] = row

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_username, uid): uid for uid in id_to_row}
        for future in as_completed(futures):
            uid, username = future.result()
            if username is not None:
                id_to_row[uid]["discordName"] = username
                print(f"{uid} -> {username}")

    fieldnames = ["id", "discordId", "name", "discordName"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done. Player.csv updated.")


if __name__ == "__main__":
    main()