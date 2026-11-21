FROM python:3.13-slim
WORKDIR /bot
COPY . .
# Pin BOT_PROFILE: "baseline", "finishable", "duelist", "duelist_lock", "duelist_route", "peek", "dodge_only", "dodge"
ENV BOT_PROFILE=duelist
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]

