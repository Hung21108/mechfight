FROM python:3.13-slim
WORKDIR /bot
COPY . .
# Pin BOT_PROFILE for each submission ("baseline", "finishable", or "duelist")
ENV BOT_PROFILE=duelist
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]

