import slack_sdk

SLACK_TOKEN = 'xoxb-2329005458561-2340049133872-QwqGVz6Io1ZQVaQEf6h0naph'
SLACK_CHANNEL = '정렌이'


def Msg_bot(slack_message):
    slack_token = SLACK_TOKEN   #slack bot token
    channel = SLACK_CHANNEL
    message = slack_message
    client = slack_sdk.WebClient(token=slack_token)
    client.chat_postMessage(channel=channel, text=message)


chat = "가나다라"

Msg_bot(chat)