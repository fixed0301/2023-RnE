import slack_sdk

SLACK_TOKEN = 'xoxb-2329005458561-2340049133872-QwqGVz6Io1ZQVaQEf6h0naph'
SLACK_CHANNEL = '정렌이'


def Msg_bot(status):
    slack_token = SLACK_TOKEN   #slack bot token
    channel = SLACK_CHANNEL
    message = chat[status]
    client = slack_sdk.WebClient(token=slack_token)
    client.chat_postMessage(channel=channel, text=message)

chat = {'backward' : '미끄럼틀을 역행하고 있습니다.',
        'sit' : '앉아있습니다.',
        'slide' : '미끄럼틀을 내려오고 있습니다.',
        'swing' : '그네를 타고 있습니다.',
        'walk' : '걷고 있습니다.',
        'collision' : '충돌 위험이 있습니다'} # 1p : swing / slide, 2p : any
