# Author: Akira Kudo
# Created: 2024/05/12
# Last Updated: 2024/05/12

from slack_sdk import WebClient

TOKEN_PATH = r"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\RaymondLab\DeepLabCut\training\slack_token.txt"
CHANNEL_NAME = "automation"
USERNAME = "U073344PQ2W" # my own username ID

def send_slack_message(message : str, 
                       token : str=None, 
                       ping_user : str=USERNAME):
    if ping_user is not None:
        message = f"<@{ping_user}> " + message
    if token is None:
        token = read_token()

    # Set up a WebClient with the Slack OAuth token
    client = WebClient(token=token)

    # Send a message
    client.chat_postMessage(
        channel=CHANNEL_NAME, 
        text=message, 
        username="Bot User"
    )

def read_token(token_path : str=TOKEN_PATH):
    """Reads and returns the token path."""
    with open(token_path, 'r') as file:
        token = file.read()
    return token


if __name__ == "__main__":
    message = "This is my first Slack message from Python!"
    send_slack_message(message=message)