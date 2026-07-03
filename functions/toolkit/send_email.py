#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = Last modified: 2026-07-03T20:19:14
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission


import os
import base64
from email.message import EmailMessage

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from obspy import UTCDateTime

# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion


# Permission for sending emails
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def get_google_service2():
    
    creds = None

    token_path = Path(project_root) / f"deploy/token.json"
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            credentials_path = Path(project_root) / f"deploy/credentials.json"
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_path, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def get_google_service():
    
    token_path = Path(project_root) / f"deploy/token.json"
    creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())

    return build("gmail", "v1", credentials=creds)


def send_email_by_google(service, sender, to, subject, body):
    
    msg = EmailMessage()
    msg.set_content(body)
    msg["To"] = to
    msg["From"] = sender
    msg["Subject"] = subject

    encoded = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    message = {"raw": encoded}

    sent = service.users().messages().send(
        userId="me",
        body=message
    ).execute()

    # print("Sent message ID:", sent["id"])


def usage():
    
    service = get_google_service()

    send_email_by_google(
        service,
        sender="qi.zhou.geo@gmail.com",
        to="ned.chow.ned@gmail.com",
        subject="This is subject.",
        body=f"{UTCDateTime.now().isoformat()} This email was sent using Google's Gmail API service to test the connection."
    )
    
if __name__ == "__main__":
    usage()
