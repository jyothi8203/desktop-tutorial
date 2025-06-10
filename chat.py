import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import pymsteams
import smtplib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
myTeamsMessage = pymsteams.connectorcard("https://netorgft10794464.webhook.office.com/webhookb2/93692ac9-48ba-473b-a7b1-aad0a8b7733a@3ff7d35e-7261-45b2-a314-df084e9a21b7/IncomingWebhook/ac4cb9839bf64fd2a4e4f11bd60d2bb2/354c7ac6-a743-497f-9377-2191f0fdbd72")#<Microsoft Webhook URL>
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

emp_eml_dict = {}
with open("emp_mails.json","r") as file:
    data = json.load(file)

    for item in data:
        name = item['name'].lower()
        e_value = item['emails']
        emp_eml_dict[name] = [e_value]

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"


def send_teams_message(strEmpName,strEmpEmail,strSubj,strTxt):
    # strSubj ="AiFA LABS Bot Message"
    myTeamsMessage.title(strSubj)
    strTxt = ''.join(["@",strEmpName,strTxt])
    myTeamsMessage.text(strTxt)
    myTeamsMessage.send()


def send_smtp_email(strEmpName,strEmpEmail,strSubj,strTxt):
    try:
        # Set sender mail, receivers mail and messages
        sender_mail = "sivajyothi.chandra@aifalabs.com"
        receivers_mail = strEmpEmail #[ 'junaid.hussain@aifalabs.com','shiva.borra@aifalabs.com','sivajyothi.chandra@aifalabs.com']
        to = ";".join(receivers_mail)
        print(to)
        message = strSubj + "This mail is sent using Bot service.\n"+strTxt
        obj = smtplib.SMTP('smtp.gmail.com', 587)
        print("after smtp")
        # secure with TLS
        obj.starttls()
        print("after smtpstart tls")
        # Mail Server Authentication
        obj.login("sivajyothi.chandra@gmail.com", "cpjvxzktwtnpxise")
        print("after smtp login")
        # sending the mail
        obj.sendmail(sender_mail, to, message)
        print("Mail sent successfully.")

    # terminating the session
        obj.quit()

    except Exception:
        print("Mail delivery failed.")


def check_emp_to_meet(tkn_lst):
    value = [tkn_lst.index(tkn)  if tkn in emp_eml_dict.keys() else -1 for tkn in tkn_lst]
    value = value[0]
    print("In get response, check emp function",value)
    if value != -1:
        tkn = tkn_lst[value]
        (nm, eml) = (tkn, emp_eml_dict[tkn])
        print("emp found",nm,eml)
        strMsg = """Some one is waiting for you at Reception. 
        You are requested to come here to meet them.
        Thanks
        AiFA Bot"""

        send_teams_message(nm,eml,"AiFA LABS Bot Message",strMsg)
        send_smtp_email(nm,eml,"AiFA LABS Bot Email",strMsg)
    return value


def get_response(msg):
    sentence = tokenize(msg)
    found = check_emp_to_meet(sentence)
    if found != -1:
        return "Request you to wait for 5 minutes. They will be here shortly.\nThank you visiting AiFA!!!"
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

