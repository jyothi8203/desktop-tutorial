from flask import Flask,render_template,jsonify,request
from chatterbot import ChatBot
from chatterbot.conversation import Statement
from chatterbot.trainers import ChatterBotCorpusTrainer

from chat import get_response

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("base.html")


@app.post("/predict")
def predict():
    # response = get_response(text)
    # create ChatBot

    # Greeting from chat bot
    print("Hi, I am ChatBot")
        # response from ChatBot
        # put query on Statement format to avoid runtime alert messages
        # Statement(text=query, search_text=query)
    query = request.get_json().get("message")
    # response = chatBot.get_response(Statement(text=query, search_text=query))

    response = get_response(query)
    message = {"answer": response}
    return jsonify(message)


def app_run():
    app.run(debug=True,use_reloader=False)
    # chatBot = ChatBot('ChatBot')
    # if(chatBot):
    # # create ChatBot trainer
    #     trainer = ChatterBotCorpusTrainer(chatBot)
    #
    #     # Train ChatBot with English language corpus
    #     # you can train with different language
    #     # or with your custom .yam file
    #     trainer.train("chatterbot.corpus.english")


if __name__ == "__main__":
    app_run()
