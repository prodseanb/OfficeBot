from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

# chatbot instance
chatbot = ChatBot(
    'OfficeBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        'chatterbot.logic.MathematicalEvaluation',
        'chatterbot.logic.TimeLogicAdapter',
        'chatterbot.logic.BestMatch',
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'I am sorry, but I do not understand. I am still learning.',
            'maximum_similarity_threshold': 0.90
        }
    ],
    database_uri='sqlite:///database.sqlite3'
) 

# training with Qs and As from txt files
training_data_training = open('training_data/training.txt').read().splitlines()
training_data_personal = open('training_data/personal.txt').read().splitlines()

training_data = training_data_training + training_data_personal

trainer = ListTrainer(chatbot)
trainer.train(training_data)

# train with english Corpus data
trainer_corpus = ChatterBotCorpusTrainer(chatbot)
trainer_corpus.train('chatterbot.corpus.english')