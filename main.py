from flask import (
    Flask,
    request,
    jsonify
)
import pickle
from flask.wrappers import Request
from textblob import TextBlob

columns = ['tamanho', 'ano', 'garagem']

model = pickle.load(open('./model.sav', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    """Rota de home"""
    return "<h1>hello word</h1>"


@app.route('/sentimento/<frase>')
def sentiment(frase):

    tb = TextBlob(frase)
    tb_en = tb.translate(to='en')
    polaridade = tb_en.sentiment.polarity
    return f'Polaridade: {polaridade}'


@app.route('/cotacao/', methods=['POST'])
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in columns]
    preco = model.predict([dados_input])
    return jsonify(preco=preco[0])


app.run(debug=True)
