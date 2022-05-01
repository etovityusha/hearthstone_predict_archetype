from typing import List, Optional

import pandas as pd
import uvicorn
import requests
import cachetools.func

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from hearthstone.deckstrings import Deck as HSDeck
from imblearn.pipeline import Pipeline

app = FastAPI()


@cachetools.func.ttl_cache(maxsize=None, ttl=3600)
def get_trained_model():
    response = requests.get('http://152.70.52.223:6100/decks').json()
    lst = []
    for archetype in response['archetypes']:
        lst.extend([(archetype['archetype_title'], ' '.join(map(str, x))) for x in archetype['decks']])
    df = pd.DataFrame(data=lst, columns=['deck_title', 'cards'])
    x = df['cards']
    y = df['deck_title']
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))), ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())])
    return text_clf.fit(x, y)


class PredictIn(BaseModel):
    cards: Optional[List[int]]
    deckstring: Optional[str]


class PredictOut(BaseModel):
    prediction: str


def deckstring_to_cards(deckstring):
    deck = HSDeck.from_deckstring(deckstring)
    cards = deck.cards
    result = []
    for card in cards:
        result.append(card[0])
        if card[1] == 2:
            result.append(card[0])
    return result


@app.post('/predict', response_model=PredictOut)
async def predict_deck_title(deck: PredictIn):
    if deck.cards:
        if len(deck.cards) != 30:
            raise HTTPException(400, 'There should be 30 cards')
        cards = ' '.join(map(str, deck.cards))
    else:
        cards = ' '.join(map(str, deckstring_to_cards(deck.deckstring)))
    df = pd.DataFrame([cards], columns=['cards'])
    model = get_trained_model()
    prediction = model.predict(df['cards'])
    return PredictOut(prediction=prediction[0])


if __name__ == "__main__":
    uvicorn.run("run:app", host='0.0.0.0', port=6200, reload=True)
