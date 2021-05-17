from dialobot import Intent

# create intent object
intent = Intent(lang="en")
intent.clear()

# add intent examples
# it is similar with chatbot builder like dialogflow
intent.add(data=[
    ("hello", "greeting"),
    ("bye", "greeting"),
    ("What time is it now", "time"),
    ("Tell me current time", "time"),
    ("Tell me time", "time"),
    ("tell me restaurant", "restaurant"),
    ("please recommend restaurant", "restaurant"),
])

# recognize intent
_intent = intent.recognize("Recommend nearest restaurant please")
print(_intent)

# recognize intent with scores
_intent = intent.recognize(
    "What time is it now?",
    detail=True,
)
print(_intent)
