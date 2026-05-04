from longai.reasoning.router import route_query


def test_router():
    assert route_query("what is next intent") == "next_intent_prediction"
