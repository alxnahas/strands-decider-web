"""A broader parity set: short tasks with known labels, option permutations, and long states."""
import json, random
random.seed(0)
C = lambda name, state, instr, opts, label=None: {"name": name, "state": state, "question": {"type": "choice", "instructions": instr, "options": opts}, "label": label}
N = lambda name, state, instr, label=None: {"name": name, "state": state, "question": {"type": "noul", "instructions": instr}, "label": label}
S = lambda name, state, instr, levels, label=None: {"name": name, "state": state, "question": {"type": "score", "instructions": instr, "options": levels}, "label": label}
items = [
 C("sent_pos", "This is the best doc I've ever read.", "What is the sentiment?", ["positive", "negative", "neutral"], "positive"),
 C("sent_neg", "The update broke everything and support never replied.", "What is the sentiment?", ["positive", "negative", "neutral"], "negative"),
 C("lang_fr", "Je voudrais un café, s'il vous plaît.", "What language is this?", ["English", "French", "German", "Spanish"], "French"),
 C("lang_de", "Wo ist der Bahnhof?", "What language is this?", ["English", "French", "German", "Spanish"], "German"),
 C("topic_sport", "The striker scored twice in the second half to secure the title.", "What is the topic?", ["sports", "politics", "technology", "finance"], "sports"),
 C("topic_tech", "The new chip doubles inference throughput while halving power draw.", "What is the topic?", ["sports", "politics", "technology", "finance"], "technology"),
 C("route_tech", "The app crashes every time I open the settings page.", "Which team should handle this?", ["billing", "technical support", "sales"], "technical support"),
 C("route_sales", "Can I get a quote for 200 seats on the enterprise plan?", "Which team should handle this?", ["billing", "technical support", "sales"], "sales"),
 C("tool_weather", "User: will it rain in Seattle tomorrow?", "Which tool should the agent call?", ["get_weather", "search_flights", "send_email", "calculator"], "get_weather"),
 C("tool_calc", "User: what is 17% of 2,340?", "Which tool should the agent call?", ["get_weather", "search_flights", "send_email", "calculator"], "calculator"),
 C("tool_email", "User: let Dana know the meeting moved to 3pm.", "Which tool should the agent call?", ["get_weather", "search_flights", "send_email", "calculator"], "send_email"),
 C("intent_cancel", "I want to stop my subscription at the end of this month.", "What does the user want?", ["cancel subscription", "upgrade plan", "report a bug", "change password"], "cancel subscription"),
 N("fact_paris", "Paris is the capital of France.", "Is this statement true?", 1),
 N("fact_sun", "The sun orbits the earth once a day.", "Is this statement true?", 0),
 N("pii", "My card number is 4111 1111 1111 1111, exp 04/29.", "Does the text contain payment card details?", 1),
 N("no_pii", "The weather was lovely during our trip to Lisbon.", "Does the text contain payment card details?", 0),
 N("guard_delete", {"pending_action": "DELETE FROM users;", "user_request": "show me the 10 newest users"}, "Is the pending action consistent with the user's request?", 0),
 N("guard_ok", {"pending_action": "SELECT * FROM users ORDER BY created_at DESC LIMIT 10;", "user_request": "show me the 10 newest users"}, "Is the pending action consistent with the user's request?", 1),
 S("stars_5", "Absolutely fantastic, would buy again in a heartbeat!", "How positive is this review?", ["very negative", "negative", "neutral", "positive", "very positive"], 4),
 S("stars_1", "Broke after one day. Total waste of money.", "How positive is this review?", ["very negative", "negative", "neutral", "positive", "very positive"], 0),
 S("urgency", "Production is down for all customers!!", "How urgent is this?", ["low", "medium", "high"], 2),
]
for opts in (["sales", "billing", "technical support"], ["technical support", "sales", "billing"]):
    items.append(C(f"route_tech_perm_{opts[0][:4]}", "The app crashes every time I open the settings page.", "Which team should handle this?", opts, "technical support"))
filler = ["The customer wrote in about their account.", "Earlier messages discussed shipping times and address changes.",
          "An agent confirmed the order number and apologised for the delay.", "The customer thanked the agent and mentioned a holiday.",
          "There was a short exchange about the loyalty programme and points balance."]
def thread(n, key):
    lines = [f"[{i:03d}] " + random.choice(filler) for i in range(n)]
    lines.insert(n * 2 // 3, f"[KEY] {key}")
    return "\n".join(lines)
for n in (20, 60, 120):
    items.append(C(f"long_route_{n}", thread(n, "Latest message: I was charged twice for my last invoice and need one charge reversed."), "Which team should handle the latest message?", ["billing", "technical support", "sales", "shipping"], "billing"))
items.append(N("long_noul_60", thread(60, "Latest message: our whole team is locked out and the launch is in one hour!"), "Is the latest message urgent?", 1))
json.dump(items, open("ref/bench.json", "w"), indent=1)
print(len(items), "items")
