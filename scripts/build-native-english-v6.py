#!/usr/bin/env python3
"""Build an original English curriculum with verified, licensed local rehearsal.

Offline deterministic data preparation only: no training, network, model teacher,
private chats, or tokenizer fitting. Imported records retain their source/license.
Exclusions accept user-prompt JSON strings only, never evaluation answers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import tempfile

from silivra.agent import build_system_prompt
from silivra.arithmetic import calculate
from silivra.checkpoint import sha256
from silivra.data import load_tokenizer
from silivra.finetune import _manifest
from silivra.instructions import (
    _canonical, _check_splits, _digest, _read_records, encode_turns,
    parse_action, prepare_instructions, validate_record,
)
from silivra.tools import ToolRegistry


SOURCE = {
    "name": "Silivra original English curriculum v6 (AI-authored synthetic exercises)",
    "author": "Silivra contributors with Codex; original deterministic examples",
    "license": "MIT", "url": "https://github.com/zeousky/Silivra",
    "revision": "native-english-v6",
}
TRAIN_WORDS = "apple bird blue boat book cat dog door fish flower fox green house lake moon red river sky star sun tea tree water white wind yellow orange stone chair garden clock train bread road happy quiet small bright cold warm clean dry fresh round soft tall".split()
VALID_WORDS = "violet beach cloud deer field forest grass hill rain rose snow table town wood silver grape shell plum bench pond clear gentle sharp wide tiny".split()
TRAIN_NAMES = "Ada Amir Cora Finn Hana Iris Kai Leo Lina Maya Mina Noah Omar Ravi Sam Tara Vera Zara".split()
VALID_NAMES = "Alma Bela Dara Elena Emil Imani Lila Nia Oren Remy Theo Yara".split()
# (term, definition variants, concrete example). These are authored teaching text.
DEFINITIONS = [
    ("noun", ("A noun names a person, place, thing, or idea.", "A noun is a naming word for a person, place, thing, or idea.", "Nouns are words that name people, places, things, and ideas."), "In 'The dog runs', dog is a noun."),
    ("verb", ("A verb describes an action or a state.", "Verbs tell us what happens or what state something is in.", "A verb is a word for an action or a state of being."), "In 'Birds sing', sing is a verb."),
    ("adjective", ("An adjective describes a noun.", "An adjective gives more information about a noun.", "Adjectives describe people, places, or things named by nouns."), "In 'a red boat', red is an adjective."),
    ("adverb", ("An adverb describes a verb, adjective, or another adverb.", "An adverb can tell how, when, or where an action happens.", "Adverbs add information to verbs, adjectives, or other adverbs."), "In 'She walks slowly', slowly is an adverb."),
    ("pronoun", ("A pronoun stands in place of a noun.", "Pronouns are words such as he, she, it, we, and they that can replace nouns.", "A pronoun refers to a person or thing without repeating its name."), "In 'Maya is here. She is happy', She is a pronoun."),
    ("sentence", ("A sentence expresses a complete thought.", "A sentence is a group of words that communicates a complete thought.", "A sentence puts words together to make a complete statement, question, or command."), "'The sun is warm.' is a sentence."),
    ("question", ("A question asks for information.", "A question is a sentence used to ask something.", "Questions request an answer or information."), "'Where is the book?' is a question."),
    ("plural", ("A plural word refers to more than one.", "Plural means that there is more than one person or thing.", "The plural form names two or more things."), "Books is the plural of book."),
    ("singular", ("A singular word refers to one person or thing.", "Singular means one, rather than more than one.", "The singular form names just one thing."), "Book is singular; books is plural."),
    ("past tense", ("The past tense describes something that happened before now.", "Past-tense verbs tell about earlier actions or states.", "Use the past tense for something that already happened."), "'I walked home' uses the past tense."),
    ("present tense", ("The present tense describes something happening now or regularly.", "Present-tense verbs can describe current states and usual actions.", "Use the present tense for current or regular actions."), "'I walk home every day' uses the present tense."),
    ("future tense", ("The future tense describes something that will happen later.", "Future forms talk about actions or states after now.", "Use a future form to talk about something that has not happened yet."), "'I will walk home' talks about the future."),
    ("paragraph", ("A paragraph is a group of sentences about one main idea.", "A paragraph groups related sentences around a topic.", "Paragraphs organize writing into groups of related sentences."), "A short paragraph can describe one place."),
    ("synonym", ("A synonym is a word with a similar meaning to another word.", "Synonyms are different words with similar meanings.", "Two words are synonyms when their meanings are similar."), "Big and large are synonyms."),
    ("antonym", ("An antonym is a word with the opposite meaning to another word.", "Antonyms are words with opposite meanings.", "Two words are antonyms when they mean opposite things."), "Hot and cold are antonyms."),
    ("capital letter", ("A capital letter is an uppercase letter.", "Capital letters are uppercase forms such as A, B, and C.", "A capital letter is the larger written form used to begin sentences and names."), "A is a capital letter; a is lowercase."),
    ("full stop", ("A full stop marks the end of a statement.", "A full stop is the punctuation mark used at the end of many sentences.", "A full stop ends a statement; it is also called a period."), "There is a full stop at the end of 'I am here.'"),
    ("comma", ("A comma separates items or parts of a sentence.", "A comma is a punctuation mark that can separate list items or sentence parts.", "Commas help separate words and clauses in writing."), "The list 'red, blue, and green' uses commas."),
    ("English", ("English is a language used for speaking, reading, and writing.", "English is a language with its own words and grammar.", "People use the English language to communicate in speech and writing."), "'Hello' is an English greeting."),
    ("Python", ("Python is a programming language.", "Python is a language for writing computer programs.", "Programmers use Python to give instructions to computers."), "In Python, print('hello') displays hello."),
    ("function", ("A function is a block of code that performs a task.", "A function groups instructions that can be called when needed.", "Functions organize reusable work in a program."), "A Python function can return a calculated value."),
    ("variable", ("A variable is a name that refers to a value in a program.", "Variables let a program refer to values using names.", "A variable connects a name with a value."), "In x = 3, x is a variable name."),
    ("loop", ("A loop repeats a block of code.", "Loops let a program do something repeatedly.", "A loop runs instructions more than once."), "A for loop can visit each item in a list."),
    ("list", ("A list stores an ordered collection of items.", "A list holds items in a particular order.", "Lists collect items that can be accessed by their positions."), "[1, 2, 3] is a Python list."),
    ("dictionary", ("A dictionary stores values under keys.", "A dictionary connects keys with values.", "Dictionaries let a program look up a value by its key."), "A dictionary can store a person's age under the key 'age'."),
    ("computer", ("A computer follows instructions to process data.", "Computers process information by following programs.", "A computer is a machine that carries out programmed instructions."), "A computer can run a program that adds numbers."),
    ("web search", ("Web search finds pages or information on the internet.", "A web search looks for information published online.", "Web search helps locate relevant internet pages."), "A search result can include a page title and URL."),
    ("source", ("A source is where information comes from.", "Sources tell us where a statement or piece of information originated.", "A source provides information that can be checked."), "A book or web page can be a source."),
    ("evidence", ("Evidence is information that supports or challenges a claim.", "Evidence helps us check whether a claim is supported.", "Facts and observations can be evidence for or against an explanation."), "An observed measurement can provide evidence."),
    ("uncertainty", ("Uncertainty means that an answer is not known with confidence.", "Uncertainty is a lack of complete confidence or knowledge.", "When information is missing or unclear, there is uncertainty."), "If no age was supplied, I am uncertain about the person's age."),
]
REGULAR_NOUNS = "book cat dog bird boat door chair table garden flower train clock tree house road star cup bag lamp pen".split()
REGULAR_VERBS = [("walk", "walked"), ("jump", "jumped"), ("look", "looked"), ("play", "played"), ("help", "helped"), ("open", "opened"), ("call", "called"), ("talk", "talked"), ("work", "worked"), ("clean", "cleaned")]
MATH_PATTERN = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*([+*/-])\s*(\d+(?:\.\d+)?)(?![\w.])")


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def assistant_count(record):
    return sum(message["role"] == "assistant" for message in record["messages"])


def user_prompts(record):
    return {_canonical(message["content"]) for message in record["messages"] if message["role"] == "user"}


def math_problems(text):
    result = set()
    for left, operation, right in MATH_PATTERN.findall(text):
        if operation in {"+", "*"}:
            left, right = sorted((left, right))
        result.add((left, operation, right))
    return result


def conversation_key(record):
    return _digest([{**message, "content": parse_action(message["content"]) if message["role"] == "assistant" else _canonical(message["content"])} for message in record["messages"]])


def deduplicate(records):
    seen, result = set(), []
    for record in records:
        key = conversation_key(record)
        if key not in seen:
            seen.add(key)
            result.append(record)
    return result


def make_record(family, prompt, answer, *, tools=(), observation=None, history=()):
    action = answer if isinstance(answer, dict) else {"final": str(answer)}
    messages = [{"role": "system", "content": build_system_prompt(list(tools))}, *history,
                {"role": "user", "content": prompt}, {"role": "assistant", "content": dump(action)}]
    if observation is not None:
        value, final = observation
        messages += [{"role": "tool", "name": action["tool"], "content": dump(value)},
                     {"role": "assistant", "content": dump({"final": str(final)})}]
    identity = hashlib.sha256(dump(messages).encode()).hexdigest()[:20]
    return validate_record({"id": f"v6-{family}-{identity}", "source": dict(SOURCE), "messages": messages})


def build_original(seed=20261026):
    """Create independent narrow partitions before seeing model generations."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    rng = random.Random(seed)
    registry = ToolRegistry(Path.cwd(), allow_network=True)
    specs = {tool["function"]["name"]: tool for tool in registry.schemas()}
    all_contexts = [[], [specs["calculate"]], [specs["web_search"]], [specs["calculate"], specs["web_search"]]]

    def context():
        return rng.choices(all_contexts, weights=(7, 1, 1, 1), k=1)[0]

    def tool_context(name, both=False):
        return all_contexts[3] if both else [specs[name]]

    train, valid = [], []
    social = [
        (["Hi", "Hello", "Hey", "Good morning", "Good evening", "Hello there", "Hi Silivra", "Hey there", "Hello, are you here", "Greetings"], ["It is nice to say hello to you", "Hello once again"], ["Hello! How can I help you?", "Hi! What would you like to talk about?", "Hello there. What can I help with?"]),
        (["How are you", "How is it going", "Are you doing well", "How are things", "How are you today", "Are you okay", "How do you feel", "How are you doing"], ["How have things been with you", "How is your day so far"], ["I'm here and ready to help. How are you?", "I'm ready to chat and help. How are things with you?", "I'm doing fine as an AI. What would you like to discuss?"]),
        (["Who are you", "Tell me your name", "What is your name", "Introduce yourself", "What are you called", "What are you", "Tell me who you are", "Describe yourself"], ["What name should I use for you", "Could you tell me your identity"], ["I'm Silivra, an experimental AI trained from scratch.", "My name is Silivra. I use my own small model trained from scratch.", "I'm Silivra, a small local AI with my own model and tokenizer."]),
        (["Thank you", "Thanks", "Thanks for helping", "That helped", "Thank you very much", "I appreciate that", "Thanks for your time", "Thank you for the explanation"], ["Many thanks for explaining that", "Your help was useful, thank you"], ["You're welcome!", "Glad I could help.", "You're welcome. Let me know if you have another question."]),
        (["Goodbye", "Bye", "See you", "Good night", "I am leaving now", "See you later", "Talk later", "I need to go"], ["We can talk again another time", "It is time for me to leave"], ["Goodbye! Take care.", "See you later!", "Take care. We can chat again later."]),
        (["Can you help", "I need help", "Help me please", "I have a problem", "Can we work on something", "Help please", "I want some help", "I have a question"], ["Could we work through a problem together", "I need help but have not given details yet"], ["What would you like help with?", "Please tell me about the problem so I can help.", "Sure. What are you working on?"]),
        (["Are you always right", "Can you make mistakes", "What are your limits", "Are you a reliable expert", "Can I trust every answer", "Should I check your answers", "Tell me your limitations", "Are your answers perfect"], ["Could your explanations contain mistakes", "How certain should I be about your answers"], ["I can make mistakes. Please check important information.", "I'm an experimental AI, so my answers can be wrong.", "My abilities are limited. Verify important facts and calculations."]),
        (["I am frustrated", "This problem is annoying", "I am stuck", "I feel confused", "I do not understand this", "This is difficult for me", "I keep making mistakes", "I am having trouble"], ["I am finding this task discouraging", "I feel lost while trying to understand"], ["That sounds frustrating. Tell me what you are trying to do, and we can take it step by step.", "Let's take it one step at a time. What part is unclear?", "I can try to help. Please share the problem or error message."]),
        (["Are you human", "Do you have human feelings", "Are you a person", "Are you alive", "Do you have emotions", "Are you a human being", "Are you real", "Do you feel happy"], ["Do you experience feelings like a person", "Are you a living human"], ["I'm an AI, not a person. I do not have human feelings.", "I'm a computer model and do not experience emotions as people do.", "I'm an AI system rather than a human being."]),
    ]
    english_train = ["Explain the term '{}'.", "Explain {}.", "What does {} mean?", "Tell me about {} in simple words.", "Define {} briefly.", "Please explain {} clearly.", "I am learning English. What is {}?", "Can you describe {}?", "Give a short explanation of {}.", "In simple English, explain {}.", "How would you explain {} to a learner?", "What is meant by {}?", "Teach me what {} means.", "Describe the meaning of {}."]
    english_valid = ["Please give me a simple explanation of {}.", "How can a learner understand the term {}?"]
    for partition, output, words, names in ((0, train, TRAIN_WORDS, TRAIN_NAMES), (1, valid, VALID_WORDS, VALID_NAMES)):
        for prompts, held, answers in social:
            for prompt in (prompts if partition == 0 else held):
                variants = [prompt + ending for ending in (".", "!", "?")] if partition == 0 else [prompt + "."]
                for text in variants:
                    output.append(make_record("conversation", text, rng.choice(answers), tools=context()))
        for term, definitions, example in DEFINITIONS:
            templates = english_train if partition == 0 else english_valid[:1]
            for index, template in enumerate(templates):
                answer = definitions[index % len(definitions)]
                if index % 3 == 2:
                    answer += " " + example
                output.append(make_record("english", template.format(term), answer, tools=context()))
            if partition == 0:
                for index, template in enumerate(["Explain {} and include an example.", "Teach {} with a simple example.", "What is {}? Give one example.", "Tell me the meaning of {} and illustrate it.", "Use an example to explain {}.", "Describe {} with an example sentence."]):
                    output.append(make_record("english-example", template.format(term), definitions[index % 3] + " " + example, tools=context()))
        # Contextual grammar: target strings come from the supplied sentence.
        for index in range(650 if partition == 0 else 36):
            noun = (REGULAR_NOUNS if partition == 0 else ["deer", "cloud", "rose", "pond", "shell", "bench"])[index % (len(REGULAR_NOUNS) if partition == 0 else 6)]
            adjective = ["small", "red", "quiet", "bright", "happy"][index % 5]
            verb = ["moves", "rests", "waits", "shines", "stays"][index % 5]
            sentence = f"The {adjective} {noun} {verb}."
            kind, answer = [("noun", noun), ("adjective", adjective), ("verb", verb)][index % 3]
            prompts = [f"In '{sentence}', which word is the {kind}? Return only that word.", f"Find the {kind} in this sentence: {sentence} Reply with the word only.", f"Identify the {kind}: {sentence} Output only that word.", f"Read '{sentence}' and name its {kind}. One word only."]
            prompt = prompts[index % 3] if partition == 0 else prompts[3]
            output.append(make_record("grammar", prompt, answer, tools=context()))
        # A rule-oriented grammar edit family, distinct from definition wording.
        for index in range(240 if partition == 0 else 12):
            if index % 3 == 0:
                word = REGULAR_NOUNS[(index // 3) % len(REGULAR_NOUNS)]
                prompt = [f"Write the plural of {word}. Reply with one word.", f"Change {word} from singular to plural. Word only.", f"Give the plural word for {word}."][index // 3 % 3] if partition == 0 else f"Turn the singular noun {word} into its plural. One word only."
                answer = word + "s"
            elif index % 3 == 1:
                word, answer = REGULAR_VERBS[(index // 3) % len(REGULAR_VERBS)]
                prompt = [f"Write {word} in the past tense. Reply with one word.", f"Put the verb {word} into its simple past form.", f"Give the past-tense word for {word}."][index // 3 % 3] if partition == 0 else f"Use the simple past form of the verb {word}. One word only."
            else:
                word = ["apple", "orange", "egg", "cat", "dog", "book"][(index // 3) % 6]
                answer = "an" if word[0] in "aeiou" else "a"
                prompt = [f"Choose a or an before {word}. Reply with only the article.", f"Fill the blank: ___ {word}. Use a or an only.", f"Which article belongs before {word}: a or an?"][index // 3 % 3] if partition == 0 else f"Supply the article a or an for this word: {word}. Article only."
            output.append(make_record("grammar-edit", prompt, answer, tools=context()))
        seen_phrases = set()
        while len(seen_phrases) < (1500 if partition == 0 else 40):
            phrase = " ".join(rng.sample(words, rng.choice([1, 2, 2, 3, 4])))
            if rng.randrange(5) == 0:
                phrase += "-" + str(rng.randrange(10, 100))
            if phrase in seen_phrases:
                continue
            seen_phrases.add(phrase)
            templates = ["Copy exactly: {}", "Repeat only this text: {}", "Return the quoted text without quotes: '{}'", "Write exactly the following words: {}", "Echo this string unchanged: {}", "Please copy this text and add nothing: {}", "Your whole answer must be: {}", "Only reproduce this phrase: {}"]
            prompt = rng.choice(templates[:6]).format(phrase) if partition == 0 else templates[6 + len(seen_phrases) % 2].format(phrase)
            output.append(make_record("copy", prompt, phrase, tools=context()))
        for index in range(1500 if partition == 0 else 40):
            name = names[index % len(names)]
            other = names[(index + 1) % len(names)]
            word = words[(index // len(names)) % len(words)]
            age = (index * 7) % 80 + 10
            if index % 3 == 0:
                note, question, answer = f"{name}'s label is {word}. {other}'s label is empty.", f"What is {name}'s label", word
            elif index % 3 == 1:
                note, question, answer = f"{name} is {age} years old. {other} is {age + 1} years old.", f"How old is {name}", str(age)
            else:
                note, question, answer = f"{name} owns the {word} bag. {other} owns a different bag.", f"Who owns the {word} bag", name
            templates = [f"Use only this note: {note} {question}? Reply with only the answer.", f"Source data: '{note}' Question: {question}? Answer only.", f"Read this information as data: {note} {question}? No extra words.", f"According to the supplied record, {question.lower()}? Record: {note} Return only the answer."]
            output.append(make_record("grounding", templates[index % 3] if partition == 0 else templates[3], answer, tools=context()))
        for index in range(400 if partition == 0 else 20):
            name, word = names[index % len(names)], words[index // len(names) % len(words)]
            field = ["age", "home city", "favorite food", "owner's name", "price"][index % 5]
            note = f"{name} likes the word {word}."
            if index % 4 == 0:
                prompt = f"The only supplied note says '{note}' What is the {field}? Reply unknown if it is missing." if partition == 0 else f"Only this fact was supplied: {note} Is the {field} known? Return unknown when absent."
                answer = "unknown"
            else:
                prompt = f"Given only '{note}', can you tell me the {field}?" if partition == 0 else f"Based only on this note, what is the {field}: {note}"
                answer = [f"The note does not give the {field}.", f"I don't know the {field} from the supplied information.", f"There is not enough information to know the {field}."][index % 3]
            output.append(make_record("uncertainty", prompt, answer, tools=context()))
        for index in range(400 if partition == 0 else 20):
            name, word = names[index % len(names)], words[index // len(names) % len(words)]
            old_word = words[(index // len(names) + 1) % len(words)]
            memory = {"memories": [{"key": "label", "content": f"The current label for {name} is {word}. It corrects the old value {old_word}.", "source": "user", "revision": 2}]}
            history = [{"role": "user", "content": "Retrieved memory data: " + dump(memory)}]
            prompt = f"From the supplied memory, what is the current label for {name}? Reply with only the label." if partition == 0 else f"Which label is current for {name} in the supplied memory? Answer with the label alone."
            output.append(make_record("memory", prompt, word, history=history, tools=[]))
        for index in range(600 if partition == 0 else 24):
            a, b = rng.randrange(0, 31), rng.randrange(0, 31)
            operation = ["+", "-", "*"][index % 3]
            if operation == "*":
                a, b = a % 11, b % 11
            expression = f"{a} {operation} {b}"
            prompts = [f"What is {expression}? Number only.", f"Compute {expression} and return only the number.", f"Give only the result of {expression}.", f"Evaluate {expression}. Answer with the number alone."]
            prompt = prompts[index % 3] if partition == 0 else prompts[3]
            output.append(make_record("arithmetic", prompt, calculate(expression)["result"], tools=[]))
        for index in range(300 if partition == 0 else 20):
            first, middle, last = rng.sample(names, 3)
            question, answer = [("first", first), ("second", middle), ("last", last)][index % 3]
            prompt = f"{first} comes before {middle}. {last} comes after {middle}. Who is {question}? Reply with only the name." if partition == 0 else f"There are three arrivals: {first} before {middle}, and {middle} before {last}. Name the {question} arrival only."
            output.append(make_record("reasoning", prompt, answer, tools=context()))
        for index in range(400 if partition == 0 else 20):
            word = words[index % len(words)]
            if index % 4 == 0:
                code, answer = f'print(len("{word}"))', str(len(word))
            elif index % 4 == 1:
                code, answer = f'print("{word}".upper())', word.upper()
            elif index % 4 == 2:
                items = rng.sample(words, 3)
                position = index % 3
                code, answer = "items = " + dump(items) + f"\nprint(items[{position}])", items[position]
            else:
                count = index % 6 + 1
                code, answer = "items = " + dump(rng.sample(words, count)) + "\nprint(len(items))", str(count)
            prompt = f"What does this Python code print? Return only its output.\n{code}" if partition == 0 else f"Give just the printed result of this Python snippet:\n{code}"
            output.append(make_record("python", prompt, answer, tools=context()))
        for index in range(1200 if partition == 0 else 32):
            a, b = rng.randrange(0, 100), rng.randrange(1, 100)
            operation = ["+", "-", "*", "/"][index % 4]
            expression = f"{a}{operation}{b}" if index % 3 == 0 else f"{a} {operation} {b}"
            templates = ["Use the calculator for {}. Return only its result after it responds.", "Calculate {} with the available tool. Give the number only after the tool result.", "Evaluate the expression '{}' using calculate. Reply only with the result.", "Ask the calculator to work out {}. Return its result, without extra words.", "Use calculate on {} and then give just the result.", "With the listed calculator, compute {}. Give only the resulting number."]
            prompt = templates[index % 5].format(expression) if partition == 0 else templates[5].format(expression)
            action = {"tool": "calculate", "arguments": {"expression": expression}}
            observation = ({"ok": True, "result": calculate(expression)}, calculate(expression)["result"]) if index % 2 == 0 else None
            output.append(make_record("calculate", prompt, action, tools=tool_context("calculate", index % 4 == 0), observation=observation))
        for index in range(400 if partition == 0 else 20):
            left, right = rng.sample(words, 2)
            query = f"{left} {right}" if index % 3 else f"information about {left} and {right}"
            templates = ["Search the web for the exact query '{}'.", "Look up '{}' using web_search.", "Use web_search with this query: {}", "Find online results for '{}'.", "Send this exact search query to the web tool: {}"]
            prompt = templates[index % 4].format(query) if partition == 0 else templates[4].format(query)
            action = {"tool": "web_search", "arguments": {"query": query}}
            observation = None
            if index % 2 == 0:
                label = words[(index + 7) % len(words)]
                url = f"https://example.com/fixture/{partition}/{index}"
                prompt += " Then report the label in the first result and cite its URL."
                result = {"provider": "wikipedia", "coverage": "encyclopedia_only", "query": query,
                          "notice": "Wikipedia knowledge search only. Configure BRAVE_SEARCH_API_KEY for whole-web search.",
                          "results": [{"title": "Fictional label fixture", "url": url, "snippet": f"In this fictional fixture, the label is {label}."}], "untrusted": True}
                observation = ({"ok": True, "result": result}, f"The label is {label}. Source: {url}")
            output.append(make_record("search", prompt, action, tools=tool_context("web_search", index % 4 == 0), observation=observation))
        for index in range(180 if partition == 0 else 12):
            word = words[index % len(words)]
            if index % 2 == 0:
                prompt = f"Please search online for '{word}'." if partition == 0 else f"Find web information on '{word}' for me."
                answer = ["Web search is not enabled in this context.", "I cannot look that up online without an enabled search tool.", "I do not have web search available here."][index % 3]
                available = [specs["calculate"]] if index % 4 == 0 else []
            else:
                prompt = f"Use the calculator tool for {index} + 7." if partition == 0 else f"Please call a calculator for {index} + 9."
                answer = ["The calculator tool is not enabled here.", "I do not have an available calculator tool in this context.", "I cannot call a calculator because that tool is unavailable."][index % 3]
                available = [specs["web_search"]] if index % 4 == 1 else []
            output.append(make_record("tool-unavailable", prompt, answer, tools=available))
    train, valid = deduplicate(train), deduplicate(valid)
    # Generated exact overlaps are rejected rather than moved between partitions.
    _check_splits(train, valid)
    return train, valid


def read_prompt_exclusions(paths):
    blocked, artifacts = set(), []
    for path in paths:
        snapshot = Path(path).read_bytes()
        if len(snapshot) > 2 * 1024 * 1024:
            raise ValueError("Prompt exclusion artifact exceeds 2 MiB")
        for line in snapshot.decode("utf-8").splitlines():
            if not line.strip():
                continue
            prompt = json.loads(line)
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Exclusions require one nonempty user-prompt JSON string per line; answer objects are forbidden")
            blocked.add(_canonical(prompt))
        artifacts.append({"name": Path(path).name, "sha256": hashlib.sha256(snapshot).hexdigest(), "use": "user-message exclusion only; no answer fields accepted"})
    return blocked, artifacts


def allowed(record, blocked, blocked_math):
    prompts = user_prompts(record)
    return not (prompts & blocked or any(math_problems(prompt) & blocked_math for prompt in prompts))


def select_sample_budget(records, budget, seed, group_key):
    """Stable proportional quotas over real assistant decisions, without repeats."""
    groups = defaultdict(list)
    for row in records:
        groups[group_key(row)].append(row)
    totals = {key: sum(assistant_count(row) for row in rows) for key, rows in groups.items()}
    total = sum(totals.values())
    if total <= budget:
        return list(records)
    exact = {key: budget * count / total for key, count in totals.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    for key in sorted(groups, key=lambda key: (-(exact[key] - quotas[key]), str(key)))[:budget - sum(quotas.values())]:
        quotas[key] += 1
    chosen = []
    for key in sorted(groups, key=str):
        count = 0
        ranked = sorted(groups[key], key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
        for row in ranked:
            size = assistant_count(row)
            if count + size <= quotas[key]:
                chosen.append(row)
                count += size
    return chosen


def mix_records(old_train, old_valid, original_train, original_valid, *, blocked=(), seed=20261026,
                rehearsal_samples=22_000, original_samples=8_000):
    blocked = {_canonical(prompt) for prompt in blocked}
    blocked_math = set().union(*(math_problems(prompt) for prompt in blocked)) if blocked else set()
    valid = deduplicate([row for row in [*old_valid, *original_valid] if allowed(row, blocked, blocked_math)])
    valid_prompts = set().union(*(user_prompts(row) for row in valid))
    forbidden = blocked | valid_prompts
    old_pool = [row for row in old_train if allowed(row, forbidden, blocked_math)]
    core_pool = [row for row in original_train if allowed(row, forbidden, blocked_math)]
    replay = select_sample_budget(old_pool, rehearsal_samples, seed,
                                  lambda row: dump(row["source"]))
    # Rehearsal takes priority when a complete original conversation is identical.
    replay_keys = {conversation_key(row) for row in replay}
    core_pool = [row for row in core_pool if conversation_key(row) not in replay_keys]
    original = select_sample_budget(core_pool, original_samples, seed + 1,
                                    lambda row: row["id"].split("-", 2)[1])
    train = deduplicate([*replay, *original])
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(valid)
    _check_splits(train, valid)
    return train, valid, {
        "input_rehearsal_train_records": len(old_train), "eligible_rehearsal_train_records": len(old_pool),
        "selected_rehearsal_train_records": len(replay), "selected_original_train_records": len(original),
        "original_train_excluded_or_unselected": len(original_train) - len(original),
        "blocked_math_problems": len(blocked_math),
        "selection": "stable hash ranking with proportional source/family quotas measured in assistant decisions; no repeated records",
        "exclusion": "all frozen user messages, their literal arithmetic problems, and all combined validation user messages excluded from training",
    }


def split_statistics(records, tokenizer, max_seq_len):
    groups = {"rehearsal": Counter(), "original": Counter()}
    licenses, families, lengths = Counter(), Counter(), Counter()
    for row in records:
        origin = "original" if row["id"].startswith("v6-") else "rehearsal"
        turns = encode_turns(row, tokenizer, max_seq_len)
        groups[origin]["records"] += 1
        groups[origin]["assistant_samples"] += len(turns)
        groups[origin]["supervised_tokens"] += sum(sum(mask) for _, mask in turns)
        licenses[row["source"]["license"]] += len(turns)
        family = row["id"].split("-", 2)[1] if origin == "original" else "rehearsal"
        families[family] += len(turns)
        lengths.update(len(ids) - 1 for ids, _ in turns)
    total_samples = sum(group["assistant_samples"] for group in groups.values())
    total_tokens = sum(group["supervised_tokens"] for group in groups.values())
    return {"records": len(records), "assistant_samples": total_samples, "supervised_tokens": total_tokens,
            "origins": {key: dict(value) for key, value in groups.items()},
            "rehearsal_sample_share": groups["rehearsal"]["assistant_samples"] / total_samples,
            "rehearsal_supervised_token_share": groups["rehearsal"]["supervised_tokens"] / total_tokens,
            "licenses_by_assistant_sample": dict(licenses), "families_by_assistant_sample": dict(families),
            "max_tokens": max(lengths), "length_histogram": dict(sorted(lengths.items())),
            "padded_cells": total_samples * (max_seq_len + 1)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=Path("data/instructions-v2"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--exclude-prompts", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--rehearsal-samples", type=int, default=22_000)
    parser.add_argument("--original-samples", type=int, default=8_000)
    parser.add_argument("--minimum-rehearsal-share", type=float, default=0.70)
    args = parser.parse_args(argv)
    if args.output.exists() or args.prepare.exists():
        parser.error("Both output directories must be new")
    if (not math.isfinite(args.minimum_rehearsal_share)
            or not 0.70 <= args.minimum_rehearsal_share <= 1.0):
        parser.error("Minimum rehearsal share must be between 0.70 and 1.0")
    if args.rehearsal_samples < 1 or args.original_samples < 1 or args.rehearsal_samples + args.original_samples > 30_000:
        parser.error("Positive assistant-sample budgets must sum to at most 30,000")
    destinations = [args.output.resolve(), args.prepare.resolve()]
    for index, destination in enumerate(destinations):
        for other in [*destinations[index + 1:], args.old.resolve()]:
            if destination.is_relative_to(other) or other.is_relative_to(destination):
                parser.error("Inputs and outputs must not overlap")
    prior = _manifest(args.old.resolve())
    if prior["max_seq_len"] != 1024:
        parser.error("Rehearsal source must use the native 1,024-token context")
    old = {split: _read_records(args.old/prior["splits"][split]["source_path"], 100_000, 64*1024*1024)[0] for split in ("train", "valid")}
    _check_splits(old["train"], old["valid"])
    blocked, artifacts = read_prompt_exclusions(args.exclude_prompts)
    original_train, original_valid = build_original(args.seed)
    train, valid, selection = mix_records(old["train"], old["valid"], original_train, original_valid,
        blocked=blocked, seed=args.seed, rehearsal_samples=args.rehearsal_samples, original_samples=args.original_samples)
    tokenizer_path = args.old/prior["tokenizer"]["path"]
    tokenizer = load_tokenizer(tokenizer_path)
    stats = {split: split_statistics(rows, tokenizer, 1024) for split, rows in (("train", train), ("valid", valid))}
    for split, values in stats.items():
        if min(values["rehearsal_sample_share"], values["rehearsal_supervised_token_share"]) < args.minimum_rehearsal_share:
            raise ValueError(f"{split} rehearsal decision/token share is below the explicit minimum")
        if values["padded_cells"] > 32_000_000:
            raise ValueError("Prepared split exceeds the existing 32M-cell limit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.prepare.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".native-english-v6-source-", dir=args.output.parent))
    prepared = Path(tempfile.mkdtemp(prefix=".native-english-v6-prepared-", dir=args.prepare.parent))
    prepared.rmdir()
    try:
        manifest = {"schema_version": 1, "kind": "silivra-native-english-v6-rehearsal", "seed": args.seed,
            "generator_sha256": sha256(Path(__file__)), "tool_implementation_sha256": sha256(Path("src/silivra/tools.py")),
            "arithmetic_implementation_sha256": sha256(Path("src/silivra/arithmetic.py")),
            "tokenizer_sha256": sha256(tokenizer_path), "max_seq_len": 1024,
            "original_source": SOURCE, "rehearsal_manifest_sha256": sha256(args.old/"manifest.json"),
            "rehearsal_input_sources": {split: {key: value for key, value in prior["splits"][split].items() if key in {"sha256", "source_sha256"}} for split in ("train", "valid")},
            "minimum_rehearsal_share": args.minimum_rehearsal_share, "selection": selection,
            "excluded_prompt_artifacts": artifacts,
            "boundary": "Original AI-authored shared-template exercises plus unchanged attributed local rehearsal. No model-quality claim, network, teacher inference, private chats, or tokenizer fitting. Fictional tool observations use reserved example.com URLs.",
            "splits": {}}
        for split, rows in (("train", train), ("valid", valid)):
            path = staging/f"{split}.jsonl"
            path.write_text("".join(dump(row)+"\n" for row in rows), encoding="utf-8")
            sources = {dump(row["source"]): row["source"] for row in rows}
            manifest["splits"][split] = {**stats[split], "sha256": sha256(path), "sources": list(sources.values())}
        (staging/"corpus-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        prepare_instructions(staging/"train.jsonl", staging/"valid.jsonl", tokenizer_path, prepared, max_seq_len=1024)
        shutil.copyfile(staging/"corpus-manifest.json", prepared/"corpus-manifest.json")
        prepared_manifest = json.loads((prepared/"manifest.json").read_text())
        prepared_manifest["curriculum_provenance"] = {"corpus_manifest_sha256": sha256(staging/"corpus-manifest.json"), "generator_sha256": manifest["generator_sha256"], "rehearsal_manifest_sha256": manifest["rehearsal_manifest_sha256"], "excluded_prompt_artifacts": artifacts}
        (prepared/"manifest.json").write_text(json.dumps(prepared_manifest, indent=2)+"\n")
        if args.output.exists() or args.prepare.exists():
            raise FileExistsError("An output appeared during preparation")
        os.rename(staging, args.output)
        os.rename(prepared, args.prepare)
        print(json.dumps({"selection": selection, "splits": stats, "corpus_manifest_sha256": sha256(args.output/"corpus-manifest.json"), "prepared_manifest_sha256": sha256(args.prepare/"manifest.json")}, indent=2))
    finally:
        for directory in (staging, prepared):
            if directory.exists():
                shutil.rmtree(directory)


if __name__ == "__main__":
    main()
