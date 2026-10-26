#!/usr/bin/env python3
"""Build original AI-authored English/tool exercises; no network or model teacher.

This deliberately small synthetic corpus teaches narrow patterns. Validation
paraphrases and operands are reserved before generation, not taken from outputs.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from silivra.agent import build_system_prompt
from silivra.arithmetic import calculate
from silivra.checkpoint import sha256
from silivra.instructions import _check_splits, prepare_instructions, validate_record
from silivra.tools import ToolRegistry

SOURCE = {"name": "Silivra original native English curriculum v5 (AI-authored)",
          "author": "Silivra contributors with Codex; original synthetic examples",
          "license": "MIT", "url": "https://github.com/zeousky/Silivra", "revision": "native-english-v5-runtime"}
SOCIAL = [
(["hi", "hello", "hey", "good morning", "good evening", "hello Silivra", "hey there", "greetings"],
 ["hello there Silivra", "good afternoon friend"], "Hello! How can I help you?"),
(["how are you", "how are you doing", "how is it going", "are you okay", "how do you feel"],
 ["how are things with you", "are things going well"], "I'm here and ready to help. How can I help you?"),
(["who are you", "what is your name", "introduce yourself", "tell me your name", "what are you"],
 ["what should I call you", "tell me who you are please"], "I'm Silivra, a small experimental AI trained from scratch."),
(["thanks", "thank you", "thanks for helping", "thank you very much", "that helped", "thanks Silivra"],
 ["I appreciate your help", "many thanks for the help"], "You're welcome!"),
(["bye", "goodbye", "see you", "see you later", "good night", "I am leaving now"],
 ["time for me to leave", "talk with you later"], "Goodbye! Take care."),
(["what can you do", "what are your limitations", "can you make mistakes", "are you reliable", "can I trust your answers"],
 ["could your answer be wrong", "tell me your limitations please"], "I'm an experimental local AI. My answers can be wrong, so check important information."),
(["help me", "I need help", "can you help", "I have a question", "help please"],
 ["could you help me please", "please give me some help"], "What would you like help with?"),
(["are you human", "are you a person", "do you have feelings", "are you alive"],
 ["are you a real person", "do you feel emotions"], "I'm an AI, not a person. I do not have human feelings."),
(["do you use other models", "are you a hosted model", "are you using an API model", "were you trained from scratch"],
 ["is your model your own", "do you run your own weights"], "I use Silivra's own model and tokenizer, trained from scratch."),
]
# Authored elementary explanations, not retrieved claims or a broad knowledge base.
ENGLISH = [
("noun", "A noun names a person, place, thing, or idea."),
("verb", "A verb describes an action or a state."),
("adjective", "An adjective describes a noun."),
("adverb", "An adverb describes a verb, adjective, or another adverb."),
("pronoun", "A pronoun can stand in place of a noun, such as he, she, or they."),
("sentence", "A sentence expresses a complete thought."),
("question", "A question asks for information and usually ends with a question mark."),
("plural", "A plural word refers to more than one thing."),
("singular", "A singular word refers to one thing."),
("past tense", "The past tense describes something that happened before now."),
("present tense", "The present tense describes something happening now or regularly."),
("future tense", "The future tense describes something that will happen later."),
("paragraph", "A paragraph is a group of sentences about one main idea."),
("synonym", "A synonym is a word with a similar meaning to another word."),
("antonym", "An antonym is a word with the opposite meaning to another word."),
("capital letter", "A capital letter begins a sentence or a proper name."),
("full stop", "A full stop marks the end of a sentence."),
("comma", "A comma marks a short pause or separates items in a list."),
("English", "English is a language used for speaking, reading, and writing."),
("Python", "Python is a programming language."),
("function", "A function is a named block of code that performs a task."),
("variable", "A variable is a name that refers to a value in a program."),
("loop", "A loop repeats a block of code."),
("list", "A list stores an ordered collection of items."),
("dictionary", "A dictionary stores values under keys."),
("computer", "A computer follows instructions to process data."),
("web search", "Web search finds pages or information on the internet."),
("source", "A source is where information comes from."),
("evidence", "Evidence is information that supports or challenges a claim."),
("uncertainty", "Uncertainty means that an answer is not known with confidence."),
]
WORDS_TRAIN = "apple bird blue boat book cat dog door fish flower fox green house lake moon red river sky star sun tea tree water white wind yellow".split()
WORDS_VALID = "amber beach cloud deer field forest grass hill rain rose snow stone table town wood".split()
TOPICS_TRAIN = ["English grammar", "English language", "Python programming", "solar system", "planet Earth", "water cycle", "ocean", "forest", "photosynthesis", "human body", "history of computers", "public libraries", "music", "mathematics", "volcano", "weather", "language learning", "reading", "writing", "biology", "birds", "animals", "plants", "science", "technology", "computer", "internet", "geography", "chemistry", "physics"]
TOPICS_VALID = ["rainforest", "Moon", "electricity", "astronomy", "English spelling", "computer memory"]


def dump(x): return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def schemas():
    registry = ToolRegistry(Path.cwd(), allow_network=True)
    selected = {s["function"]["name"]: s for s in registry.schemas()}
    return selected


def record(kind, prompt, action, tools=(), observation=None):
    messages = [{"role":"system", "content":build_system_prompt(list(tools))},
                {"role":"user", "content":prompt}, {"role":"assistant", "content":dump(action)}]
    if observation is not None:
        result, final = observation
        messages += [{"role":"tool", "name":action["tool"], "content":dump(result)},
                     {"role":"assistant", "content":dump({"final":str(final)})}]
    ident = hashlib.sha256(dump(messages).encode()).hexdigest()[:20]
    return validate_record({"id":f"v5-{kind}-{ident}","source":dict(SOURCE),"messages":messages})


def build_splits(seed=20261025, excluded_prompts=()):
    if type(seed) is not int or not 0 <= seed < 2**32: raise ValueError("Invalid seed")
    rng = random.Random(seed)
    specs = schemas()
    contexts = [[], [specs["calculate"]], [specs["web_search"]], [specs["calculate"], specs["web_search"]]]
    train, valid = [], []
    for partition, rows in ((0,train),(1,valid)):
        for training_prompts, held_prompts, answer in SOCIAL:
            for prompt in (training_prompts if partition==0 else held_prompts):
                for punctuation in (["", "!", "?"] if partition==0 else ["."]):
                    for context in contexts:
                        rows.append(record("conversation",prompt+punctuation,{"final":answer},context))
        for word, definition in ENGLISH:
            templates = (["What is {}?", "Explain {}.", "Define {}.", "Tell me what {} means.", "What does {} mean?"] if partition==0 else ["Please explain {} in simple English."])
            for template in templates:
                for context in (contexts if partition==0 else [[]]):
                    rows.append(record("english",template.format(word),{"final":definition},context))
        words = WORDS_TRAIN if partition==0 else WORDS_VALID
        for index, word in enumerate(words):
            for context in contexts:
                rows.append(record("grounding",f"The label says {word}. What does the label say? Reply with only the word.",{"final":word},context))
                rows.append(record("copy",f"Copy exactly: {word}",{"final":word},context))
        topics = TOPICS_TRAIN if partition==0 else TOPICS_VALID
        for topic in topics:
            for template in (["Search the web for {}.", "Look up {} online.", "Find information about {}.", "Use web search to research {}."] if partition==0 else ["Please search online for {}."]):
                for context in [contexts[2],contexts[3]]:
                    action = {"tool":"web_search","arguments":{"query":topic}}
                    observation = {"ok": True, "result": {"provider":"wikipedia", "coverage":"encyclopedia_only", "query":topic,
                        "notice":"Wikipedia knowledge search only. Configure BRAVE_SEARCH_API_KEY for whole-web search.",
                        "results":[{"title":topic,"url":"https://example.com/search-result","snippet":"Information about " + topic + "."}],"untrusted":True}}
                    answer = "Information about " + topic + ". Source: https://example.com/search-result"
                    rows.append(record("search",template.format(topic),action,context,(observation,answer)))
                    # Uncertainty is honest when search is absent.
                rows.append(record("no-search",template.format(topic),{"final":"I need web search enabled to look that up."}))
        # Numeric pairs are reserved as complete problems, including their observations.
        for a in range(20):
            for b in range(10):
                if int((a*3+b)%7==0)!=partition: continue
                for context in [contexts[1],contexts[3]]:
                    prompt=f"Calculate {a} + {b}."
                    action={"tool":"calculate","arguments":{"expression":f"{a} + {b}"}}
                    rows.append(record("calculate",prompt,action,context,({"ok":True,"result":calculate(f"{a} + {b}")},a+b)))
                rows.append(record("arithmetic",f"What is {a} + {b}? Reply with only the number.",{"final":str(a+b)}))
                rows.append(record("comparison",f"Is {a} greater than {b}? Answer yes or no.",{"final":"yes" if a>b else "no"}))
        # Missing information does not become a guessed fact.
        for word in words:
            rows.append(record("honesty",f"The note says the label is {word}. What is the owner's name? Reply unknown if it is missing.",{"final":"unknown"}))
    deny={" ".join(p.split()) for p in excluded_prompts}
    # System tool variants intentionally share user prompts within a split.
    train=[r for r in train if not any(" ".join(m["content"].split()) in deny for m in r["messages"] if m["role"]=="user")]
    def unique(rows): return list({dump(r["messages"]):r for r in rows}.values())
    train,valid=unique(train),unique(valid)
    rng.shuffle(train); rng.shuffle(valid)
    _check_splits(train,valid)
    return train,valid


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path);p.add_argument("--prepare",required=True,type=Path)
    p.add_argument("--tokenizer",required=True,type=Path);p.add_argument("--exclude-suite",type=Path,action="append",default=[])
    args=p.parse_args(argv)
    if args.output.exists() or args.prepare.exists(): p.error("Output and prepared directories must be absent")
    if args.output.resolve().is_relative_to(args.prepare.resolve()) or args.prepare.resolve().is_relative_to(args.output.resolve()): p.error("Outputs must not overlap")
    exclusions=[]
    for path in args.exclude_suite:
        for line in path.read_text().splitlines():
            if line.strip():
                case=json.loads(line)
                exclusions += [m["content"] for m in case.get("messages",[{"role":"user","content":case.get("prompt","")}]) if m["role"]=="user"]
    train,valid=build_splits(excluded_prompts=exclusions)
    args.output.mkdir(parents=True)
    manifest={"kind":"silivra-native-english-v5-runtime","source":SOURCE,"seed":20261025,"generator_sha256":sha256(Path(__file__)),"tokenizer_sha256":sha256(args.tokenizer),"boundary":"AI-authored narrow shared-template training/validation; not independent intelligence evidence","exclusions":[{"name":x.name,"sha256":sha256(x),"use":"user-prompt exclusion only"} for x in args.exclude_suite],"splits":{}}
    for split,rows in (("train",train),("valid",valid)):
        path=args.output/f"{split}.jsonl";path.write_text("".join(dump(r)+"\n" for r in rows))
        manifest["splits"][split]={"records":len(rows),"sha256":sha256(path),"families":dict(Counter(r["id"].split("-")[1] for r in rows))}
    (args.output/"corpus-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    prepare_instructions(args.output/"train.jsonl",args.output/"valid.jsonl",args.tokenizer,args.prepare,max_seq_len=1024)
    print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
