# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic generators for the default item banks.

Every default item is generated procedurally from a fixed seed, so the banks
need no dataset download and carry no third-party licence.  The shipped JSONL
files under ``banks/items`` are the canonical banks; ``regenerate()`` reproduces
them byte for byte (a reference test checks this).
"""

from __future__ import annotations

import random

NO_THINK = {"enable_thinking": False}
SEEDS = {"math_short": 11, "multiple_choice": 23, "tool_calling": 37, "instruction_following": 41,
         "code_unit": 53, "needle_16k": 67}
COUNTS = {"math_short": 140, "multiple_choice": 140, "tool_calling": 90, "instruction_following": 100,
          "code_unit": 30, "needle_16k": 8}

NAMES = ["Ava", "Ben", "Chloe", "Daniel", "Elena", "Farid", "Grace", "Hiro", "Isla", "Jonas", "Kofi", "Lena",
         "Mateo", "Nadia", "Omar", "Priya", "Quinn", "Rosa", "Sami", "Tara"]


def _user(text: str) -> list[dict]:
    return [{"role": "user", "content": text}]


# --------------------------------------------------------------------------- math
def gen_math_short(rng: random.Random, count: int) -> list[dict]:
    items = []
    for i in range(count):
        if i % 2 == 0:  # answer only, short output
            kind = rng.randrange(5)
            if kind == 0:
                a, b, c = rng.randint(10, 99), rng.randint(2, 19), rng.randint(2, 19)
                q, ans = f"Compute {a} + {b} * {c}.", a + b * c
            elif kind == 1:
                a, b, c, d = rng.randint(2, 30), rng.randint(2, 30), rng.randint(2, 9), rng.randint(1, 20)
                q, ans = f"Compute ({a} + {b}) * {c} - {d}.", (a + b) * c - d
            elif kind == 2:
                a, b, c = rng.randint(11, 49), rng.randint(3, 19), rng.randint(5, 99)
                q, ans = f"What is {a} * {b} - {c}?", a * b - c
            elif kind == 3:
                p = rng.choice([10, 20, 25, 50, 75])
                b = rng.choice([40, 80, 120, 200, 240, 360, 400, 600, 800])
                q, ans = f"What is {p} percent of {b}?", p * b // 100
            else:
                a, b = rng.randint(50, 500), rng.randint(3, 13)
                q, ans = f"What is the remainder when {a} is divided by {b}?", a % b
            text = q + " Reply with only the final number."
            items.append({"messages": _user(text), "max_new_tokens": 24,
                          "grader": {"type": "numeric", "answer": ans, "tol": 1e-6}, "variant": "answer_only"})
        else:  # brief explanation, medium output
            kind = rng.randrange(5)
            name = rng.choice(NAMES)
            if kind == 0:
                obj = rng.choice(["marbles", "stickers", "cards", "stamps"])
                a, b, c = rng.randint(10, 60), rng.randint(2, 6), rng.randint(3, 12)
                d = rng.randint(1, a)
                q = (f"{name} has {a} {obj}. {name} buys {b} packs with {c} {obj} in each pack, then gives away "
                     f"{d} {obj}. How many {obj} does {name} have now?")
                ans = a + b * c - d
            elif kind == 1:
                s, h, s2, h2 = rng.randint(40, 120), rng.randint(2, 5), rng.randint(40, 120), rng.randint(1, 4)
                q = (f"A train travels at {s} km per hour for {h} hours and then at {s2} km per hour for {h2} "
                     f"hours. How many kilometres does it travel in total?")
                ans = s * h + s2 * h2
            elif kind == 2:
                p, n = rng.randint(2, 9), rng.randint(2, 8)
                bill = rng.choice([50, 100])
                q = (f"A shop sells notebooks for {p} dollars each. {name} buys {n} notebooks and pays with a "
                     f"{bill} dollar bill. How much change, in dollars, does {name} receive?")
                ans = bill - p * n
            elif kind == 3:
                r, c = rng.randint(5, 20), rng.randint(6, 15)
                t = rng.randint(1, r * c - 1)
                q = (f"A hall has {r} rows of chairs with {c} chairs in each row. {t} chairs are taken. "
                     f"How many chairs are free?")
                ans = r * c - t
            else:
                p, d = rng.randint(5, 25), rng.randint(3, 9)
                total = p * d + rng.randint(10, 200)
                q = (f"{name} reads {p} pages every day. The book has {total} pages. After {d} days, how many "
                     f"pages are left to read?")
                ans = total - p * d
            text = q + " Explain briefly, then finish with 'Answer: <number>'."
            items.append({"messages": _user(text), "max_new_tokens": 160,
                          "grader": {"type": "numeric", "answer": ans, "tol": 1e-6}, "variant": "brief_steps"})
    return items


# ---------------------------------------------------------------- multiple choice
CAPITALS = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Portugal", "Lisbon"),
    ("Japan", "Tokyo"), ("China", "Beijing"), ("India", "New Delhi"), ("Russia", "Moscow"), ("Canada", "Ottawa"),
    ("Australia", "Canberra"), ("Brazil", "Brasilia"), ("Argentina", "Buenos Aires"), ("Mexico", "Mexico City"),
    ("Egypt", "Cairo"), ("Kenya", "Nairobi"), ("Nigeria", "Abuja"), ("South Korea", "Seoul"),
    ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"), ("Turkey", "Ankara"),
    ("Greece", "Athens"), ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Finland", "Helsinki"),
    ("Denmark", "Copenhagen"), ("Poland", "Warsaw"), ("Austria", "Vienna"), ("Switzerland", "Bern"),
    ("Netherlands", "Amsterdam"), ("Belgium", "Brussels"), ("Ireland", "Dublin"), ("United Kingdom", "London"),
    ("Chile", "Santiago"), ("Peru", "Lima"), ("Colombia", "Bogota"), ("Venezuela", "Caracas"), ("Cuba", "Havana"),
    ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Saudi Arabia", "Riyadh"), ("Pakistan", "Islamabad"),
    ("Bangladesh", "Dhaka"), ("Philippines", "Manila"), ("Malaysia", "Kuala Lumpur"), ("New Zealand", "Wellington"),
    ("Hungary", "Budapest"), ("Czech Republic", "Prague"), ("Ukraine", "Kyiv"), ("Morocco", "Rabat"),
    ("Ethiopia", "Addis Ababa"), ("Ghana", "Accra"), ("Lebanon", "Beirut"), ("Jordan", "Amman"),
    ("Afghanistan", "Kabul"), ("Nepal", "Kathmandu"), ("Romania", "Bucharest"), ("Bulgaria", "Sofia"),
    ("Serbia", "Belgrade"), ("Croatia", "Zagreb"), ("Iceland", "Reykjavik"), ("Qatar", "Doha"),
    ("Algeria", "Algiers"), ("Tunisia", "Tunis"), ("Senegal", "Dakar"), ("Uganda", "Kampala"),
    ("Zimbabwe", "Harare"), ("Zambia", "Lusaka"), ("Angola", "Luanda"), ("Mongolia", "Ulaanbaatar"),
    ("Cambodia", "Phnom Penh"), ("Laos", "Vientiane"), ("Uruguay", "Montevideo"), ("Paraguay", "Asuncion"),
    ("Ecuador", "Quito"), ("Jamaica", "Kingston"),
]
ELEMENTS = [
    ("hydrogen", "H"), ("helium", "He"), ("lithium", "Li"), ("carbon", "C"), ("nitrogen", "N"), ("oxygen", "O"),
    ("fluorine", "F"), ("neon", "Ne"), ("sodium", "Na"), ("magnesium", "Mg"), ("aluminium", "Al"),
    ("silicon", "Si"), ("phosphorus", "P"), ("sulfur", "S"), ("chlorine", "Cl"), ("argon", "Ar"),
    ("potassium", "K"), ("calcium", "Ca"), ("iron", "Fe"), ("copper", "Cu"), ("zinc", "Zn"), ("silver", "Ag"),
    ("tin", "Sn"), ("gold", "Au"), ("mercury", "Hg"), ("lead", "Pb"), ("uranium", "U"), ("nickel", "Ni"),
    ("cobalt", "Co"), ("manganese", "Mn"), ("chromium", "Cr"), ("titanium", "Ti"), ("platinum", "Pt"),
    ("iodine", "I"), ("bromine", "Br"), ("barium", "Ba"), ("tungsten", "W"), ("krypton", "Kr"), ("xenon", "Xe"),
]
FACTS = [
    ("Which planet is closest to the Sun?", "Mercury", ["Venus", "Mars", "Earth", "Jupiter"]),
    ("Which planet is the largest in our solar system?", "Jupiter", ["Saturn", "Neptune", "Earth", "Mars"]),
    ("How many legs does a spider have?", "8", ["6", "10", "4", "12"]),
    ("How many legs does an insect have?", "6", ["8", "4", "10", "12"]),
    ("How many days are in a leap year?", "366", ["365", "364", "360", "367"]),
    ("How many hours are in a day?", "24", ["12", "36", "48", "20"]),
    ("At sea level, at what temperature in degrees Celsius does water boil?", "100", ["90", "80", "120", "212"]),
    ("At what temperature in degrees Celsius does water freeze?", "0", ["10", "-10", "32", "4"]),
    ("How many continents are there on Earth?", "7", ["5", "6", "8", "9"]),
    ("Which gas do plants absorb from the air for photosynthesis?", "Carbon dioxide",
     ["Oxygen", "Nitrogen", "Helium", "Hydrogen"]),
    ("What is the largest ocean on Earth?", "Pacific Ocean",
     ["Atlantic Ocean", "Indian Ocean", "Arctic Ocean", "Southern Ocean"]),
    ("How many sides does a hexagon have?", "6", ["5", "7", "8", "4"]),
    ("How many minutes are in an hour?", "60", ["100", "30", "90", "24"]),
    ("Which organ pumps blood through the human body?", "Heart", ["Liver", "Lungs", "Kidney", "Brain"]),
]
UNITS = [("minutes", "hours", 60), ("seconds", "minutes", 60), ("grams", "kilograms", 1000),
         ("centimetres", "metres", 100), ("millilitres", "litres", 1000), ("hours", "days", 24)]


def _mc(rng: random.Random, question: str, correct: str, distractors: list[str]) -> dict:
    opts = [correct] + rng.sample(distractors, 3)
    rng.shuffle(opts)
    letters = "ABCD"
    ans = letters[opts.index(correct)]
    body = "\n".join(f"{letters[i]}) {o}" for i, o in enumerate(opts))
    text = f"Question: {question}\n{body}\nAnswer with the letter of the correct option only."
    return {"messages": _user(text), "max_new_tokens": 8, "grader": {"type": "choice", "answer": ans}}


def gen_multiple_choice(rng: random.Random, count: int) -> list[dict]:
    items = []
    caps = list(CAPITALS)
    rng.shuffle(caps)
    els = list(ELEMENTS)
    rng.shuffle(els)
    facts = list(FACTS)
    plan = (["capital"] * 45 + ["element"] * 30 + ["fact"] * len(facts) + ["arith"] * 26 + ["unit"] * 25)
    plan = plan[:count]
    ci = ei = fi = 0
    for kind in plan:
        if kind == "capital":
            country, capital = caps[ci % len(caps)]
            ci += 1
            others = [c for _, c in CAPITALS if c != capital]
            items.append(_mc(rng, f"What is the capital of {country}?", capital, others))
        elif kind == "element":
            name, sym = els[ei % len(els)]
            ei += 1
            others = [s for _, s in ELEMENTS if s != sym]
            items.append(_mc(rng, f"What is the chemical symbol for {name}?", sym, others))
        elif kind == "fact":
            q, a, d = facts[fi % len(facts)]
            fi += 1
            items.append(_mc(rng, q, a, d))
        elif kind == "arith":
            a, b = rng.randint(12, 49), rng.randint(3, 19)
            v = a * b
            ds = sorted({v + rng.choice([-1, 1]) * rng.randint(1, 3) * b, v + 10, v - 10, v + a, v - a} - {v})
            items.append(_mc(rng, f"Which of the following equals {a} x {b}?", str(v), [str(x) for x in ds]))
        else:
            small, big, f = rng.choice(UNITS)
            n = rng.randint(2, 9)
            v = n * f
            ds = sorted({v + f, v - f, v * 10, v // 2 if v // 2 != v else v + 1, n * (f + 10)} - {v})
            items.append(_mc(rng, f"How many {small} are in {n} {big}?", str(v), [str(x) for x in ds]))
    return items


# ---------------------------------------------------------------------- tool calling
TOOLS = {
    "get_weather": {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather in a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "Name of the city"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}},
            "required": ["city", "unit"]}}},
    "convert_currency": {"type": "function", "function": {
        "name": "convert_currency", "description": "Convert an amount of money between two currencies.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number", "description": "Amount to convert"},
            "from_currency": {"type": "string", "description": "Three-letter code of the source currency"},
            "to_currency": {"type": "string", "description": "Three-letter code of the target currency"}},
            "required": ["amount", "from_currency", "to_currency"]}}},
    "set_timer": {"type": "function", "function": {
        "name": "set_timer", "description": "Start a countdown timer.",
        "parameters": {"type": "object", "properties": {
            "minutes": {"type": "integer", "description": "Length of the timer in minutes"},
            "label": {"type": "string", "description": "Name shown for the timer"}},
            "required": ["minutes", "label"]}}},
    "search_restaurants": {"type": "function", "function": {
        "name": "search_restaurants", "description": "Search for restaurants in a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City to search in"},
            "cuisine": {"type": "string", "description": "Type of food"},
            "max_price": {"type": "integer", "description": "Maximum price per person in dollars"}},
            "required": ["city", "cuisine"]}}},
}
CITIES = ["Paris", "Tokyo", "Berlin", "Madrid", "Toronto", "Sydney", "Cairo", "Lima", "Oslo", "Seoul", "Dublin",
          "Nairobi", "Lisbon", "Vienna", "Chicago", "Boston", "Denver", "Mumbai", "Bangkok", "Prague"]
CURRENCIES = ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "CNY", "INR", "SEK"]
CUISINES = ["italian", "thai", "mexican", "indian", "japanese", "greek", "korean", "vegan", "french", "turkish"]
LABELS = ["tea", "pasta", "laundry", "workout", "nap", "bread", "study", "meeting", "eggs", "pizza"]


def gen_tool_calling(rng: random.Random, count: int) -> list[dict]:
    items = []
    names = list(TOOLS)
    for i in range(count):
        target = names[i % len(names)]
        if target == "get_weather":
            city, unit = rng.choice(CITIES), rng.choice(["celsius", "fahrenheit"])
            text = rng.choice([f"What's the weather like in {city} right now? Please use {unit}.",
                               f"Tell me the current weather in {city} in {unit}."])
            args = {"city": city, "unit": unit}
        elif target == "convert_currency":
            amount = rng.choice([5, 12, 20, 45, 99, 150, 250, 300, 1000, 2500])
            a, b = rng.sample(CURRENCIES, 2)
            text = rng.choice([f"Convert {amount} {a} to {b}.", f"How much is {amount} {a} in {b}?"])
            args = {"amount": amount, "from_currency": a, "to_currency": b}
        elif target == "set_timer":
            m, label = rng.randint(1, 90), rng.choice(LABELS)
            text = rng.choice([f"Set a timer for {m} minutes called {label}.",
                               f"Please start a {m} minute timer labelled {label}."])
            args = {"minutes": m, "label": label}
        else:
            city, cuisine = rng.choice(CITIES), rng.choice(CUISINES)
            if rng.random() < 0.5:
                p = rng.choice([15, 20, 30, 40, 50])
                text = f"Find {cuisine} restaurants in {city} that cost at most {p} dollars per person."
                args = {"city": city, "cuisine": cuisine, "max_price": p}
            else:
                text = f"Find me some {cuisine} restaurants in {city}."
                args = {"city": city, "cuisine": cuisine}
        others = rng.sample([n for n in names if n != target], 2)
        tool_names = [target] + others
        rng.shuffle(tool_names)
        items.append({
            "messages": _user(text), "tools": [TOOLS[n] for n in tool_names], "max_new_tokens": 96,
            "grader": {"type": "tool_call", "name": target, "arguments": args,
                       "parameters": TOOLS[target]["function"]["parameters"]},
        })
    return items


# ---------------------------------------------------------------- instruction following
TOPICS = ["the ocean", "coffee", "bicycles", "libraries", "volcanoes", "gardening", "the moon", "trains", "rain",
          "music", "bread", "mountains", "bees", "chess", "winter", "rivers", "kites", "lighthouses", "owls",
          "football"]
KEYWORDS = ["bright", "quiet", "ancient", "simple", "green", "golden", "gentle", "busy", "small", "warm"]


def gen_instruction_following(rng: random.Random, count: int) -> list[dict]:
    items = []
    kinds = ["lowercase", "max_words", "bullets", "keyword", "no_commas", "quotes", "end_with", "json", "upper",
             "start_with"]
    for i in range(count):
        kind = kinds[i % len(kinds)]
        topic = rng.choice(TOPICS)
        if kind == "lowercase":
            text, checks = (f"Write one sentence about {topic} using only lowercase letters.",
                            [{"type": "lowercase"}])
        elif kind == "max_words":
            n = rng.choice([10, 12, 15])
            text, checks = (f"Describe {topic} in at most {n} words.", [{"type": "max_words", "n": n}])
        elif kind == "bullets":
            n = rng.choice([2, 3, 4])
            text, checks = (f"List exactly {n} facts about {topic} as bullet points that start with '- '. "
                            f"Do not write anything else.", [{"type": "bullet_count", "n": n}])
        elif kind == "keyword":
            kw = rng.choice(KEYWORDS)
            text, checks = (f"Write a short sentence about {topic} that includes the word '{kw}'.",
                            [{"type": "include_keywords", "keywords": [kw]}])
        elif kind == "no_commas":
            text, checks = (f"Write two sentences about {topic} without using any commas.", [{"type": "no_commas"}])
        elif kind == "quotes":
            text, checks = (f"Give a title for an article about {topic}, wrapped in double quotes, and nothing "
                            f"else.", [{"type": "wrap_quotes"}])
        elif kind == "end_with":
            text, checks = (f"Write two sentences about {topic} and end your response with the exact phrase "
                            f"'That is all.'", [{"type": "end_with", "suffix": "That is all."}])
        elif kind == "json":
            text, checks = (f"Respond with a JSON object with the keys \"topic\" and \"summary\" describing "
                            f"{topic}. Output only the JSON.", [{"type": "json_keys", "keys": ["topic", "summary"]}])
        elif kind == "upper":
            text, checks = (f"Write one sentence about {topic} in all capital letters.", [{"type": "uppercase"}])
        else:
            text, checks = (f"Start your answer with 'Indeed,' and write one sentence about {topic}.",
                            [{"type": "start_with", "prefix": "Indeed,"}])
        items.append({"messages": _user(text), "max_new_tokens": 160,
                      "grader": {"type": "instruction", "checks": checks}, "variant": kind})
    return items


# ------------------------------------------------------------------------------ code
CODE_TASKS = [
    ("reverse_string(s)", "returns the string s reversed",
     "assert reverse_string('abc') == 'cba'\nassert reverse_string('') == ''\nassert reverse_string('ab c') == 'c ba'"),
    ("count_vowels(s)", "returns the number of vowels (a, e, i, o, u, upper or lower case) in s",
     "assert count_vowels('hello') == 2\nassert count_vowels('AEIOU') == 5\nassert count_vowels('xyz') == 0"),
    ("factorial(n)", "returns n factorial for a non-negative integer n",
     "assert factorial(0) == 1\nassert factorial(5) == 120\nassert factorial(10) == 3628800"),
    ("fibonacci(n)", "returns the n-th Fibonacci number, with fibonacci(0) == 0 and fibonacci(1) == 1",
     "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55"),
    ("is_prime(n)", "returns True if the integer n is a prime number and False otherwise",
     "assert is_prime(2)\nassert is_prime(13)\nassert not is_prime(1)\nassert not is_prime(15)"),
    ("sum_digits(n)", "returns the sum of the decimal digits of a non-negative integer n",
     "assert sum_digits(0) == 0\nassert sum_digits(1234) == 10\nassert sum_digits(999) == 27"),
    ("is_palindrome(s)", "returns True if the string s reads the same forwards and backwards",
     "assert is_palindrome('racecar')\nassert is_palindrome('')\nassert not is_palindrome('abc')"),
    ("gcd(a, b)", "returns the greatest common divisor of two positive integers a and b",
     "assert gcd(12, 18) == 6\nassert gcd(7, 5) == 1\nassert gcd(100, 25) == 25"),
    ("celsius_to_fahrenheit(c)", "converts a temperature c in degrees Celsius to degrees Fahrenheit",
     "assert celsius_to_fahrenheit(0) == 32\nassert celsius_to_fahrenheit(100) == 212\n"
     "assert abs(celsius_to_fahrenheit(37) - 98.6) < 1e-9"),
    ("flatten(lists)", "takes a list of lists and returns one list with all the elements in order",
     "assert flatten([[1, 2], [3], []]) == [1, 2, 3]\nassert flatten([]) == []"),
    ("unique(items)", "returns the items of a list without duplicates, keeping the first occurrence order",
     "assert unique([3, 1, 3, 2, 1]) == [3, 1, 2]\nassert unique([]) == []"),
    ("word_count(s)", "returns the number of whitespace-separated words in the string s",
     "assert word_count('a b  c') == 3\nassert word_count('') == 0\nassert word_count('one') == 1"),
    ("second_largest(nums)", "returns the second largest distinct value in a list of at least two distinct numbers",
     "assert second_largest([1, 5, 3]) == 3\nassert second_largest([7, 7, 2]) == 2"),
    ("capitalize_words(s)", "returns s with the first letter of every space-separated word in upper case",
     "assert capitalize_words('hello world') == 'Hello World'\nassert capitalize_words('a') == 'A'"),
    ("count_char(s, c)", "returns how many times the character c occurs in the string s",
     "assert count_char('banana', 'a') == 3\nassert count_char('abc', 'z') == 0"),
    ("sum_even(nums)", "returns the sum of the even numbers in a list of integers",
     "assert sum_even([1, 2, 3, 4]) == 6\nassert sum_even([]) == 0\nassert sum_even([1, 3]) == 0"),
    ("merge_sorted(a, b)", "merges two sorted lists into one sorted list",
     "assert merge_sorted([1, 4], [2, 3]) == [1, 2, 3, 4]\nassert merge_sorted([], [1]) == [1]"),
    ("is_anagram(a, b)", "returns True if the strings a and b contain the same letters in a different order or "
     "the same order", "assert is_anagram('listen', 'silent')\nassert not is_anagram('abc', 'abd')"),
    ("clamp(x, lo, hi)", "returns x limited to the range from lo to hi inclusive",
     "assert clamp(5, 0, 10) == 5\nassert clamp(-3, 0, 10) == 0\nassert clamp(12, 0, 10) == 10"),
    ("running_sum(nums)", "returns a list whose i-th element is the sum of the first i+1 elements of nums",
     "assert running_sum([1, 2, 3]) == [1, 3, 6]\nassert running_sum([]) == []"),
    ("reverse_digits(n)", "returns the integer formed by reversing the decimal digits of a non-negative integer n",
     "assert reverse_digits(123) == 321\nassert reverse_digits(0) == 0\nassert reverse_digits(120) == 21"),
    ("is_power_of_two(n)", "returns True if the positive integer n is a power of two",
     "assert is_power_of_two(1)\nassert is_power_of_two(16)\nassert not is_power_of_two(12)"),
    ("chunk(items, k)", "splits a list into consecutive chunks of length k (the last chunk may be shorter)",
     "assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\nassert chunk([], 3) == []"),
    ("min_max(nums)", "returns a tuple (smallest, largest) for a non-empty list of numbers",
     "assert min_max([3, 1, 2]) == (1, 3)\nassert min_max([5]) == (5, 5)"),
    ("square_all(nums)", "returns a list with the square of every number in nums",
     "assert square_all([1, 2, 3]) == [1, 4, 9]\nassert square_all([]) == []"),
    ("remove_negatives(nums)", "returns the list without its negative numbers, keeping the order",
     "assert remove_negatives([1, -2, 3, -4, 0]) == [1, 3, 0]\nassert remove_negatives([]) == []"),
    ("triangular(n)", "returns the n-th triangular number 1 + 2 + ... + n for n >= 0",
     "assert triangular(0) == 0\nassert triangular(4) == 10\nassert triangular(10) == 55"),
    ("count_long_words(s, k)", "returns how many space-separated words in s are longer than k characters",
     "assert count_long_words('a bb ccc dddd', 2) == 2\nassert count_long_words('', 1) == 0"),
    ("average(nums)", "returns the arithmetic mean of a non-empty list of numbers",
     "assert average([1, 2, 3]) == 2\nassert average([5]) == 5"),
    ("vowel_positions(s)", "returns the list of indices of the vowels a, e, i, o, u in the lower-case string s",
     "assert vowel_positions('banana') == [1, 3, 5]\nassert vowel_positions('xyz') == []"),
]


def gen_code_unit(rng: random.Random, count: int) -> list[dict]:
    tasks = list(CODE_TASKS)
    rng.shuffle(tasks)
    items = []
    for sig, desc, tests in tasks[:count]:
        text = (f"Write a Python function `{sig}` that {desc}. Reply with only the code in a single "
                f"```python code block.")
        items.append({"messages": _user(text), "max_new_tokens": 192,
                      "grader": {"type": "unit_tests", "tests": tests, "timeout_s": 10}})
    return items


# ---------------------------------------------------------------------------- needle
_SUBJ = ["The old miller", "A young farmer", "The baker", "The ferryman", "A quiet shepherd", "The weaver",
         "An old sailor", "The schoolteacher", "A travelling merchant", "The blacksmith"]
_VERB = ["walked past", "looked at", "spoke about", "painted", "remembered", "repaired", "visited", "described",
         "watched", "admired"]
_OBJ = ["the stone bridge", "the wheat fields", "the river bank", "the narrow lane", "the village square",
        "the willow trees", "the harbour wall", "the hill path", "the red barn", "the market stalls"]
_TAIL = ["before the rain began.", "while the bells rang.", "in the pale morning light.", "after the long winter.",
         "as the wind turned cold.", "under a grey sky.", "on a slow afternoon.", "during the harvest.",
         "near the end of the day.", "without saying much."]
COLORS = ["amber", "blue", "crimson", "green", "ivory", "orange", "silver", "violet"]


def gen_needle_16k(rng: random.Random, count: int, words_target: int = 13900) -> list[dict]:
    items = []
    depths = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95]
    for i in range(count):
        sents, words = [], 0
        while words < words_target:
            s = f"{rng.choice(_SUBJ)} {rng.choice(_VERB)} {rng.choice(_OBJ)} {rng.choice(_TAIL)}"
            sents.append(s)
            words += len(s.split())
        color = COLORS[i % len(COLORS)]
        code = rng.randint(1_000_000, 9_999_999)
        pos = int(len(sents) * depths[i % len(depths)])
        sents.insert(pos, f"The secret code for the {color} vault is {code}.")
        text = (" ".join(sents) + f"\n\nWhat is the secret code for the {color} vault? Reply with only the number.")
        items.append({"messages": _user(text), "max_new_tokens": 16,
                      "grader": {"type": "numeric", "answer": code, "tol": 0}, "variant": f"depth_{depths[i % 8]}"})
    return items


GENERATORS = {"math_short": gen_math_short, "multiple_choice": gen_multiple_choice,
              "tool_calling": gen_tool_calling, "instruction_following": gen_instruction_following,
              "code_unit": gen_code_unit, "needle_16k": gen_needle_16k}


def generate_suite(name: str) -> list[dict]:
    rng = random.Random(SEEDS[name])
    raw = GENERATORS[name](rng, COUNTS[name])
    out = []
    for i, it in enumerate(raw):
        item = {"id": f"{name}-{i:04d}", "suite": name, "messages": it["messages"],
                "tools": it.get("tools"), "template_kwargs": dict(NO_THINK),
                "max_new_tokens": it["max_new_tokens"], "grader": it["grader"]}
        if "variant" in it:
            item["variant"] = it["variant"]
        out.append(item)
    return out
