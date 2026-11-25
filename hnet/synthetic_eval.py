"""Evaluation-only, deterministic bilingual reasoning tasks."""
import random


def tasks(count, seed):
    rng = random.Random(seed)
    for level in range(1, 7):
        for index in range(count):
            a, b, c = [rng.randrange(3, 40) for _ in range(3)]
            if level == 1:
                en, ru, v = (f"Box A has {a} tokens. Box B has {b} tokens. How many tokens are in box A?",
                             f"В коробке A {a} жетонов. В коробке B {b} жетонов. Сколько жетонов в A?", a)
            elif level in (2, 3):
                path = rng.sample(range(12, 99), 3 if level == 2 else rng.randrange(5, 9))
                links = list(zip(path, path[1:])) + [(100, 101)]
                rng.shuffle(links)
                edges = ", ".join(f"{x}->{y}" for x, y in links)
                hops = 1 if level == 2 else len(path) - 1
                en = f"Directed edges: {edges}. Start at {path[0]} and follow {hops} edges. Where do you end?"
                ru = f"Дуги графа: {edges}. Начни с {path[0]} и пройди {hops} дуг. Где остановишься?"
                v = path[hops]
            elif level == 4:
                expression = f"(({a} + {b}) * {c}) - {a}"
                en, ru, v = f"Compute {expression}.", f"Вычисли {expression}.", (a + b) * c - a
            elif level == 5:
                en = f"Let A(x)=2*x+{a} and B(x)={b}-x. Find A(B(A({c})))."
                ru = f"Пусть A(x)=2*x+{a}, B(x)={b}-x. Найди A(B(A({c})))."
                v = 2 * (b - (2 * c + a)) + a
            else:
                n = rng.randrange(4, 10)
                en = f"Set x={a}. Repeat {n} times: x=(x*{b}+{c}) mod 101. What is x?"
                ru = f"Задай x={a}. Повтори {n} раз: x=(x*{b}+{c}) mod 101. Чему равно x?"
                v = a
                for _ in range(n):
                    v = (v * b + c) % 101
            prompt = {"en": en + " Answer with a single integer.",
                      "ru": ru + " Ответь одним целым числом."}
            yield {"id": f"{seed}:{level}:{index}", "level": level, "prompt": prompt, "answer": str(v)}
