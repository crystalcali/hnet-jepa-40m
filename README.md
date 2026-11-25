# H-Net-JEPA-40M

Байтовая H-Net с тремя уровнями сжатия, рекуррентным ядром и JEPA. Целевая
машина — NVIDIA DGX Spark. [ТЗ](SPEC.md) · [План](PLAN.md) · [Решения и статус](DECISIONS.md)

## 1. Установка и smoke-test на Spark

На хосте начни с `python3`. Проверь, что этот интерпретатор видит CUDA PyTorch.
Если импорт `torch` не проходит или CUDA недоступна, активируй уже настроенное
CUDA-окружение на Spark либо зайди в NVIDIA-контейнер с доступом к GPU и
повтори проверку там. В командах используй интерпретатор этого окружения.
Для SSH-туннеля панель должна быть доступна на loopback Spark. В контейнере
для этого подойдёт существующий запуск с host network.

```bash
git clone https://github.com/crystalcali/hnet-jepa-40m.git
cd hnet-jepa-40m
python3 -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), "CUDA unavailable"'
python3 -m pip install -r requirements-data.txt
python3 -m scripts.check_env
```

Если Spark по-прежнему не видит GitHub, клонируй на доступной машине, создай
bundle и перенеси его по уже настроенному SSH (используй свой SSH-адрес).

```bash
git clone https://github.com/crystalcali/hnet-jepa-40m.git
git -C hnet-jepa-40m bundle create hnet-jepa-40m.bundle --all
scp hnet-jepa-40m.bundle user@SPARK_HOST:~/
ssh user@SPARK_HOST
git clone ~/hnet-jepa-40m.bundle hnet-jepa-40m
cd hnet-jepa-40m
python3 -m pip install -r requirements-data.txt
```

При отсутствии рабочих `causal-conv1d` и `mamba-ssm` в окружении потребуется
совместимая сборка. В уже работающей CUDA-среде с `nvcc` и компилятором:

```bash
python3 -m pip install ninja packaging wheel setuptools
python3 -m pip install --no-build-isolation causal-conv1d
python3 -m pip install --no-build-isolation mamba-ssm
python3 -m scripts.check_env --cuda-smoke
python3 -m scripts.count_params --config configs/hnet_jepa_40m.yaml
python3 -m pytest -q
```

Перед пилотом smoke-test должен успешно проверить реальный CUDA forward/backward,
BF16 и сопоставимость Mamba-2 с CPU. Зависимости CUDA собираются под установленный
PyTorch. Успешно проверенные версии сохрани как локальный lock-файл через
`python3 -m pip freeze > requirements.spark.lock`. Если Spark работает
без доступа к индексам пакетов, установи подготовленные для этой среды wheel
через `pip --no-index --find-links`.

## 2. Корпус и короткая проверка

Можно сразу проверить весь цикл и панель на отдельной синтетической фикстуре,
в том числе на CPU. Она маркируется как диагностика и блокируется для
исследовательского обучения.

```bash
python3 -m scripts.debug_corpus --out cache/debug-fixture
python3 train.py --config configs/debug.yaml --data cache/debug-fixture --run runs/debug --device cpu --debug-corpus --max-steps 5
```


Заполни `configs/data_sources.yaml` готовыми RU–EN парами с проверенными
лицензиями, схемой выравнивания и зафиксированными ревизиями. Для benchmark
holdout нужен отдельный двуязычный список. Результат аудита и дефициты по
категориям записываются в `audit.json` и `deficits.json`. Поля
`quality_reviewed`, `near_dedup_reviewed` и `benchmark_reviewed` подтверждают
реально выполненные проверки по источникам. Шаблон поставляется пустым.

```bash
python3 -m data.audit --config configs/data_sources.yaml --target-bpe 30000000 --out cache/pilot
python3 -m data.pack --manifest cache/pilot/manifest.jsonl --out cache/pilot
python3 train.py --config configs/pilot.yaml --data cache/pilot --run runs/c-smoke --dry-run
python3 benchmark.py --config configs/pilot.yaml --data cache/pilot --steps 10 --full-objectives
```

`--dry-run` выполняет один forward/backward без шага оптимизатора. Benchmark
печатает измеренные исходные байт/с, пиковую память, K и покрытие каждого
objective. Проверь, что есть пригодные примеры для Dynamics, Task CE и Local JEPA.
Аудит малого корпуса можно использовать для диагностики с `--debug-corpus`,
но такие прогоны не входят в сравнение моделей.

## 3. Обучение и онлайн-метрики

На Spark запусти панель в отдельной сессии:

```bash
tmux new -s hnet-metrics
python3 -m dashboard.server --runs runs --port 7860
```

На своём компьютере открой туннель:

```bash
ssh -N -L 7860:127.0.0.1:7860 user@SPARK_HOST
```

Открой [http://localhost:7860](http://localhost:7860). Панель опрашивает журнал
раз в две секунды. Показывает loss и его компоненты, BPB на валидации,
фактическое сжатие, K, норму градиента, байт/с и память. Не использует GPU.
Сырые события доступны в `runs/<имя>/metrics.jsonl`. Каждые 200 шагов
пишутся нормы и попарные косинусы градиентов, а также снимок латентов и
границ на фиксированной пробе. Интервал меняется через `--probe-every`.

В отдельной сессии Spark:

```bash
tmux new -s hnet-pilot
python3 train.py --config configs/pilot.yaml --data cache/pilot --run runs/pilot_c --max-hours 6
```

`Ctrl-b`, затем `d` отсоединяет терминал, `tmux attach -t hnet-pilot` возвращает.
`Ctrl-c` просит сохранить чекпоинт после завершения текущего шага. Для
возобновления используй ту же конфигурацию, корпус, seed и горизонт экспозиции:

```bash
python3 train.py --config configs/pilot.yaml --data cache/pilot --run runs/pilot_c --resume latest --max-hours 6
```

`--max-hours` задаёт суммарный лимит времени запуска с учётом resume.
Для A и B используй `configs/transformer_40m.yaml` и `configs/hnet_40m.yaml`
с отдельными директориями запусков. Для сравнения сохрани одинаковые
корпус, seed и `--schedule-bpe` (по умолчанию 30M для пилота).
Для полного корпуса при аудите используй `--target-bpe 1000000000
--tokenizer cache/pilot/tokenizer.json`. Полный запуск требует `--full`,
корпус с пройденным аудитом и отдельную команду, максимум 72 ч на модель.

## 4. Генерация и оценка

```bash
python3 generate.py --checkpoint runs/pilot_c/latest --mode task --iterations 8 --language ru --prompt 'Сколько будет (7+3)*5?'
python3 evaluate.py --checkpoint runs/pilot_c/latest --data cache/pilot --split test --out runs/pilot_c/eval.json
python3 evaluate.py --checkpoint runs/pilot_c/latest --data cache/pilot --split test --reasoning-per-level 200 --iterations 1,2,4,8,16,32
```

В параллельном train лучше не запускать генерацию на том же GPU.
Для проверки на CPU можно добавить `--device cpu`, короткий контекст и
небольшое число примеров. K=16 и 32 — экстраполяция глубины.

### Стандартные бенчмарки

`evaluate.py --benchmark /path/to/heldout.jsonl` принимает нормализованную
валидированную выгрузку. Путь и SHA выгрузки фиксируй отдельно от train. На
одну задачу нужна строка с `dataset`, `id`, `language`, `prompt` и `kind`.

| kind | Поля и оценка |
|---|---|
| choice | `choices`, `answer_index`. Суммарная условная log-likelihood или `score: "per_byte"`. Для BLiMP, RuBLiMP и HellaSwag. |
| exact | `answer`, при необходимости `mode: "lm"`, `compare: "integer"` или `compare: "normalized_text"`. Для LAMBADA, GSM8K, MGSM и логики при соответствующей подготовке эталона. |
| translation | `answer`, `source_language`, `language` как язык ответа. BLEU и chrF++ для готовых пар FLORES. |

`max_bytes` ограничивает генерацию, по умолчанию 128. Для benchmark с
выбором варианта ранжируются одинаковые строки. Для режима генерации явно
укажи `--benchmark-iterations`. Код оценки не исполняет содержимое датасета.
