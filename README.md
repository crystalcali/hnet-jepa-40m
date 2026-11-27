# H-Net-JEPA-40M

Байтовая H-Net с тремя уровнями сжатия, рекуррентным ядром и JEPA. Целевая
машина — NVIDIA DGX Spark. [ТЗ](SPEC.md) · [План](PLAN.md) · [Решения и статус](DECISIONS.md)

**Новый эксперимент:** [англоязычное сравнение BPE Transformer и одноуровневой H-Net](docs/pretrain_en.md).

## 1. Окружение и CUDA smoke-test на Spark

### Установка без Docker

На ARM64 DGX Spark с Python 3.12 и CUDA Toolkit 13.0 выполни:

```bash
git clone https://github.com/crystalcali/hnet-jepa-40m.git
cd hnet-jepa-40m
bash scripts/spark-native.sh smoke
```

Если репозиторий уже клонирован, используй `git pull --ff-only` вместо
клонирования. Установщик создаёт `.spark/venv-cu130`, загружает
`torch==2.9.1+cu130` из официального индекса PyTorch, собирает расширения из
исходников и проверяет CUDA. Он использует PyTorch и PyPI для загрузок.
Существующее окружение `.venv` остаётся как есть.

Если CUDA Toolkit установлен по нестандартному пути, передай его через
`CUDA_HOME=/path/to/cuda-13.0 bash scripts/spark-native.sh smoke`. Утилита
`nvcc` нужна для сборки. Версия CUDA в `nvidia-smi` показывает возможности
драйвера. При ошибке создания venv нужен пакет `python3.12-venv`.

Обычная сборка Mamba компилирует несколько GPU-архитектур. Если нужно
ускорить новую сборку на GB10, есть `bash scripts/spark-native.sh smoke-fast`.
Этот вариант использует ту же версию Mamba, компилирует только `sm_121`
и показывает прогресс компиляции. Если обычная сборка уже запущена, не
запускай вторую параллельно. Если Mamba уже установлена, быстрая команда
переходит к smoke-test. Wheel causal-conv1d переиспользуется из pip-кэша,
если он там есть.

После smoke-test активируй окружение для команд следующих разделов:

```bash
source .spark/venv-cu130/bin/activate
```

Панель из отдельного терминала запускается командой
`bash scripts/spark-native.sh dashboard --port 7860`. Для обучения можно
использовать активированное окружение или
`bash scripts/spark-native.sh run python3 train.py ...`.

### Установка через NVIDIA Docker

Для контейнерного варианта используй Docker с NVIDIA GPU runtime. Базовый образ
`nvcr.io/nvidia/pytorch:25.11-py3` содержит CUDA PyTorch и CUDA Toolkit.
При первой сборке установщик добавляет зависимости проекта в образ и собирает
закреплённые версии Mamba-2 и causal-conv1d под этот PyTorch. Системный Python
Spark не участвует. Нужен доступ к NGC и PyPI для первой сборки.

```bash
git clone https://github.com/crystalcali/hnet-jepa-40m.git
cd hnet-jepa-40m
bash scripts/spark.sh smoke
```

Из уже клонированного репозитория сначала выполни `git pull --ff-only`.
Команда `smoke` собирает образ при первом запуске и запускает CUDA-проверку.
Она проверяет прямой и обратный проход, BF16 и сопоставимость Mamba-2 с CPU.
Для пересборки образа есть `bash scripts/spark.sh build`.

При установке через Docker для команд следующих разделов открой оболочку контейнера.
Исходники, `cache/`, `runs/` и чекпоинты остаются в каталоге на хосте.

```bash
bash scripts/spark.sh shell
```

Если Spark не видит GitHub, клонируй на другой машине и перенеси Git bundle
по SSH. Команды ниже используют твой SSH-адрес вместо `user@SPARK_HOST`.

```bash
git clone https://github.com/crystalcali/hnet-jepa-40m.git
git -C hnet-jepa-40m bundle create hnet-jepa-40m.bundle --all
scp hnet-jepa-40m.bundle user@SPARK_HOST:~/
ssh user@SPARK_HOST
git clone ~/hnet-jepa-40m.bundle hnet-jepa-40m
cd hnet-jepa-40m
bash scripts/spark.sh smoke
```

При работе в собственном CUDA-окружении выполни `python3 -m pip install -r
requirements-data.txt` и `python3 -m scripts.check_env --cuda-smoke` внутри
этого окружения. Проверенные версии можно сохранить командой
`python3 -m pip freeze > requirements.spark.lock`.

## 2. Корпус и короткая проверка

Можно сразу проверить весь цикл и панель на отдельной синтетической фикстуре,
в том числе на CPU. Она маркируется как диагностика и блокируется для
исследовательского обучения.

```bash
python3 -m scripts.debug_corpus --out cache/debug-fixture
python3 train.py --config configs/debug.yaml --data cache/debug-fixture --run runs/debug --device cpu --debug-corpus --max-steps 5
```


### Реальный двуязычный корпус

Конфигурация [configs/data_sources.yaml](configs/data_sources.yaml) содержит
закреплённые ревизии опубликованных источников и их адаптеры. Если уже есть
подготовленный архив корпуса, распакуй его в `cache/`. Сборка с нуля идёт
на CPU и может скачать около 2–3 ГБ исходных файлов:

```bash
python3 -m data.build_pilot --out cache/pilot
python3 -m data.pack --manifest cache/pilot/manifest.jsonl --out cache/pilot
```

Проверь `audit.json` и `deficits.json`. В текущем корпусе точный поиск дублей
и проверка совпадений с GSM8K/MGSM выполнены. Расширенная проверка
контаминации и близких дублей открыта. Logic-701 используется как источник
размеченных данных и исключён из независимой оценки. Python-набор содержит
пары заданий. Парных решений в этом источнике нет.

Диагностика на реальном корпусе с недостающими квотами запускается явно:

```bash
python3 train.py --config configs/pilot.yaml --data cache/pilot --run runs/real_debug \
  --debug-corpus --max-steps 10
```

Для оценки памяти и скорости после проверки CUDA:

```bash
python3 benchmark.py --config configs/pilot.yaml --data cache/pilot --steps 10 --full-objectives
```

## 3. Обучение и онлайн-метрики

На хосте Spark запусти панель в отдельной сессии:

```bash
tmux new -s hnet-metrics
bash scripts/spark.sh dashboard --port 7860
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

В отдельной сессии Spark запусти обучение. Для установки без Docker:

```bash
tmux new -s hnet-pilot
bash scripts/spark-native.sh run python3 train.py --config configs/pilot.yaml --data cache/pilot --run runs/pilot_c --max-hours 6
```

Для установки через Docker:

```bash
tmux new -s hnet-pilot
bash scripts/spark.sh run python3 train.py --config configs/pilot.yaml --data cache/pilot --run runs/pilot_c --max-hours 6
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
