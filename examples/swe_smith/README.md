# Запуск ARBAS

Поддерживаются два режима запуска:
1. **Автономный на одной машине / в Slurm** (`local` runner): обучение, шлюз, контроллер и SWE-smith роллауты запускаются прямо на сервере в локальном Docker. Подключение ноутбука или Kubernetes не требуется.
2. **Распределённый** (`k8s` runner): GPU-обучение и шлюз работают на сервере, а контроллер запускается на Mac и шедулит задачи в локальный Kubernetes через SSH-туннель.

---

## Вариант 1: Автономный запуск (на сервере или в Slurm)

Все процессы (шлюз `agl-server`, тренер VERL и контроллер `LocalReconciler`) работают на одной машине. Роллауты SWE-smith выполняются локально в изолированных контейнерах `docker run`.

### В интерактивной сессии Slurm (rootless Docker):
```bash
# 1. Запрос GPU и интерактивной сессии с rootless Docker
srun -p gpu --time=04:00:00 --gres=gpu:h200:1 -c16 --mem=64G \
  --pty slurm-docker-session --workspace arbas -- bash

# 2. В терминале 1: запуск обучения (поднимает шлюз и тренер VERL)
bash examples/swe_smith/run.sh train smoke-h200

# 3. В терминале 2 (или tmux): запуск локального контроллера
bash examples/swe_smith/run.sh controller local
```

Контроллер автоматически:
- Подключается к локальному шлюзу `127.0.0.1:8080`.
- Проверяет образы задач из `active.json` и при необходимости подтягивает их локально.
- Запускает задачи через `DockerRunner` с монтированием `/agl/hierarchical:ro` и логов `/agl-logs`.
- Пишет единый лог в `outputs/swe-smith/<run_id>/<run_id>-controller.log`.

---

## Вариант 2: Распределённый запуск (GPU-сервер + Mac с K8s)

### 1. На GPU-сервере:
```bash
# На выделенной H200 или внутри gpu-run на iPavlov gpu12
bash examples/swe_smith/run.sh train
```

### 2. На Mac:
```bash
# Если создан профиль ~/.config/arbas/h200.env или ipavlov.env
bash examples/swe_smith/run.sh controller h200
```

`train` поднимает шлюз `agl-server` и тренера VERL; при завершении или остановке
команды шлюз и все дочерние процессы зачищаются автоматически. `controller`
создаёт K8s Job-ы задач и работает в своём терминале. Для нового прогона
остановите controller и запустите его снова после старта шлюза. На обеих машинах
должна быть одна версия кода.

## Каталоги запусков

Имя после `train` — подпись эксперимента (по умолчанию `experiment.name` из YAML).
Каталог запуска создаётся в формате `outputs/swe-smith/<name>-<MMDD-HH>/` по
московскому времени (при повторном запуске в тот же час добавляется суффикс
`-2`, `-3`). Внутри сохраняются снимок `training.yaml`, логи
`<run_id>-server.log` и `<run_id>-trainer.log`, TensorBoard и checkpoints.
Симлинк `outputs/swe-smith/latest` указывает на последний запуск.

Временные файлы тренера и сокетов Ray изолируются во временной папке и
гарантированно зачищаются при завершении. Для смены расположения временных файлов
(например, если на `/tmp` мало места) укажите `AGL_TMPDIR` в `machine.env`.

## Настройка окружения (`.env`)

Подготовьте GPU-окружение командой `bash examples/swe_smith/setup_training.sh`.
Скрипт создаёт `.venv` на Python 3.12 от `uv` и до установки пакетов проверяет
наличие `Python.h`, который нужен Triton для компиляции. На Slurm `.venv` может
быть симлинком на ваш каталог окружения.

Прямые зависимости обучения перечислены в `requirements.in`. Установщик
берёт версии всего дерева зависимостей из `requirements.txt`. После изменения
прямых зависимостей пересоберите этот файл на Linux x86_64 командой
`bash examples/swe_smith/setup_training.sh --lock`.

На Mac нужны Docker Desktop, Kubernetes, `kubectl` и Python-окружение проекта
(с пакетом `docker` для автоматической сборки `:arbas`-образов). Скрипт `run.sh`
подхватывает активное виртуальное окружение или `.venv` в корне репозитория и
сохраняет `CUDA_VISIBLE_DEVICES`, выданный `gpu-run` или Slurm.

На GPU-сервере создайте `~/.config/arbas/machine.env`:

```env
AGL_KEY=общий-ключ-связи
```

На Mac создайте `~/.config/arbas/ipavlov.env` (или `h200.env` / `machine.env`) из
**4 обязательных полей**:

```env
SSH_TARGET=ipavlov-a100
REMOTE_REPO_DIR=/data/home/valeriy_gaideek/arbas
AGL_KEY=общий-ключ-связи
POLZA_API_KEY=ключ-polza
```

Указывайте ключ провайдера из `agents.subagent.base_url`. Для OpenRouter
используйте `OPENROUTER_API_KEY` вместо `POLZA_API_KEY`.

Порты (`18080` локально -> `8080` на сервере), адрес для подов
(`http://host.docker.internal:18080`), привязка туннеля (`0.0.0.0`), namespace
(`default`) и путь к `active.json` (`$REMOTE_REPO_DIR/outputs/swe-smith/active.json`)
вычисляются автоматически и при необходимости переопределяются через
`AGL_SERVER_PORT`, `AGL_SSH_REMOTE_PORT`, `AGL_SSH_BIND`, `AGL_SERVER_AGENT_URL`,
`AGL_NAMESPACE`, `AGL_REMOTE_OUTPUT_DIR`. Ограничьте права: `chmod 600 ~/.config/arbas/*.env`.

При запуске `controller` применяет один ConfigMap `swe-smith-agent-scripts` (из
`examples/swe_smith/hierarchical/`), сохраняет ключ в Secret `swe-smith-polza`
или `swe-smith-openrouter` и подтягивает образы из серверного `active.json` через
`pull_images.py`.

## Единый конфиг `training.yaml`

Все параметры эксперимента, VERL, Ray и агентов собраны в одном файле
`examples/swe_smith/training.yaml`. Для нового эксперимента скопируйте его рядом
и передайте через `--config`:

```bash
bash examples/swe_smith/run.sh train my-exp --config examples/swe_smith/my-exp.yaml
```

Секция `agents:` валидируется схемой `AgentSettings` (`hierarchical/config.py`)
прямо при старте до запуска Ray и передаётся в под одним JSON-параметром
`ARBAS_AGENT_CONFIG`:
- `orchestrator`, `subagent`, `judge` — модели, лимиты ходов/токенов, температура, `reasoning`;
- `sandbox` — тайм-ауты команд/оценки/API, ожидание шлюза, лимит вывода, `f2p_only`;
- `reward` — штрафы за число ходов и длину промпта на обучении;
- `oracle` — список моделей (`models: []` отключает роутинг), чекпоинт `linucb.pkl`, веса стоимости и тип обратной связи (`semantic` / `episode`).
