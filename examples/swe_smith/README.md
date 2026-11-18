# Запуск ARBAS

Настройте `examples/swe_smith/training.yaml` на GPU-сервере. Из корня репозитория
запустите обучение и после появления подсказки запустите controller на Mac.

```bash
# На выделенной H200
bash examples/swe_smith/run.sh train

# На Mac, если создан профиль ~/.config/arbas/h200.env
bash examples/swe_smith/run.sh controller h200
```

На iPavlov `gpu12` сначала посмотрите доступные слоты через `gpu-run status` и
запускайте обучение внутри выделения `gpu-run`. На Mac используйте `controller ipavlov`.

```bash
gpu-run --gpu "$GPU" --slots "$SLOTS" -- bash examples/swe_smith/run.sh train ipavlov-2b
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

Подготовьте GPU-окружение скриптом `examples/swe_smith/setup_training.sh`.
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
OPENROUTER_API_KEY=ключ-openrouter
```

Порты (`18080` локально -> `8080` на сервере), адрес для подов
(`http://host.docker.internal:18080`), привязка туннеля (`0.0.0.0`), namespace
(`default`) и путь к `active.json` (`$REMOTE_REPO_DIR/outputs/swe-smith/active.json`)
вычисляются автоматически и при необходимости переопределяются через
`AGL_SERVER_PORT`, `AGL_SSH_REMOTE_PORT`, `AGL_SSH_BIND`, `AGL_SERVER_AGENT_URL`,
`AGL_NAMESPACE`, `AGL_REMOTE_OUTPUT_DIR`. Ограничьте права: `chmod 600 ~/.config/arbas/*.env`.

При запуске `controller` применяет один ConfigMap `swe-smith-agent-scripts` (из
`examples/swe_smith/hierarchical/`), Secret `swe-smith-openrouter` и подтягивает
образы из серверного `active.json` через `pull_images.py`.

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
