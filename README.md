<div align="center">

# ⚡ inference-from-scratch

**Пошаговая реконструкция LLM-рантайма — от токенизатора до Blackwell**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](.)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia&logoColor=white)](.)
[![GPU](https://img.shields.io/badge/GPU-Blackwell_SM_12.0-76B900?logo=nvidia&logoColor=white)](.)
[![Status](https://img.shields.io/badge/Status-In_Progress-0f766e?style=flat-square)](.)

<p>
  <strong>Jupyter-уроки с полной реализацией</strong> ·
  <strong>Целевая платформа: NVIDIA Blackwell</strong>
</p>

---

</div>

## 📌 О проекте

**inference-from-scratch** — это серия уроков для глубокого понимания LLM-инференса изнутри.

Каждый урок — Jupyter-ноутбук с **полной готовой реализацией** и объяснением концептов. После прохождения всех уроков мы построим движок **с нуля**.

> Когда понимаешь, как работают KV-кэш, планировщик и CUDA Graph — строить production LLM-рантайм становится в разы проще.

---

## 🎯 Что вы построите

Минимально жизнеспособный LLM-рантайм, способный:

- Токенизировать промпты через HF-адаптер
- Выполнять prefill и decode loop с KV-кэшем
- Запускать семплинг (temperature, top-k, top-p)
- Стримить токены
- Обслуживать Qwen3 8B в BF16 и FP8 на NVIDIA Blackwell

```
Prompt → Tokenize → Prefill → KV Cache
                                  ↓
                    Decode Loop (Scheduler → Model → Sample)
                                  ↓
                         Detokenize / Stream
```

---

## 📚 План уроков

### Фаза A — Основа

| # | Тема | Статус |
|---|------|--------|
| 1 | Токенизация (HF Qwen3 adapter) | ✅ |
| 2 | Семплинг (temperature, top-k, top-p) | — |
| 3 | Mock-модель и prefill логиты | — |
| 4 | Decode loop и простой планировщик | — |

### Фаза B — Память и стриминг

| # | Тема | Статус |
|---|------|--------|
| 5 | KV-кэш: интерфейсы и наивная реализация | — |
| 6 | Пагинация и минимальный Radix-кэш | — |
| 7 | Стриминг ответов | — |

### Фаза C — Реальная модель и бэкенды

| # | Тема | Статус |
|---|------|--------|
| 8 | Интеграция Qwen3 8B (Hugging Face) | — |
| 9 | Ограничения памяти/батча, OOM-защита | — |
| 10 | Бенчи и CUDA Graph (BF16) | — |
| 11 | Варианты внимания (FlashAttention / FlashInfer) | — |
| 12 | TRT-LLM + Blackwell FP8 | — |

<details>
<summary>Расширенный план — Фазы D–I (шаги 13–29)</summary>

**Фаза D — Протокол и сервер**
- 13: Протокол сообщений (UserMsg, BatchBackendMsg)
- 14: ZMQ шины и очереди
- 15: Tokenizer/Detokenizer воркеры
- 16: Scheduler I/O и backpressure
- 17: FastAPI сервер (OpenAI-совместимый `/v1/chat/completions`)
- 18: Оркестрация процессов
- 19: CLI / Args / ENV / Логи

**Фаза E — Распределёнка**
- 20: Tensor Parallel (NCCL/pynccl)
- 21: Шардирование весов
- 22: MoE-путь (Qwen3-MoE)

**Фаза F — Ядра**
- 23: Triton/CUDA ядра (radix/tensor)
- 24: FlashInfer/FA продвинутая интеграция

**Фаза G — Расширенный KV-кэш**
- 25: Политики prefix-кэша (lock/unlock, eviction, lazy_free)

**Фаза H — Наблюдаемость**
- 26: Метрики / Логирование / NVTX
- 27: Ошибки / таймауты / деградация

**Фаза I — Бенчи и деплой**
- 28: Онлайн/офлайн бенчи, sweep параметров
- 29: Пакетирование / Dockerfile / документация

</details>

---

## 🚀 Быстрый старт

### Требования

- Python `3.10+`
- CUDA Toolkit `12.x`
- GPU: NVIDIA с SM ≥ 8.0 (оптимально: RTX PRO 6000 Blackwell, SM 12.0)
- `uv` для управления окружением

### Установка

```bash
git clone <repo-url>
cd inference-from-scratch

uv venv --python=3.12
source .venv/bin/activate
```

### Запуск Урока 1

```bash
jupyter notebook tutorials/Lesson-01/lesson-01-tokenizer-adapter.ipynb
```

### Тесты

```bash
pytest tests/
pytest --nbmake tutorials/
```

---

## 📊 Стек

| Компонент | Технология |
|-----------|-----------|
| Модель | Qwen3 8B (Hugging Face) |
| Рантайм | PyTorch 2.x + CUDA 12.x |
| Внимание | FlashAttention / FlashInfer |
| FP8 / Blackwell | TensorRT-LLM |
| Ядра | Triton / CUDA |
| Тесты | pytest + nbmake |

---

## 🗺️ Словарь

| Термин | Значение |
|--------|---------|
| **KV-кэш** | Key-Value cache — хранилище промежуточных attention-состояний для ускорения decode |
| **Prefill** | Первый проход по промпту; результат — KV-записи и первый токен |
| **Decode loop** | Авторегрессивная генерация токенов с использованием KV-кэша |
| **Chunked Prefill** | Ограничение prefill по бюджету токенов для контроля пиковой памяти |
| **Radix cache** | Переиспользование общих KV-префиксов между запросами |
| **Blackwell / SM 12.0** | Архитектура GPU NVIDIA RTX PRO 6000; целевая платформа |

---

<div align="center">

Made with love in Russia ❤️

[Начать с Урока 1](./tutorials/Lesson-01/lesson-01-tokenizer-adapter.ipynb)

</div>
