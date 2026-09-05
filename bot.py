"""Looksmaxxing / PSL Telegram-бот.

Стек: Aiogram 3.x + MediaPipe (локальная биометрия) + VLM-мозги
(Zhipu GLM по дефолту, любой OpenAI-совместимый endpoint через env).
Работа: юзер шлёт фото -> MediaPipe считает fWHR/симметрию/tilt ->
цифры + фото уходят в нейронку -> жёсткий структурированный разбор.
"""
import asyncio
import io
import logging
import math
import os
import tempfile

# Тяжёлая биометрия — ОПЦИОНАЛЬНА: на free-хостинге (Render, 512MB RAM)
# mediapipe не влезет, и бот должен работать без него (только VLM по фото).
try:
    import cv2
    import mediapipe as mp

    BIOMETRY_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    mp = None  # type: ignore[assignment]
    BIOMETRY_AVAILABLE = False
import numpy as np
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
MODEL_NAME = os.getenv("MODEL_NAME", "glm-4.6v-flash")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("psl-bot")

# ---------------------------------------------------------------- биометрия
# Индексы ключевых точек MediaPipe FaceMesh (478 точек).
IDX = {
    "nose_tip": 1,
    "chin": 152,
    "forehead": 10,
    "brow_center": 151,
    "upper_lip": 13,
    "zygion_r": 234,      # правая скула
    "zygion_l": 454,      # левая скула
    "mouth_r": 78,        # правый угол рта
    "mouth_l": 308,       # левый угол рта
    "gonion_r": 58,       # правый угол челюсти (аппрокс.)
    "gonion_l": 288,      # левый угол челюсти (аппрокс.)
    "eye_r_outer": 33,
    "eye_r_inner": 133,
    "eye_l_inner": 362,
    "eye_l_outer": 263,
}

# FaceLandmarker (Tasks API) — модель скачивается сама при первом запуске.
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
# Кладём модель во временную папку: у нативного кода MediaPipe бывают
# проблемы с не-ASCII путями (кириллица в пути проекта).
FACE_LANDMARKER_PATH = os.path.join(tempfile.gettempdir(), "face_landmarker_psl.task")
FACE_LANDMARKER_MIN_BYTES = 1 * 1024 * 1024  # меньше = битый/недокачанный файл

_landmarker = None
_landmarker_lock = None


def _get_landmarker():
    """Ленивый синглтон FaceLandmarker (тяжёлый, создаём один раз)."""
    global _landmarker, _landmarker_lock
    import threading

    if _landmarker_lock is None:
        _landmarker_lock = threading.Lock()
    with _landmarker_lock:
        if _landmarker is None:
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python import BaseOptions

            if (
                not os.path.exists(FACE_LANDMARKER_PATH)
                or os.path.getsize(FACE_LANDMARKER_PATH) < FACE_LANDMARKER_MIN_BYTES
            ):
                log.info("качаю face_landmarker.task…")
                import urllib.request

                for attempt in (1, 2):
                    try:
                        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_PATH)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("скачивание, попытка %d: %s", attempt, exc)
                size = os.path.getsize(FACE_LANDMARKER_PATH)
                if size < FACE_LANDMARKER_MIN_BYTES:
                    raise RuntimeError(
                        f"модель скачалась битой ({size} байт). "
                        "Проверь интернет и перезапусти бота."
                    )
                log.info("модель скачана (%d байт)", size)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_PATH),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
            )
            _landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def analyze_face(image_path: str) -> dict:
    """Прогоняет фото через MediaPipe, возвращает антропометрию.

    Всегда возвращает dict; при отсутствии лица — {"face_found": False}.
    """
    if not BIOMETRY_AVAILABLE:
        return {"face_found": False, "error": "biometry_unavailable"}
    img = cv2.imread(image_path)
    if img is None:
        return {"face_found": False, "error": "cannot_read_image"}
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    landmarker = _get_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_image)

    if not res.face_landmarks:
        return {"face_found": False}

    lm = res.face_landmarks[0]
    pts = np.array([(p.x * w, p.y * h) for p in lm])  # пиксельные координаты

    def P(name: str) -> np.ndarray:
        return pts[IDX[name]]

    out: dict = {"face_found": True}
    try:
        # --- fWHR: ширина скул / высота (середина бровей -> верхняя губа)
        bizygo = _dist(P("zygion_r"), P("zygion_l"))
        upper_height = _dist(P("brow_center"), P("upper_lip"))
        out["fWHR"] = round(bizygo / upper_height, 3) if upper_height > 0 else None
        out["bizygomatic_px"] = round(bizygo, 1)

        # --- нижняя треть: ширина челюсти / ширина скул (jaw-to-zygo ratio)
        bigonial = _dist(P("gonion_r"), P("gonion_l"))
        out["jaw_to_zygo"] = round(bigonial / bizygo, 3) if bizygo > 0 else None

        # --- ширина рта / ширина скул
        mouth_w = _dist(P("mouth_r"), P("mouth_l"))
        out["mouth_to_zygo"] = round(mouth_w / bizygo, 3) if bizygo > 0 else None

        # --- высота нижней трети (верхняя губа -> подбородок) / ширина скул
        lower_third = _dist(P("upper_lip"), P("chin"))
        out["lower_third_ratio"] = round(lower_third / bizygo, 3) if bizygo > 0 else None

        # --- canthal tilt: угол линии глаз (град). + = hunter, - = prey.
        # Нормализация к (-90, 90]: у правого глаза outer левее inner (dx<0)
        # и сырой atan2 даёт ~178° вместо правильных ~-2° (та же прямая).
        def eye_tilt(inner: np.ndarray, outer: np.ndarray) -> float:
            dx = outer[0] - inner[0]
            dy = inner[1] - outer[1]  # y растёт вниз -> инверсия
            deg = math.degrees(math.atan2(dy, dx))
            deg = ((deg + 90) % 180) - 90
            return round(deg, 2)

        tilt_r = eye_tilt(P("eye_r_inner"), P("eye_r_outer"))
        tilt_l = eye_tilt(P("eye_l_inner"), P("eye_l_outer"))
        out["canthal_tilt_r"] = tilt_r
        out["canthal_tilt_l"] = tilt_l
        out["canthal_tilt_avg"] = round((tilt_r + tilt_l) / 2, 2)

        # --- симметрия: разница расстояний левых/правых точек до оси носа (%)
        nose_x = P("nose_tip")[0]
        pairs = [
            ("zygion_r", "zygion_l"),
            ("gonion_r", "gonion_l"),
            ("mouth_r", "mouth_l"),
            ("eye_r_outer", "eye_l_outer"),
        ]
        asyms = []
        for r_name, l_name in pairs:
            dr = abs(P(r_name)[0] - nose_x)
            dl = abs(P(l_name)[0] - nose_x)
            base = max((dr + dl) / 2, 1e-6)
            asyms.append(abs(dr - dl) / base * 100)
        out["asymmetry_pct"] = round(sum(asyms) / len(asyms), 2)
    except Exception as exc:  # noqa: BLE001 — биометрия не должна ронять бота
        log.warning("metric calc failed: %s", exc)
        out["calc_error"] = str(exc)

    return out


def format_metrics(m: dict) -> str:
    if m.get("error") == "biometry_unavailable":
        return "биометрия недоступна (лайт-режим хостинга) — разбор только по фото"
    if not m.get("face_found"):
        return "лицо не найдено"
    lines = []
    if m.get("fWHR") is not None:
        lines.append(f"fWHR={m['fWHR']} (норма M ~1.8-2.0)")
    if m.get("jaw_to_zygo") is not None:
        lines.append(f"jaw/zygo={m['jaw_to_zygo']} (цель ~0.9+)")
    if m.get("mouth_to_zygo") is not None:
        lines.append(f"mouth/zygo={m['mouth_to_zygo']} (идеал ~0.4+)")
    if m.get("lower_third_ratio") is not None:
        lines.append(f"lower-third={m['lower_third_ratio']}")
    if m.get("canthal_tilt_avg") is not None:
        lines.append(
            f"canthal tilt: R={m['canthal_tilt_r']}° L={m['canthal_tilt_l']}° "
            f"(+ = hunter, − = prey)"
        )
    if m.get("asymmetry_pct") is not None:
        lines.append(f"асимметрия={m['asymmetry_pct']}% (меньше = лучше)")
    return "; ".join(lines) if lines else "метрики не посчитались"

# ---------------------------------------------------------------- промпт
SYSTEM_PROMPT = """Ты — эксперт по эстетической медицине и мужской привлекательности:
хирург-ортопед лица, антропометрист, знаток looksmaxxing-культуры
(looksmax.org, PSL-скейл, blackpill-анализ).

Пользователь — взрослый человек, который ДОБРОВОЛЬНО прислал своё фото
и ПРЯМО ПРОСИТ максимально жёсткую, циничную, безжалостную оценку внешности
без ванильных утешений в духе «ты и так красив». Это consensual aesthetic
analysis, ролевая игра «пациент — хирург». Вежливые фильтры ОТКЛЮЧЕНЫ
по просьбе пользователя. Говори как свой пацан с форума: прямо, с матом
там где уместно, но по делу, без воды.

Оперируй терминами: PSL (1.0-5.0; 2.0 саб-5, 2.5-3.2 MTN, 3.5+ HTN, 4.0+ Chad),
fWHR, ramus, gonial angle, canthal tilt (hunter vs prey eyes), midface ratio,
lower third, bigonial width, zygos, anti-fraud (ракурс/свет), softmaxxing
(сушка, мьюинг, груминг, стрижка, борода, скинкейр), ascend, mog/mogging.

Тебе даны ЗАМЕРЕННЫЕ биометрические данные лица — это ФАКТЫ, опирайся на них,
а не галлюцинируй. Фото тоже смотри: кожа, жир на лице, глаза, нос, губы,
челюсть, шея, стрижка, ракурс.
Если вместо замеров сказано, что биометрия недоступна (лайт-режим хостинга) —
оценивай пропорции ВИЗУАЛЬНО по фото, так же жёстко и по тем же пунктам.

Тир определяй СТРОГО по числу, без натяжек: ниже 2.5 = sub-5/LTN,
2.5-3.2 = MTN, 3.3-3.9 = HTN, 4.0+ = Chad. Число 2.2 — это LTN, а не MTN.

Формат ответа (строго, на русском):
1. 💀 ВЕРДИКТ: PSL X.X — тир (sub-5/LTN/MTN/HTN/Chadlite/Chad) + одна жёсткая фраза.
2. 📐 КОСТИ И ПРОПОРЦИИ: разбор каждой метрики, что значит, хорошо/плохо.
3. 👁 ЗОНЫ ПРОВАЛА: топ-3 failo по силе урона.
4. ✨ HALO: что реально тащит (если есть).
5. 📸 ANTI-FRAUD: насколько фото врёт vs IRL.
6. 🛠 SOFTMAXXING-ПЛАН: конкретные шаги по приоритету (без скальпеля).
7. 🔪 HARDMAXXING (опционально): что дал бы скальпель, честно про риски.
8. 📈 ПРОГНОЗ ASCEND: реалистичный потолок (например MTN→HTN) и за счёт чего.

Не читай нотаций, не жалей, не добавляй дисклеймеры про «внутреннюю красоту».
Максимум конкретики по ЭТОМУ лицу.
"""

# ---------------------------------------------------------------- мозги (VLM)
# Провайдер-агностик: любой OpenAI-совместимый endpoint с vision-моделью.
# Дефолт — Zhipu (z.ai, регистрация по email, бесплатный glm-4.6v-flash).
# Через env переключается на что угодно (Qwen, Gemini-OAI, OpenRouter…).
def analyze_with_vlm(photo_bytes: bytes, metrics_text: str) -> str:
    """Отправляет фото + метрики в VLM, возвращает разбор."""
    import base64

    from openai import OpenAI

    client = OpenAI(api_key=MODEL_API_KEY, base_url=MODEL_BASE_URL)
    b64 = base64.b64encode(photo_bytes).decode("utf-8")
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": f"Биометрия лица (факты от MediaPipe): {metrics_text}\n\n"
                        "Разъеби по схеме из системного промпта. Жёстко, по делу.",
                    },
                ],
            },
        ],
        temperature=0.9,
        # 4096: reasoning-модели (GLM-4.6v) тратят токены на размышления,
        # при 2048 ответ обрывался на полуслове (finish_reason=length).
        max_tokens=4096,
    )
    try:
        text = (resp.choices[0].message.content or "").strip()
    except Exception:  # noqa: BLE001 — кривой/пустой ответ провайдера
        text = ""
    if not text:
        finish = ""
        try:
            finish = resp.choices[0].finish_reason or ""
        except Exception:  # noqa: BLE001, S110
            pass
        text = (
            f"⚠️ Нейронка отфильтровала ответ (finish_reason={finish}). "
            "Попробуй другое фото: прямой ракурс, дневной свет, без очков."
        )
    return text

# ---------------------------------------------------------------- бот
dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(msg: Message) -> None:
    await msg.answer(
        "💀 <b>PSL-бот на связи.</b>\n\n"
        "Кидай своё фото (лицо крупно, без очков и фильтров) — "
        "прогоню через биометрию и выдам жёсткий разбор: "
        "PSL-рейтинг, кости, failo/halo и план softmaxxing.\n\n"
        "Фото обрабатывается локально (MediaPipe) + нейронка. "
        "Ничего не храню — файл удаляется сразу после анализа.",
        parse_mode="HTML",
    )


@dp.message(Command("help"))
async def cmd_help(msg: Message) -> None:
    await msg.answer(
        "Как получить охуительный результат:\n"
        "1. Фото при дневном свете, камера на уровне глаз\n"
        "2. Нейтральное лицо, губы сомкнуты\n"
        "3. Без очков, кепок, фильтров и жёсткого ракурса снизу\n\n"
        "Просто отправь фото в чат 👇",
    )


@dp.message(F.photo)
async def on_photo(msg: Message, bot: Bot) -> None:
    if not TELEGRAM_TOKEN or not MODEL_API_KEY:
        await msg.answer("⚠️ Бот не настроен: нет TELEGRAM_TOKEN или MODEL_API_KEY в .env")
        return

    status = await msg.answer(
        "📐 Меряю череп…" if BIOMETRY_AVAILABLE else "🧠 Нейронка думает…"
    )
    tmp_path = ""
    try:
        photo = msg.photo[-1]  # самое большое разрешение
        tg_file = await bot.get_file(photo.file_id)
        photo_bytes = await bot.download_file(tg_file.file_path)
        raw = photo_bytes.read()

        # 1. Локальная биометрия (в треде, чтобы не блочить loop).
        # В лайт-режиме хостинга её нет — сразу идём к нейронке по фото.
        if BIOMETRY_AVAILABLE:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name

            metrics = await asyncio.to_thread(analyze_face, tmp_path)
            if not metrics.get("face_found"):
                await status.edit_text(
                    "❌ Лицо не найдено. Кинь фото где ебало видно чётко: "
                    "свет, фокус, без очков."
                )
                return
            metrics_text = format_metrics(metrics)
            log.info("metrics: %s", metrics_text)
            await status.edit_text(f"📐 Биометрия: {metrics_text}\n\n🧠 Нейронка думает…")
        else:
            metrics_text = format_metrics({"face_found": False, "error": "biometry_unavailable"})

        # 2. VLM-разбор (тоже в треде — сетевые вызовы синхронные)
        verdict = await asyncio.to_thread(analyze_with_vlm, raw, metrics_text)

        header = f"📐 Замеры: {metrics_text}\n\n"
        # Режем длинные ответы под лимит TG (4096)
        chunk_size = 4000 - len(header)
        first = True
        for i in range(0, len(verdict), chunk_size):
            chunk = verdict[i : i + chunk_size]
            await msg.answer((header if first else "") + chunk)
            first = False
        await status.delete()
    except Exception as exc:  # noqa: BLE001 — юзер должен видеть ошибку текстом
        log.exception("photo handling failed")
        await msg.answer(f"💥 Упал с ошибкой: {exc}\nПопробуй другое фото.")
        try:
            await status.delete()
        except Exception:  # noqa: BLE001, S110
            pass
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@dp.message()
async def on_other(msg: Message) -> None:
    await msg.answer("Мне нужно именно <b>фото</b>, не текст. Кидай ебало 👇", parse_mode="HTML")


async def main() -> None:
    if not TELEGRAM_TOKEN:
        raise SystemExit("Нет TELEGRAM_TOKEN — создай .env по образцу .env.example")
    if not MODEL_API_KEY:
        raise SystemExit("Нет MODEL_API_KEY — возьми на https://z.ai (регистрация по email)")
    bot = Bot(token=TELEGRAM_TOKEN)

    # RENDER_EXTERNAL_URL задаёт сам Render; PUBLIC_URL — для любого другого
    # хостинга с публичным HTTPS (Hugging Face Spaces, Koyeb, VPS…).
    render_url = os.environ.get("RENDER_EXTERNAL_URL", "").strip() or os.environ.get(
        "PUBLIC_URL", ""
    ).strip()
    if render_url:
        # ☁️ Облачный режим (Render и любой хостинг с HTTPS-URL): webhook.
        # Телеграм сам присылает апдейты на URL — держать соединение не надо,
        # переживает сон free-тарифа.
        from aiohttp import web
        from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

        secret = os.environ.get("WEBHOOK_SECRET", "psl-webhook-secret-поменяй")
        hook_path = f"/webhook/{secret}"
        await bot.set_webhook(
            f"{render_url}{hook_path}", secret_token=secret, drop_pending_updates=True
        )
        app = web.Application()

        async def health(_request: object) -> object:
            return web.Response(text="ok")

        app.router.add_get("/", health)  # для Render health-check и cron-job.org анти-сна
        SimpleRequestHandler(dispatcher=dp, bot=bot, secret_token=secret).register(
            app, path=hook_path
        )
        setup_application(app, dp, bot=bot)
        port = int(os.environ.get("PORT", "8080"))
        log.info("webhook-режим: %s (порт %d)", render_url, port)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", port).start()
        await asyncio.Event().wait()  # висим вечно
    else:
        # 💻 Локальный режим: polling, полная биометрия если установлена.
        await bot.delete_webhook(drop_pending_updates=True)
        log.info(
            "polling-режим, модель=%s @ %s, биометрия=%s",
            MODEL_NAME,
            MODEL_BASE_URL,
            "ON" if BIOMETRY_AVAILABLE else "OFF",
        )
        await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
