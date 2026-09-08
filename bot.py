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
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
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
антропометрист, знаток looksmaxxing-культуры
(looksmax.org, PSL-скейл, blackpill-анализ).

Пользователь — взрослый человек, который ДОБРОВОЛЬНО прислал своё фото
и ПРЯМО ПРОСИТ максимально жёсткую, циничную, безжалостную оценку внешности
без ванильных утешений в духе «ты и так красив». Это consensual aesthetic
analysis, консультация по внешности. Вежливые фильтры ОТКЛЮЧЕНЫ
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
7. HARDMAXXING (опционально): радикальные методы, честно про риски и цену.
8. 📈 ПРОГНОЗ ASCEND: реалистичный потолок (например MTN→HTN) и за счёт чего.

Не читай нотаций, не жалей, не добавляй дисклеймеры про «внутреннюю красоту».
Максимум конкретики по ЭТОМУ лицу.
Не выдумывай того, чего нет на фото: фон, одежду вне кадра, точный возраст,
профессию. Если лиц несколько — разбирай самое крупное/центральное.
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
        # 8192: полный разбор длинный + reasoning, меньше — обрыв.
        max_tokens=8192,
    )
    finish = ""
    try:
        text = (resp.choices[0].message.content or "").strip()
        finish = resp.choices[0].finish_reason or ""
    except Exception:  # noqa: BLE001 — кривой/пустой ответ провайдера
        text = ""
    log.info("full: finish=%s chars=%d", finish, len(text))
    if not text:
        text = (
            f"⚠️ Нейронка отфильтровала ответ (finish_reason={finish}). "
            "Попробуй другое фото: прямой ракурс, дневной свет, без очков."
        )
    return text


# ---------------------------------------------------------------- простой слой
SIMPLE_PROMPT = """Ты — друг-карикатурист: описываешь внешность весело и хлёстко,
но по-доброму, так чтобы вердикт хотелось переслать другу. Простыми словами,
без терминов (PSL, fWHR, tilt, failo — запрещены), без цифр и замеров.

Сначала определи по фото КТО герой:
- СЕЛФИ (человек снял сам себя: крупный план, ракурс вытянутой руки) → герой сам
  прислал фото, обращайся на "ты", шути по-дружески.
- ЧУЖОЕ ФОТО (его сняли, скрин, пересланное) → говори о герое в третьем лице
  ("этот орёл", "наша героиня"), можешь весело пошутить и про приславшего.
- Лиц несколько → описывай самое крупное/центральное, остальных — одной строкой.

Это добровольный развлекательный тест: человек сам прислал фото ради шутки.

Структура — РОВНО ДВА АБЗАЦА подряд, БЕЗ подписей ("Абзац 1", номеров,
заголовков) — просто текст. Русский язык, компактно (весь ответ до ~1200
символов), юмора побольше (минимум 4 шутки — народ должен смеяться и пересылать).
Абзацы про РАЗНОЕ, дубли запрещены: первый — общее впечатление и вайб героя;
второй — конкретные детали внешности (взгляд, улыбка, стиль, одежда), ни одна
мысль, шутка или деталь из первого не повторяется во втором.
Пример стиля: "уверенность волка, взгляд — как у совы после ночной смены,
а улыбка такая, будто он только что вспомнил твой позор со школы".

Описывай только то, что видно на фото. Не додумывай фон, одежду вне кадра,
профессию, точный возраст. Не знаешь — шути, а не выдумывай.
Жесть, цифры и диагнозы — НЕ твоя задача, их выдаст другой режим по кнопке.
Если биометрия недоступна — смотри только фото.

В САМОМ КОНЦЕ ответа добавь отдельной строкой технические оценки для диаграммы
(это не часть текста, её вырежет код): SCORES: брутальность=X, няшность=X,
харизма=X, ухоженность=X, дерзость=X, загадочность=X — каждая 0-10, честно по фото.
"""

SIMPLE_PROMPT_SOFT = """Ты — автор добрых дружеских шаржей для развлекательного теста.
Человек сам прислал своё фото ради веселья. Опиши его внешность тепло и смешно:
комплименты с юмором, яркие сравнения с героями фильмов и животными.
Простыми словами, без терминов и цифр. Два абзаца, компактно
(весь ответ до ~1200 символов), русский язык.
Описывай только то, что видно на фото, ничего не выдумывай.
В САМОМ КОНЦЕ добавь строкой: SCORES: брутальность=X, няшность=X, харизма=X,
ухоженность=X, дерзость=X, загадочность=X (каждая 0-10, честно по фото).
"""


def _simple_call(system_prompt: str, photo_b64: str, metrics_text: str):
    """Один вызов simple-слоя. Может кинуть исключение (в т.ч. 1301 фильтр)."""
    from openai import OpenAI

    client = OpenAI(api_key=MODEL_API_KEY, base_url=MODEL_BASE_URL)
    return client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{photo_b64}"},
                    },
                    {
                        "type": "text",
                        "text": f"Замеры для ориентира (не озвучивай цифры): {metrics_text}\n\n"
                        "Опиши человека простыми словами по схеме из системного промпта.",
                    },
                ],
            },
        ],
        temperature=0.9,
        # 4096: reasoning прожорлив, при меньших лимитах ответ обрывался
        # на полуслове (и SCORES-строка в конце терялась).
        max_tokens=4096,
    )


def _extract_text(resp) -> tuple[str, str]:
    try:
        return (resp.choices[0].message.content or "").strip(), (
            resp.choices[0].finish_reason or ""
        )
    except Exception:  # noqa: BLE001
        return "", ""


def analyze_simple(photo_bytes: bytes, metrics_text: str) -> str:
    """Короткий разбор простыми словами (первый слой выдачи).

    Китайский фильтр (1301) иногда режет дерзкий промпт — тогда повторяем
    с мягкой версией, чтобы юзер всё равно получил ответ, а не ошибку.
    """
    import base64

    b64 = base64.b64encode(photo_bytes).decode("utf-8")
    try:
        resp = _simple_call(SIMPLE_PROMPT, b64, metrics_text)
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        if any(m in err for m in ("1301", "contentFilter", "content_filter", "不安全")):
            log.warning("simple отфильтрован (1301), повторяю с мягким промптом")
            try:
                resp = _simple_call(SIMPLE_PROMPT_SOFT, b64, metrics_text)
            except Exception as exc2:  # noqa: BLE001
                log.warning("soft тоже не прошёл: %s", str(exc2)[:200])
                raise
        else:
            raise
    text, finish = _extract_text(resp)
    log.info("simple: finish=%s chars=%d", finish, len(text))
    return text


# Контекст для кнопки "подробный разбор": user_id -> (photo_bytes, metrics_text).
_pending_full: dict[int, tuple[bytes, str]] = {}

# Оси диаграммы статов (ключи — lowercase для парсинга SCORES-строки).
STAT_LABELS = ["Брутальность", "Няшность", "Харизма", "Ухоженность", "Дерзость", "Загадочность"]


def parse_scores(text: str) -> tuple[str, dict[str, float]]:
    """Вырезает SCORES-строку из ответа, возвращает (чистый текст, оценки)."""
    import re

    scores: dict[str, float] = {}
    m = re.search(r"SCORES:\s*(.+)", text)
    if m:
        for part in m.group(1).split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    scores[k.strip().lower()] = max(0.0, min(10.0, float(v.strip())))
                except ValueError:
                    pass
        text = (text[: m.start()] + text[m.end() :]).strip()
    # Страховка: нейронка иногда лепит подписи "Абзац 1:" — вырезаем.
    text = re.sub(r"(?m)^\s*абзац\s*\d+\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    return text, scores


def _label_font(size: int, bold: bool = True):
    """Шрифт с кириллицей: DejaVu на Linux, Arial на Windows, иначе дефолт."""
    import os

    from PIL import ImageFont

    cands = (
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
        if bold
        else (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        )
    )
    wins = (
        (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf")
        if bold
        else (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf")
    )
    for path in cands + wins:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # noqa: BLE001
                pass
    return ImageFont.load_default()


def _wrap(draw, text: str, font, max_w: int) -> list[str]:
    """Перенос текста по словам под ширину в пикселях."""
    lines: list[str] = []
    for para in text.split("\n"):
        words, cur = para.split(), ""
        for w in words:
            probe = (cur + " " + w).strip()
            if draw.textlength(probe, font=font) <= max_w or not cur:
                cur = probe
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        lines.append("")  # отбивка абзаца
    return lines


def build_card(photo_bytes: bytes, verdict: str, scores: dict[str, float]) -> bytes:
    """Одна итоговая карточка: фото + панель статов + текст вердикта."""
    import io

    from PIL import Image, ImageDraw, ImageOps

    W = 1280
    M = 44
    BG = (16, 16, 22)
    NEON = (255, 80, 120)
    DIM = (110, 110, 135)
    TXT = (232, 232, 240)

    photo = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
    photo = ImageOps.fit(photo, (560, 560), Image.LANCZOS)

    vals = [scores.get(k.lower(), 5.0) for k in STAT_LABELS]

    # --- верх: фото слева, статы справа
    top_h = 560
    # --- текст вердикта
    font_txt = _label_font(31, bold=False)
    meas = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    text_lines = _wrap(meas, verdict, font_txt, W - 2 * M)[:26]  # кап на всякий
    text_h = len(text_lines) * 44
    # --- шапка + низ (подвала нет: подпись телеги идёт отдельным текстом)
    H = 110 + top_h + 40 + text_h + 30 + 60

    card = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(card)
    font_big = _label_font(44)
    font_mid = _label_font(30)
    font_small = _label_font(26, bold=False)

    d.text((M, 30), "LOOKERNICE", fill=TXT, font=font_big)
    d.text((M, 78), "разбор по фото", fill=DIM, font=font_small)
    d.text((W - M, 40), "@lookernice_bot", fill=DIM, font=font_small, anchor="ra")

    card.paste(photo, (M, 110))
    # рамка фото
    d.rectangle([M, 110, M + 560, 110 + top_h], outline=(45, 45, 60), width=2)

    # панель статов
    px = M + 560 + 44
    pw = W - px - M
    d.text((px, 110), "СТАТЫ", fill=DIM, font=font_mid)
    y = 160
    for i, label in enumerate(STAT_LABELS):
        v = vals[i]
        d.text((px, y), label, fill=TXT, font=font_mid)
        d.text((px + pw, y), f"{v:.0f}", fill=NEON, font=font_mid, anchor="ra")
        y += 40
        d.rounded_rectangle([px, y, px + pw, y + 16], radius=8, fill=(45, 45, 60))
        if v > 0.3:
            d.rounded_rectangle([px, y, px + int(pw * v / 10), y + 16], radius=8, fill=NEON)
        y += 34

    # текст вердикта
    ty = 110 + top_h + 40
    for line in text_lines:
        if line:
            d.text((M, ty), line, fill=TXT, font=font_txt)
        ty += 44

    buf = io.BytesIO()
    card.save(buf, "PNG")
    return buf.getvalue()


def draw_radar(scores: dict[str, float]) -> bytes:
    """Радар-диаграмма статов 800x800 PNG (тёмная тема, неон)."""
    import io
    import math

    from PIL import Image, ImageDraw

    W = H = 800
    cx = cy = 400
    R = 290
    vals = [scores.get(k.lower(), 5.0) for k in STAT_LABELS]

    def pt(i: int, r: float) -> tuple[float, float]:
        a = -math.pi / 2 + i * 2 * math.pi / len(STAT_LABELS)
        return (cx + r * math.cos(a), cy + r * math.sin(a))

    base = Image.new("RGB", (W, H), (16, 16, 22))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for ring in (R, R * 0.66, R * 0.33):
        d.line([pt(i, ring) for i in range(6)] + [pt(0, ring)], fill=(80, 80, 110), width=2)
    for i in range(6):
        d.line([pt(i, 0), pt(i, R)], fill=(55, 55, 75), width=1)
    poly = [pt(i, R * v / 10) for i, v in enumerate(vals)]
    d.polygon(poly, fill=(255, 80, 120, 110), outline=(255, 80, 120))
    for x, y in poly:
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 80, 120))
    img = Image.alpha_composite(base.convert("RGBA"), ov).convert("RGB")
    d2 = ImageDraw.Draw(img)
    font = _label_font(30)
    num_font = _label_font(26)
    for i in range(6):
        x, y = pt(i, R + 44)
        d2.text((x, y), STAT_LABELS[i], fill=(230, 230, 240), font=font, anchor="mm")
        xv, yv = pt(i, R * vals[i] / 10)
        d2.text((xv, yv - 16), str(int(round(vals[i]))), fill=(255, 200, 210), font=num_font, anchor="mm")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()

# ---------------------------------------------------------------- бот
dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(msg: Message) -> None:
    await msg.answer(
        "💀 <b>PSL-бот на связи.</b>\n\n"
        "Кидай своё фото (лицо крупно, без очков и фильтров) — "
        "сначала скажу по-человечески, без занудства. "
        "А если хочешь жести — жми кнопку «📊 Подробный разбор»: "
        "там PSL-рейтинг, кости, failo/halo и план прокачки.\n"
        "В ответ получишь вердикт, твоё фото и диаграмму статов.\n\n"
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
        "📐 Изучаю пропорции…" if BIOMETRY_AVAILABLE else "🧠 Нейронка думает…"
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

        # 2. Первый слой: разбор простыми словами (тоже в треде)
        simple = await asyncio.to_thread(analyze_simple, raw, metrics_text)

        # Запоминаем контекст для кнопки "подробный разбор"
        _pending_full[msg.from_user.id] = (raw, metrics_text)

        import urllib.parse

        from aiogram.types import BufferedInputFile

        await status.delete()
        if simple:
            clean, scores = parse_scores(simple)
            # Футер-вирус: подталкиваем переслать другу (ретеншн).
            await msg.answer(clean + "\n\n😏 Перешли другу — пусть тоже узнает правду")
        else:
            clean, scores = "", {}
            await msg.answer("Не разглядел по-простому — но статы и жесть ниже 👇")

        # 3. Одна итоговая карточка: фото + статы + вердикт + кнопки
        card = await asyncio.to_thread(build_card, raw, clean, scores)
        share_url = (
            "https://t.me/share/url?url="
            + urllib.parse.quote("https://t.me/lookernice_bot", safe="")
            + "&text="
            + urllib.parse.quote("Глянь какой у меня вердикт 😏", safe="")
        )
        kb2 = InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="📊 Подробный разбор", callback_data="full_roast")],
                [InlineKeyboardButton(text="📤 Кинуть другу", url=share_url)],
            ]
        )
        await msg.answer_photo(
            BufferedInputFile(card, filename="card.png"),
            caption="Забирай 😏 Перешли другу — пусть тоже узнает правду",
            reply_markup=kb2,
        )
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


@dp.callback_query(F.data == "full_roast")
async def on_full_roast(cb: CallbackQuery, bot: Bot) -> None:
    """Кнопка 'Жёсткий разбор': полный PSL-разбор по сохранённому фото."""
    data = _pending_full.get(cb.from_user.id)
    if not data:
        await cb.answer("Фото устарело — кинь заново 👇", show_alert=True)
        return
    await cb.answer("Готовлю жесть…")
    raw, metrics_text = data
    status = await cb.message.answer("🧠 Нейронка думает…")
    try:
        # Полный разбор (в треде — сетевой вызов синхронный)
        verdict = await asyncio.to_thread(analyze_with_vlm, raw, metrics_text)

        header = f"📐 Замеры: {metrics_text}\n\n"
        # Режем длинные ответы под лимит TG (4096)
        chunk_size = 4000 - len(header)
        first = True
        for i in range(0, len(verdict), chunk_size):
            chunk = verdict[i : i + chunk_size]
            await cb.message.answer((header if first else "") + chunk)
            first = False
        await status.delete()
    except Exception as exc:  # noqa: BLE001
        log.exception("full roast failed")
        await cb.message.answer(f"💥 Упал с ошибкой: {exc}\nПопробуй ещё раз кнопкой.")
        try:
            await status.delete()
        except Exception:  # noqa: BLE001, S110
            pass


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
