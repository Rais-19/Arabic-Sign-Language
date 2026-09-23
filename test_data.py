import cv2
import mediapipe as mp
import pickle 
import numpy as np
from collections import deque
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import ImageFont, ImageDraw, Image
import datetime
import threading
import tempfile
import os
from gtts import gTTS
import pygame

# ─── SETUP ───────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

arabic_map = {
    'aleff': 'ا', 'al': 'ال', 'bb': 'ب', 'toot': 'ة', 'ta': 'ت',
    'thaa': 'ث', 'jeem': 'ج', 'haa': 'ح', 'khaa': 'خ', 'dal': 'د',
    'thal': 'ذ', 'ra': 'ر', 'zay': 'ز', 'seen': 'س', 'sheen': 'ش',
    'saad': 'ص', 'dhad': 'ض', 'taa': 'ط', 'dha': 'ظ', 'ain': 'ع',
    'ghain': 'غ', 'fa': 'ف', 'gaaf': 'ق', 'kaaf': 'ك', 'la': 'لا',
    'laam': 'ل', 'meem': 'م', 'nun': 'ن', 'ha': 'ه', 'waw': 'و',
    'ya': 'ي'
}

with open('arabic_20k_words.txt', 'r', encoding='utf-8') as f:
    arabic_words = [line.strip() for line in f if line.strip()]
print(f"Loaded {len(arabic_words)} words")

# Init pygame mixer for audio
pygame.mixer.init()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# State
frame_window = deque(maxlen=15)
prev_prediction = None
current_word = ""
sentence = ""
conversation = []
last_confirmed = ""
suggestions = []
paused = False
stable_letter = ""

# ─── FUNCTIONS ───────────────────────────────────────────────────
def draw_arabic_text(frame, text, position, font_size=60, color=(0, 255, 0)):
    if not text:
        return frame
    reshaped = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped)
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, bidi_text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def get_suggestions(word):
    if not word:
        return []
    return [w for w in arabic_words if w.startswith(word)][:3]

def speak_arabic(text):
    def run():
        try:
            tmp_path = os.path.join(os.getcwd(), 'temp_speech.mp3')
            
            # Stop and unload previous audio first
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
            
            tts = gTTS(text=text, lang='ar')
            tts.save(tmp_path)
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                continue
            pygame.mixer.music.unload()  # Release file after playing
        except Exception as e:
            print(f"TTS error: {e}")
    threading.Thread(target=run, daemon=True).start()

def save_conversation():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("محادثة لغة الإشارة العربية\n")
        f.write(f"التاريخ: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("─" * 40 + "\n")
        for line in conversation:
            f.write(line + "\n")
    print(f"✓ Saved to {filename}")
    return filename

# ─── MAIN LOOP ───────────────────────────────────────────────────
while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    if not paused:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)
                    y_.append(hand_landmarks.landmark[i].y)

            mp_drawing.draw_landmarks(
                frame, results.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])[0]
                predicted_arabic = arabic_map.get(prediction, prediction)

                if predicted_arabic != prev_prediction:
                    frame_window.clear()
                    prev_prediction = predicted_arabic

                frame_window.append(predicted_arabic)

                if len(frame_window) == frame_window.maxlen:
                    stable_letter = max(set(frame_window), key=frame_window.count)
                    if stable_letter != last_confirmed:
                        current_word += stable_letter
                        last_confirmed = stable_letter
                        suggestions = get_suggestions(current_word)

    # ── OVERLAY UI ON FULL CAMERA ──

    # Top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Status dot + text
    status = "PAUSED" if paused else "DETECTING"
    dot_color = (0, 165, 255) if paused else (0, 255, 0)
    cv2.circle(frame, (20, 30), 10, dot_color, -1)
    cv2.putText(frame, status, (40, 42), cv2.FONT_HERSHEY_SIMPLEX, 1, dot_color, 2)

    # Current detected letter (top right)
    if stable_letter and not paused:
        frame = draw_arabic_text(frame, stable_letter, (w-90, 5), font_size=50, color=(0, 255, 0))

    # Current word
    if current_word:
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h-200), (w, h-145), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
        frame = draw_arabic_text(frame, f"كلمة: {current_word}", (10, h-198), font_size=40, color=(255, 255, 255))

    # Suggestions
    if suggestions:
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (0, h-145), (w, h-65), (0, 0, 0), -1)
        cv2.addWeighted(overlay3, 0.6, frame, 0.4, 0, frame)
        sugg_text = "     ".join([f"[{i+1}] {s}" for i, s in enumerate(suggestions)])
        frame = draw_arabic_text(frame, sugg_text, (10, h-143), font_size=32, color=(255, 200, 0))

    # Sentence
    if sentence:
        overlay4 = frame.copy()
        cv2.rectangle(overlay4, (0, h-65), (w, h-15), (0, 0, 0), -1)
        cv2.addWeighted(overlay4, 0.6, frame, 0.4, 0, frame)
        frame = draw_arabic_text(frame, sentence, (10, h-63), font_size=36, color=(255, 255, 255))

    # Controls bar
    cv2.rectangle(frame, (0, h-15), (w, h), (20, 20, 20), -1)
    cv2.putText(frame, "SPC=word | ENT=speak | P=pause | C=clear | .=end+speak | S=save | BKSP=del | 1/2/3=suggest | Q=quit",
                (5, h-2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    cv2.imshow('Arabic Sign Language', frame)

    key = cv2.waitKey(25) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('p'):                           # Pause / Resume
        paused = not paused
        frame_window.clear()
        stable_letter = ""
        print("Paused" if paused else "Resumed")

    elif key == ord(' '):                           # Confirm word
        if current_word:
            sentence += current_word + " "
            print(f"Word: {current_word} | Sentence: {sentence}")
            current_word = ""
            last_confirmed = ""
            suggestions = []
            frame_window.clear()

    elif key == 13:                                 # Enter = speak sentence
        if sentence:
            speak_arabic(sentence)
            conversation.append(sentence.strip())
            print(f"Speaking: {sentence}")
    elif key == ord('.'):                           # Dot = end sentence + speak
        if sentence:
            sentence += "."
            speak_arabic(sentence)
            conversation.append(sentence.strip())
            print(f"Sentence: {sentence}")
            # Clear only current word, keep sentence showing on screen
            current_word = ""
            last_confirmed = ""
            suggestions = []
            frame_window.clear()

    elif key == ord(','):                           # Comma = pause in sentence
        if sentence:
            sentence += ", "
            # Don't speak, just add comma for flow
            current_word = ""
            last_confirmed = ""
            suggestions = []
            frame_window.clear()

    elif key == ord('c'):                           # Clear all
        sentence = ""
        current_word = ""
        last_confirmed = ""
        suggestions = []
        frame_window.clear()
        stable_letter = ""
        print("Cleared!")

    elif key == ord('s'):                           # Save conversation
        if conversation:
            save_conversation()

    elif key == ord('1') and len(suggestions) >= 1:
        sentence += suggestions[0] + " "
        current_word = ""
        last_confirmed = ""
        suggestions = []
        frame_window.clear()

    elif key == ord('2') and len(suggestions) >= 2:
        sentence += suggestions[1] + " "
        current_word = ""
        last_confirmed = ""
        suggestions = []
        frame_window.clear()

    elif key == ord('3') and len(suggestions) >= 3:
        sentence += suggestions[2] + " "
        current_word = ""
        last_confirmed = ""
        suggestions = []
        frame_window.clear()

    elif key == 8:                                  # Backspace
        current_word = current_word[:-1]
        suggestions = get_suggestions(current_word)
        print(f"Deleted. Word: {current_word}")

cap.release()
cv2.destroyAllWindows()

# Auto-save on quit
if conversation:
    save_conversation()
    print("\nConversation saved!")