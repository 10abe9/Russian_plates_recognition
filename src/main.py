from __future__ import print_function
import cv2 as cv
import pytesseract

# Укажите путь к Tesseract (если требуется)
pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\qdaid\\Tesseract-OCR\\tesseract.exe"
tessdata_dir_config = r'--tessdata-dir C:/Users/qdaid/Tesseract-OCR/tessdata'


def detectAndDisplay(frame):
    # Переводим кадр в оттенки серого
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    # Применяем Гауссово размытие
    blur = cv.GaussianBlur(frame_gray, (5, 5), 0)

    # Применяем пороговую обработку по методу Отсу
    ret3, th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)# Распознаем автомобильные номера
    morph_image = th3
    plates = plate_cascade.detectMultiScale(morph_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(plates) == 0:
        print("Номерные знаки не найдены.")

    for (x, y, w, h) in plates:
        # Уменьшаем размеры ROI для более точного захвата номерного знака
        padding = 6  # Параметр для уменьшения размера ROI
        x, y, w, h = x + padding, y + padding, w - 2 * padding, h - 2 * padding

        # Отметим область номерного знака
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Обрезаем изображение для обработки
        roi = morph_image[y:y + h, x:x + w]

        if roi.size > 0:
            cv.imwrite('debug_roi.png', roi)  # Сохраняем ROI для отладки

            try:
                # Пример распознавания текста с указанием папки tessdata
                text = pytesseract.image_to_string(roi, lang='rus+eng', config='--psm 8 ' + tessdata_dir_config)
                print("Распознанный номер:", text.strip())

                # Пост-обработка текста, чтобы избавиться от лишних символов
                filtered_text = ''.join([char for char in text if char.isalnum() or char == ' '])
                print("Отфильтрованный номер:", filtered_text.strip())

            except pytesseract.TesseractError as e:
                print("Ошибка Tesseract:", e)
        else:
            print("ROI пустой или некорректный.")

    # Отображаем кадр
    cv.imshow("Capture - License Plate Detection", frame)


# Загружаем каскад для автомобильных номеров
plate_cascade_name = './data/haarcascade_russian_plate_number.xml'
plate_cascade = cv.CascadeClassifier()

if not plate_cascade.load(cv.samples.findFile(plate_cascade_name)):
    print("--(!)Error loading plate cascade")
    exit(0)

# Захват видеопотока
cap = cv.VideoCapture(0)
if not cap.isOpened:
    print("--(!)Error opening video capture")
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print("--(!) No captured frame -- Break!")
        break
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:  # Выход по нажатию ESC
        break

cap.release()
cv.destroyAllWindows()
