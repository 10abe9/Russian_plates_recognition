import cv2 as cv
import pytesseract
import numpy as np
import cv2

pytesseract.pytesseract.tesseract_cmd = "C:\\Users\\qdaid\\Tesseract-OCR\\tesseract.exe"
tessdata_dir_config = r'--tessdata-dir C:/Users/qdaid/Tesseract-OCR/tessdata'


img1 ="C:\\Users\\qdaid\\Downloads\\Telegram Desktop\\01-393.jpg"
img2 ="C:\\Users\\qdaid\\Downloads\\Telegram Desktop\\01-541.jpg"


# Загрузка изображения
image_path = img2
img = cv.imread(image_path)


def image_filters(roi):
    # Шаг 2: Преобразование в оттенки серого
    # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    normalized_image = cv2.normalize(
        roi, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return normalized_image


def line_intersection(line1, line2):
    # Получаем координаты двух линий
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Решение пересечения двух прямых
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Линии не пересекаются

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    return int(px), int(py)


def preprocess_image(img):
    # Преобразование в оттенки серого
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Уменьшение шума с помощью Гауссового размытия
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Пороговая фильтрация методом Otsu
    _, binary = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Морфологическая обработка (закрытие)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    morph = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)

    return morph


def detect_license_plate(img):
    # Предобработка изображения
    processed_img = preprocess_image(img)

    # Поиск контуров
    contours, _ = cv.findContours(processed_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    detected_plates = []

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = w / h
        area = w * h

        # Фильтруем контуры по размеру и пропорциям (подходит для номерных знаков)
        if 2 < aspect_ratio < 6 and area > 500:  # Пропорции и минимальная площадь
            detected_plates.append((x, y, w, h))
            roi = img[y:y+h, x:x+w]
            cv.imshow("Roi before ", show_big_roi(roi))

            roi = image_filters(roi)

            cv.imshow("Roi after", show_big_roi(roi))

            # Распознавание текста
            text = pytesseract.image_to_string(roi, lang='rus', config='--psm 7 ' + tessdata_dir_config)
            cv.imwrite('debug_roi_try.png', roi)

            scale_percent = 500  # Увеличить на 200% (или 2 раза)
            width = int(roi.shape[1] * scale_percent / 100)
            height = int(roi.shape[0] * scale_percent / 100)
            dim = (width, height)

            # Масштабирование изображения
            roi = cv.resize(roi, dim, interpolation=cv.INTER_LINEAR)

            cv.imshow("Detected License Plate", roi)
            print("Распознанный номер:", text.strip())

    return img, detected_plates


def show_big_roi(roi):
    scale_percent = 500  # Увеличить на 200% (или 2 раза)
    width = int(roi.shape[1] * scale_percent / 100)
    height = int(roi.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Масштабирование изображения
    roi = cv.resize(roi, dim, interpolation=cv.INTER_LINEAR)

    return roi

# Запуск алгоритма
result_img, plates = detect_license_plate(img)

# Сохранение и показ результата
cv.imwrite('detected_plate.jpg', result_img)
# cv.imshow("Detected License Plate", result_img)
cv.waitKey(0)
cv.destroyAllWindows()

