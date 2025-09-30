import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import os
import json
from sklearn.cluster import KMeans
from scipy.spatial import KDTree


class CrossStitchGenerator:
    """
    Класс для генерации схемы вышивки крестиком из изображения.
    """

    def __init__(self, dmc_palette_file="dmc_colors.json"):
        """
        Инициализация генератора. Загружает палитру DMC и подготавливает данные.
        """
        self.dmc_palette = self._load_dmc_palette(dmc_palette_file)
        self.dmc_codes, self.dmc_rgb_values = self._prepare_palette_for_tree()
        self.dmc_tree = KDTree(self.dmc_rgb_values)

        # Символы для обозначения цветов. Можно расширить.
        self.symbols = [
            '■', '□', '●', '○', '▲', '△', '▼', '▽', '◆', '◇',
            '★', '☆', '♠', '♣', '♥', '♦', '◈', '◉', '◎', '✦',
            '✶', '✷', '✸', '✹', '✺', '✻', '✼', '✽', '✾', '❀',
            '✢', '✣', '✤', '✥', '✚', '✗', '✘', '✙', '∓', '⊕',
            '⊗', '⊙', '⊚', '⊛', '⊜', '⊝', '⊞', '⊟', '⊠', '⊡',
            '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '[', '\\', ']', '^', '_', '`',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            '{', '|', '}', '~',
            '¡', '¢', '£', '¤', '¥', '¦', '§', '¨', '©', 'ª', '«', '¬', '®', '¯', '°', '±', '²', '³', '´', 'µ', '¶', '·', '¸', '¹', 'º', '»', '¼', '½', '¾', '¿',
            'À', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'Æ', 'Ç', 'È', 'É', 'Ê', 'Ë', 'Ì', 'Í', 'Î', 'Ï', 'Ð', 'Ñ', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', '×', 'Ø', 'Ù', 'Ú', 'Û', 'Ü', 'Ý', 'Þ', 'ß',
            'à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï', 'ð', 'ñ', 'ò', 'ó', 'ô', 'õ', 'ö', '÷', 'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'ÿ',
            'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ж', 'З', 'И', 'Й', 'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф', 'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я',
            'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я',
            '№', 'ё', 'Ё',
            '≈', '≠', '≡', '≤', '≥', '∫', '∑', '∏', '√', '∝', '∞', '∇', '∂',
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
            'Ω', 'Δ', 'Φ', 'Π', 'Ψ', 'Σ', 'Ξ', 'Λ', 'Γ', 'Θ', 'Ψ',
        ]

    def _load_dmc_palette(self, filename):
        """
        Загружает палитру цветов DMC из JSON-файла.
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Ошибка: Файл палитры '{filename}' не найден.")
            return []
        except json.JSONDecodeError:
            print(f"Ошибка: Некорректный формат JSON в файле '{filename}'.")
            return []

    def _prepare_palette_for_tree(self):
        """
        Подготавливает данные палитры для быстрого поиска с помощью KDTree.
        """
        codes = []
        rgb_values = []
        for color in self.dmc_palette:
            codes.append((color['code'], color['name'], tuple(color['rgb'])))
            rgb_values.append(color['rgb'])
        return codes, np.array(rgb_values)

    def load_and_resize_image(self, image_path, max_stitches):
        """
        Загружает и изменяет размер изображения, сохраняя пропорции.
        """
        try:
            image = Image.open(image_path).convert('RGB')
            width, height = image.size

            # Определяем новый размер, чтобы одна из сторон была равна max_stitches
            if width > height:
                new_width = max_stitches
                new_height = int(height * (new_width / width))
            else:
                new_height = max_stitches
                new_width = int(width * (new_height / height))

            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            return resized_image, new_width, new_height
        except Exception as e:
            print(f"Ошибка загрузки или изменения размера изображения: {e}")
            return None, 0, 0

    def find_closest_dmc(self, rgb_color):
        """
        Находит ближайший цвет DMC с помощью KDTree.
        """
        distance, index = self.dmc_tree.query(rgb_color)
        return self.dmc_codes[index]

    def get_text_color_for_background(self, bg_color):
        """
        Определяет цвет текста (черный или белый) для лучшей читаемости.
        """
        r, g, b = bg_color
        # Вычисление яркости (luminance)
        brightness = (r * 0.299 + g * 0.587 + b * 0.114)
        return (0, 0, 0) if brightness > 186 else (255, 255, 255)

    def generate_pattern(self, image_path, max_stitches=150, max_colors=50, cell_size=20):
        """
        Основная функция для генерации схемы.
        """
        print("---")
        print(f"🛠️  Начинаем генерацию схемы для: '{image_path}'")
        print(f"Разрешение схемы: {max_stitches} крестиков по большей стороне")
        print(f"Максимальное количество цветов: {max_colors}")
        print("---")

        image, width, height = self.load_and_resize_image(image_path, max_stitches)
        if image is None:
            return

        pixels = np.array(image).reshape(-1, 3)

        # Применяем K-Means для уменьшения количества цветов
        print("🎨 Применяем K-Means для уменьшения количества цветов...")
        kmeans = KMeans(n_clusters=max_colors, random_state=0, n_init='auto')
        kmeans.fit(pixels)

        # Получаем центры кластеров и метки для каждого пикселя
        cluster_centers = kmeans.cluster_centers_.astype(int)
        pixel_labels = kmeans.labels_

        # Создаем словарь для маппинга кластера на DMC-цвет и символ
        cluster_to_dmc_info = {}
        unique_dmc_info = {}
        for i, center in enumerate(cluster_centers):
            dmc_code, dmc_name, dmc_rgb = self.find_closest_dmc(tuple(center))
            if dmc_code not in unique_dmc_info:
                symbol = self.symbols[len(unique_dmc_info) % len(self.symbols)]
                unique_dmc_info[dmc_code] = {
                    'code': dmc_code,
                    'name': dmc_name,
                    'rgb': dmc_rgb,
                    'symbol': symbol,
                    'count': 0
                }
            cluster_to_dmc_info[i] = unique_dmc_info[dmc_code]

        # Подсчитываем количество крестиков для каждого цвета
        for label in pixel_labels:
            dmc_info = cluster_to_dmc_info.get(label)
            if dmc_info:
                dmc_info['count'] += 1

        # Создаем изображение схемы
        img_width = width * cell_size + 200
        img_height = height * cell_size + 200 + len(unique_dmc_info) * 40

        pattern_image = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(pattern_image)

        # Загружаем шрифт
        try:
            font_path = "arial.ttf"
            if os.name == 'nt':  # Windows
                font_path = "C:/Windows/Fonts/Arial.ttf"
            elif os.name == 'posix':  # Mac, Linux
                font_path = "/Library/Fonts/Arial.ttf"

            title_font = ImageFont.truetype(font_path, 20)
            symbol_font = ImageFont.truetype(font_path, int(cell_size * 0.6))
            legend_font = ImageFont.truetype(font_path, 14)
        except Exception:
            print("❌ Шрифт Arial не найден. Используется шрифт по умолчанию.")
            title_font = ImageFont.load_default()
            symbol_font = ImageFont.load_default()
            legend_font = ImageFont.load_default()

        # Рисуем заголовок
        title = f"СХЕМА ВЫШИВКИ КРЕСТИКОМ - {width}x{height} КРЕСТИКОВ"
        draw.text(((img_width - draw.textlength(title, font=title_font)) / 2, 20),
                  title, fill='black', font=title_font)

        # Рисуем нумерацию и сетку
        self._draw_grid_and_numbers(draw, width, height, cell_size, legend_font)

        # Рисуем крестики с символами
        self._draw_cross_stitches(draw, pixel_labels, cluster_to_dmc_info, width, height, cell_size, symbol_font)

        # Рисуем легенду
        self._draw_legend(draw, unique_dmc_info, img_width, height, cell_size, legend_font)

        # Сохранение и вывод информации
        output_filename = f"scheme_{width}x{height}_{len(unique_dmc_info)}colors.png"
        self._save_pattern_image(pattern_image, output_filename, width, height, len(unique_dmc_info))

    def _draw_grid_and_numbers(self, draw, width, height, cell_size, font):
        """Рисует сетку и нумерацию."""
        x_offset, y_offset = 100, 100
        for y in range(height + 1):
            y_pos = y * cell_size + y_offset
            draw.line([(x_offset, y_pos), (width * cell_size + x_offset, y_pos)],
                      fill='lightgray' if y % 10 != 0 else 'gray', width=1 if y % 10 != 0 else 2)
            if y % 10 == 0 and y > 0:
                draw.text((x_offset - 30, y_pos - cell_size / 2 - 7), str(y), fill='black', font=font)

        for x in range(width + 1):
            x_pos = x * cell_size + x_offset
            draw.line([(x_pos, y_offset), (x_pos, height * cell_size + y_offset)],
                      fill='lightgray' if x % 10 != 0 else 'gray', width=1 if x % 10 != 0 else 2)
            if x % 10 == 0 and x > 0:
                draw.text((x_pos - cell_size / 2 - 7, y_offset - 30), str(x), fill='black', font=font)

    def _draw_cross_stitches(self, draw, pixel_labels, cluster_to_dmc_info, width, height, cell_size, font):
        """Рисует каждую клетку схемы."""
        x_offset, y_offset = 100, 100
        for i, label in enumerate(pixel_labels):
            x = i % width
            y = i // width

            dmc_info = cluster_to_dmc_info.get(label)
            if not dmc_info:
                continue

            color_rgb = dmc_info['rgb']
            symbol = dmc_info['symbol']
            text_color = self.get_text_color_for_background(color_rgb)

            x1, y1 = x * cell_size + x_offset, y * cell_size + y_offset
            x2, y2 = x1 + cell_size, y1 + cell_size

            draw.rectangle([x1, y1, x2, y2], fill=color_rgb)

            bbox = draw.textbbox((0, 0), symbol, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = x1 + (cell_size - text_width) / 2
            text_y = y1 + (cell_size - text_height) / 2
            draw.text((text_x, text_y), symbol, fill=text_color, font=font)

    def _draw_legend(self, draw, dmc_info, img_width, height, cell_size, font):
        """
        Исправленная функция для рисования легенды.
        Динамически рассчитывает позиции элементов, чтобы избежать наложения.
        """
        sorted_colors = sorted(dmc_info.values(), key=lambda x: x['count'], reverse=True)

        legend_x, legend_y = 100, height * cell_size + 150
        current_x = legend_x
        row_y = legend_y

        for info in sorted_colors:
            # Текст, который будет отображаться в легенде
            text_content = f"{info['symbol']} {info['code']} ({info['name']}) - {info['count']} шт."

            # Рассчитываем ширину текста
            text_bbox = draw.textbbox((0, 0), text_content, font=font)
            text_width = text_bbox[2] - text_bbox[0]

            # Ширина элемента легенды (квадрат + текст + небольшой отступ)
            item_width = 30 + text_width + 20

            # Проверяем, помещается ли элемент в текущую строку
            if current_x + item_width > img_width - 100:  # 100 - это отступ справа
                # Если не помещается, переходим на новую строку
                current_x = legend_x
                row_y += 40  # Увеличиваем y-координату для новой строки

            # Рисуем цветной квадрат
            draw.rectangle([current_x, row_y, current_x + 20, row_y + 20], fill=info['rgb'])

            # Рисуем текст
            draw.text((current_x + 30, row_y), text_content, fill='black', font=font)

            # Обновляем x-координату для следующего элемента
            current_x += item_width

    def _save_pattern_image(self, pattern_image, filename, width, height, colors_count):
        """Сохраняет изображение и выводит информацию."""
        try:
            pattern_image.save(filename, quality=95, dpi=(300, 300))
            print("---")
            print(f"✅ Схема успешно создана!")
            print(f"📐 Размер: {width} × {height} крестиков")
            print(f"🎨 Цветов: {colors_count}")
            print(f"💾 Файл сохранен: {filename}")
            print(f"Размер файла: {os.path.getsize(filename) / 1024:.2f} KB")
            print("---")
        except Exception as e:
            print(f"❌ Ошибка сохранения файла: {e}")


if __name__ == "__main__":
    generator = CrossStitchGenerator()
    generator.generate_pattern(
        image_path="test.jpg",
        max_stitches=120,
        max_colors=20,
        cell_size=30
    )