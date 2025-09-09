import numpy as np
from PIL import Image
import math


class CrossStitchGenerator:
    def __init__(self):
        # Упрощенная палитра DMC (RGB -> код DMC)
        self.dmc_palette = {
            (255, 0, 0): ("DMC 321", "Красный"),
            (0, 255, 0): ("DMC 699", "Зеленый"),
            (0, 0, 255): ("DMC 798", "Синий"),
            (255, 255, 0): ("DMC 444", "Желтый"),
            (255, 0, 255): ("DMC 554", "Фиолетовый"),
            (0, 255, 255): ("DMC 597", "Бирюзовый"),
            (128, 128, 128): ("DMC 317", "Серый"),
            (0, 0, 0): ("DMC 310", "Черный"),
            (255, 255, 255): ("DMC Blanc", "Белый"),
            (255, 165, 0): ("DMC 606", "Оранжевый"),
            (165, 42, 42): ("DMC 301", "Коричневый"),
            (128, 0, 128): ("DMC 550", "Пурпурный"),
            (255, 192, 203): ("DMC 776", "Розовый"),
            (0, 128, 0): ("DMC 701", "Темно-зеленый"),
            (0, 0, 128): ("DMC 823", "Темно-синий"),
            (128, 0, 0): ("DMC 498", "Темно-красный"),
            (192, 192, 192): ("DMC 762", "Серебряный"),
            (255, 215, 0): ("DMC 444", "Золотой"),
            (139, 69, 19): ("DMC 434", "Коричневый"),
            (75, 0, 130): ("DMC 333", "Фиолетовый")
        }

        # Символы для обозначения цветов
        self.symbols = ['■', '□', '●', '○', '▲', '△', '▼', '▽', '◆', '◇', '★', '☆',
                        '♠', '♣', '♥', '♦', '◈', '◉', '◎', '✦', '✧', '✩', '✪', '✫']

    def load_and_resize_image(self, image_path, max_stitches):
        """Загрузка и изменение размера изображения"""
        try:
            image = Image.open(image_path).convert('RGB')

            # Определяем новый размер с сохранением пропорций
            width, height = image.size
            ratio = min(max_stitches / width, max_stitches / height)
            new_width = max(1, int(width * ratio))
            new_height = max(1, int(height * ratio))

            # Уменьшаем изображение
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)
            return resized_image, new_width, new_height
        except Exception as e:
            print(f"Ошибка загрузки изображения: {e}")
            return None, 0, 0

    def find_closest_dmc(self, rgb_color):
        """Нахождение ближайшего цвета DMC"""
        min_distance = float('inf')
        closest_dmc = None

        # Преобразуем в int чтобы избежать переполнения
        r, g, b = int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])

        for dmc_rgb, (dmc_code, dmc_name) in self.dmc_palette.items():
            # Простое вычисление расстояния между цветами
            dr = r - dmc_rgb[0]
            dg = g - dmc_rgb[1]
            db = b - dmc_rgb[2]

            distance = math.sqrt(dr * dr + dg * dg + db * db)

            if distance < min_distance:
                min_distance = distance
                closest_dmc = (dmc_code, dmc_name, dmc_rgb)

        return closest_dmc

    def reduce_colors(self, image, max_colors):
        """Упрощенное уменьшение количества цветов"""
        pixels = np.array(image)
        unique_dmc_colors = {}

        # Собираем уникальные DMC цвета
        for y in range(pixels.shape[0]):
            for x in range(pixels.shape[1]):
                rgb = tuple(pixels[y, x])
                dmc_info = self.find_closest_dmc(rgb)
                if dmc_info:
                    dmc_code = dmc_info[0]
                    if dmc_code not in unique_dmc_colors:
                        unique_dmc_colors[dmc_code] = dmc_info

        # Если цветов больше чем нужно, берем первые max_colors
        if len(unique_dmc_colors) > max_colors:
            limited_colors = dict(list(unique_dmc_colors.items())[:max_colors])
        else:
            limited_colors = unique_dmc_colors

        return limited_colors

    def create_pattern(self, image, max_colors):
        """Создание схемы вышивки"""
        width, height = image.size
        pixels = np.array(image)

        # Уменьшаем количество цветов
        color_mapping = self.reduce_colors(image, max_colors)

        # Создаем обратное отображение: DMC -> символ
        dmc_to_symbol = {}
        symbol_index = 0

        for dmc_code, (_, dmc_name, dmc_rgb) in color_mapping.items():
            if dmc_code not in dmc_to_symbol:
                dmc_to_symbol[dmc_code] = {
                    'symbol': self.symbols[symbol_index % len(self.symbols)],
                    'name': dmc_name,
                    'rgb': dmc_rgb
                }
                symbol_index += 1

        # Создаем схему
        pattern_grid = []
        for y in range(height):
            row = []
            for x in range(width):
                rgb = tuple(pixels[y, x])
                dmc_info = self.find_closest_dmc(rgb)
                if dmc_info and dmc_info[0] in dmc_to_symbol:
                    symbol = dmc_to_symbol[dmc_info[0]]['symbol']
                    row.append(symbol)
                else:
                    # Используем первый доступный символ
                    first_symbol = list(dmc_to_symbol.values())[0]['symbol'] if dmc_to_symbol else '?'
                    row.append(first_symbol)
            pattern_grid.append(row)

        return pattern_grid, dmc_to_symbol

    def add_grid_lines(self, pattern_grid):
        """Добавление счетных линий"""
        if not pattern_grid or not pattern_grid[0]:
            return "Пустая схема"

        height = len(pattern_grid)
        width = len(pattern_grid[0])

        result = []

        # Для больших схем упрощаем отображение
        if width > 50:
            # Показываем только каждую 5-ю линию
            result.append(f"Схема слишком большая для отображения в консоли ({width}x{height})")
            result.append("Рекомендуется сохранить в файл")
            return "\n".join(result)

        # Добавляем нумерацию столбцов
        header = "   "
        for x in range(width):
            if x % 10 == 0:
                header += f"{x:2d}"
            else:
                header += "  "
        result.append(header)

        # Добавляем строки с нумерацией
        for y in range(height):
            if y > 100:  # Ограничиваем количество строк для отображения
                result.append(f"... (показано 100 из {height} строк)")
                break

            row_str = f"{y:2d} "
            for x in range(width):
                # Добавляем вертикальные линии каждые 10 крестиков
                if x % 10 == 0 and x != 0:
                    row_str += "|"
                row_str += pattern_grid[y][x] + " "

            result.append(row_str)

            # Добавляем горизонтальные линии каждые 10 крестиков
            if y % 10 == 9 and y != height - 1 and y < 100:
                line = "   "
                for x in range(width):
                    if x % 10 == 0 and x != 0:
                        line += "+"
                    line += "--"
                result.append(line)

        return "\n".join(result)

    def create_legend(self, dmc_to_symbol):
        """Создание легенды"""
        legend = ["\n=== ЛЕГЕНДА ==="]
        for dmc_code, info in dmc_to_symbol.items():
            legend.append(f"{info['symbol']} : {dmc_code} ({info['name']})")
        return "\n".join(legend)

    def save_to_file(self, pattern_grid, dmc_to_symbol, filename="cross_stitch_pattern.txt"):
        """Сохранение схемы в файл"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("СХЕМА ДЛЯ ВЫШИВКИ КРЕСТИКОМ\n")
                f.write("=" * 50 + "\n")

                height = len(pattern_grid)
                width = len(pattern_grid[0])
                f.write(f"Размер: {width} x {height} крестиков\n")
                f.write(f"Цветов: {len(dmc_to_symbol)}\n\n")

                # Записываем схему
                for y in range(height):
                    if y % 10 == 0 and y != 0:
                        f.write("-" * (width * 2 + 10) + "\n")

                    row_str = f"{y:3d}: "
                    for x in range(width):
                        if x % 10 == 0 and x != 0:
                            row_str += "|"
                        row_str += pattern_grid[y][x]
                    f.write(row_str + "\n")

                # Записываем легенду
                f.write("\n" + "=" * 50 + "\n")
                f.write("ЛЕГЕНДА:\n")
                for dmc_code, info in dmc_to_symbol.items():
                    f.write(f"{info['symbol']} : {dmc_code} ({info['name']})\n")

            print(f"Схема сохранена в файл: {filename}")
            return True
        except Exception as e:
            print(f"Ошибка сохранения файла: {e}")
            return False

    def generate_pattern(self, image_path, max_stitches=100, max_colors=12):
        """Основная функция генерации схемы"""
        print(f"Обрабатываем изображение: {image_path}")

        # Загрузка и изменение размера
        image, width, height = self.load_and_resize_image(image_path, max_stitches)
        if image is None:
            return None, None, 0, 0

        print(f"Размер схемы: {width} x {height} крестиков")

        # Создание паттерна
        pattern_grid, dmc_to_symbol = self.create_pattern(image, max_colors)

        # Добавление сетки
        pattern_with_grid = self.add_grid_lines(pattern_grid)

        # Создание легенды
        legend = self.create_legend(dmc_to_symbol)

        # Автоматическое сохранение в файл для больших схем
        if width > 50 or height > 50:
            self.save_to_file(pattern_grid, dmc_to_symbol)

        return pattern_with_grid, legend, width, height


# Использование программы
def main():
    generator = CrossStitchGenerator()

    # Параметры - увеличиваем размер и количество цветов
    image_path = "test.jpg"  # ваше изображение
    max_stitches = 100  # БОЛЬШЕ крестиков! (можно до 200-300)
    max_colors = 12  # больше цветов

    try:
        pattern, legend, width, height = generator.generate_pattern(
            image_path, max_stitches, max_colors
        )

        if pattern:
            print("=" * 50)
            print("СХЕМА ДЛЯ ВЫШИВКИ КРЕСТИКОМ")
            print("=" * 50)
            print(f"Размер: {width} x {height} крестиков")
            print(f"Цветов: {max_colors}")
            print("\n" + pattern)
            print(legend)

            # Предлагаем сохранить в файл
            if width <= 50:
                save = input("\nСохранить схему в файл? (y/n): ")
                if save.lower() == 'y':
                    generator.save_to_file(pattern.split('\n'), legend.split('\n'))
        else:
            print("Не удалось создать схему.")

    except Exception as e:
        print(f"Ошибка при выполнении: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()