import pprint

import numpy as np
from PIL import Image, ImageOps, ImageDraw

from utlis.processTools import preprocess_image, horizontal_projection, vertical_projection


class ComputerCharDivider:
    def __init__(self):
        pass

    def calculate_text_height(self, proj):
        heights = []
        in_text = False
        start = 0
        for i, val in enumerate(proj):
            if not in_text and val > 0:
                start = i
                in_text = True
            elif in_text and val == 0:
                heights.append(i - start)
                in_text = False
        return int(np.mean(heights))



    def segment_lines(self, binary, text_height):
        proj = horizontal_projection(binary)
        lines = []
        in_line = False
        start = 0
        threshold = np.max(proj) * 0.05  # 基于最大值的百分比动态设置阈值

        for i, val in enumerate(proj):
            if not in_line and val > threshold:
                start = i
                in_line = True
            elif in_line and val < threshold:
                if i - start > text_height * 0.5:  # 确保行高至少为预计行高的一半
                    lines.append(binary[start:i, :])
                in_line = False
        return lines

    def segment_characters(self, line, text_height):
        proj = vertical_projection(line)
        characters = []
        in_char = False
        start = 0
        width = text_height
        for i, val in enumerate(proj):
            if not in_char and val > 0:
                start = i
                in_char = True
            elif in_char and val < 500:
                if i - start > width * 0.9:
                    characters.append(line[:, start:i])
                    in_char = False
        if in_char:
            characters.append(line[:, start:])
        return characters

    def divide(self, img_file):
        divide_result = []
        binary = preprocess_image(img_file)
        text_height = self.calculate_text_height(horizontal_projection(binary))
        lines = self.segment_lines(binary, text_height)

        for i, line in enumerate(lines):
            # 保存行
            img = Image.fromarray(line).convert('L')
            inverted_img = Image.fromarray(255 - np.array(img))
            # inverted_img.save('./divideChar/invert' + str(i) + '.png')
            characters = self.segment_characters(line, text_height)
            divide_char = []
            j = 0
            for char in characters:
                binary_array = np.where(char < 128, 0, 255).astype(np.uint8)

                # 反转图像颜色
                inverted_array = 255 - binary_array
                rgb_array = np.stack([inverted_array] * 3, axis=-1)

                # 将所有非纯黑色像素填充为白色
                rgb_array[(rgb_array != [0, 0, 0]).any(axis=-1)] = [255, 255, 255]
                pprint.pprint(rgb_array)
                inverted_img = Image.fromarray(rgb_array, 'RGB')
                # inverted_img.save('./divideChar/invert' + str(i) + str(j) + '.png')
                # 将反转后的图像转换为RGB
                rgb_img = inverted_img.convert('RGB')
                # inverted_img.save('./divideChar/rgb' + str(i) + str(j) + '.png')
                # 添加五个像素的白边
                border_size = 2
                rgb_img_with_border = ImageOps.expand(rgb_img, border=border_size, fill='white')

                # 调整图片大小
                rgb_img_resized = rgb_img_with_border.resize((64, 64))
                divide_char.append(rgb_img_resized)
                # rgb_img_resized.save('./divideChar/test' + str(i) + str(j) + '.png')
                j += 1
            divide_result.append({
                'line': i,
                'characters': divide_char
            })
        return divide_result


class HandeWritingCharDivider:
    def __init__(self):
        pass

    def cut_image(self, img_file):
        binary = preprocess_image(img_file)
        binary = binary[2:-2, 2:-2]

        # 计算垂直投影
        vertical = vertical_projection(binary)

        # 计算水平投影
        horizontal = horizontal_projection(binary)

        # 寻找字符的垂直首尾边界
        v_start = 0
        v_end = len(vertical) - 1
        for i in range(len(vertical)):
            if vertical[i] > 0:
                v_start = i
                break
        for i in range(len(vertical) - 1, -1, -1):
            if vertical[i] > 0:
                v_end = i
                break

        # 寻找字符的水平首尾边界
        h_start = 0
        h_end = len(horizontal) - 1
        for i in range(len(horizontal)):
            if horizontal[i] > 0:
                h_start = i
                break
        for i in range(len(horizontal) - 1, -1, -1):
            if horizontal[i] > 0:
                h_end = i
                break

        # 根据确定的边界切割字符
        print(h_start, h_end, v_start, v_end)
        character_img = binary[h_start:h_end + 1, v_start:v_end + 1]
        img = Image.fromarray(character_img).convert('L')
        inverted_img = Image.fromarray(255 - np.array(img))
        inverted_img.save('./divideChar/invert.png')
        # 将反转后的灰度图转换为RGB
        rgb_img = inverted_img.convert('RGB')
        rgb_img = rgb_img.resize((64, 64))
        return [{'line': 0, 'characters': [rgb_img]}]





