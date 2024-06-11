import base64
import os

from flask import Flask, request, jsonify

from flask_cors import cross_origin

from classification.utils import build_single_text, build_iterator
from model.run import ClassifyConfig,ClassifyModel
from utlis.charDivide import *
from model.chineseOcr import *

app = Flask(__name__)

# 加载模型
# 获取绝对路径
computer_ocr_model = ChineseOcrModel(os.path.abspath('./model/pth/computer_2000_turn20_99.pth'), 2000)
handWriting_ocr_model = ChineseOcrModel(os.path.abspath('./model/pth/handWriting.pth'), 3754)

classify_config = ClassifyConfig(os.path.abspath('./model/THUCNews'), 'embedding_SougouNews.npz')
classify_model = ClassifyModel(classify_config)


@app.route('/computer_ocr', methods=['POST'])
@cross_origin()
def computerOcr():
    if request.method == 'POST':
        computerDivider = ComputerCharDivider()
        divideCharImg = computerDivider.divide(request.files['file'])
        predictions = computer_ocr_model.inference(divideCharImg)
        return jsonify(predictions)


@app.route('/handWriting_ocr', methods=['POST'])
@cross_origin()
def handWritingOcr():
    if request.method == 'POST':
        handWritingDivider = HandeWritingCharDivider()
        divideCharImg = handWritingDivider.cut_image(request.files['file'])
        predictions = handWriting_ocr_model.inference(divideCharImg)
        return jsonify(predictions)


@app.route('/judge_content', methods=['POST'])
@cross_origin()
def classify_text():
    if request.method == 'POST':
        text = request.json['text']
        _, personal_data = build_single_text(classify_config, False, text)
        personal_iter = build_iterator(personal_data, classify_config)
        predictions = classify_model.predict_results(personal_iter)
        response = []
        for i in range(len(predictions)):
            response.append({'name': predictions[i][0], 'value': str(predictions[i][1])})
        return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
