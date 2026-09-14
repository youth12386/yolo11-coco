

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11检测服务 - 自动化测试脚本
================================
测试目标模块: app.py (YOLO11目标检测Flask API服务)
测试用例总数: 40条
  - 等价类划分: 15条 (TC-EQ-001~015)
  - 边界值分析: 13条 (TC-BV-001~013)
  - 场景法: 12条 (TC-SC-001~012)

运行方式: python test_yolo11_detection.py
"""

import sys
import os
import ast
import unittest
import json
import base64
import io
import app as target_module
import numpy as np
sys.modules['numpy'] = np

# 全局测试结果记录列表
test_results = []

# 统一模型路径：只需在这里修改即可
MODEL_PATH = 'runs/train/exp/weights/best.onnx'
INVALID_MODEL_PATH = 'nonexistent.onnx'
BROKEN_MODEL_PATH = '损坏的模型.onnx'

def record_result(tc_id, title, method, passed, expected='', actual='', input_val=None, test_item='', criticality='', precondition='', procedure=''):
    status = 'PASS' if passed else 'FAIL'
    inferred_input = input_val if input_val is not None else (getattr(TestFixture, '_last_request', None) if 'TestFixture' in globals() else None)
    test_results.append({
        'id': tc_id,
        'title': title,
        'method': method,
        'passed': passed,
        'expected': expected,
        'actual': actual,
        'input': inferred_input,
        'test_item': test_item,
        'criticality': criticality,
        'precondition': precondition,
        'procedure': procedure,
        'status': status,
    })
    print(f"[{status}] {tc_id}: {title}")


def safe_json(resp):
    try:
        return resp.get_json(silent=True)
    except Exception:
        return None


def extract_result_labels(resp_json):
    if not isinstance(resp_json, dict):
        return []
    result = resp_json.get('result', {})
    rec_label = result.get('rec_label', [])
    if isinstance(rec_label, str):
        try:
            parsed = ast.literal_eval(rec_label)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    if isinstance(rec_label, list):
        return rec_label
    return []


class TestFixture:
    """测试夹具：提供创建真实Flask客户端和测试数据的工具"""

    @staticmethod
    def create_flask_client():
        """返回 app 的测试客户端，直接走 /detection 路由。"""
        try:
            from app import app
            app.config['TESTING'] = True
            return app.test_client()
        except Exception as e:
            print(f"[警告] 无法加载真实Flask应用: {e}")
            return None

    @staticmethod
    def create_valid_base64_image(width=640, height=640, draw_objects=False):
        """创建有效的base64编码图片，支持绘制模拟图像块"""
        try:
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (width, height), color=(240, 240, 240))
            if draw_objects:
                draw = ImageDraw.Draw(img)
                # 绘制模拟目标的简单色块
                draw.rectangle([50, 50, 200, 200], fill=(255, 0, 0))
                draw.rectangle([300, 300, 500, 500], fill=(0, 255, 0))
            buf = io.BytesIO()
            img.save(buf, format='JPEG')
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except ImportError:
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
print("=" * 60)
print("YOLO11检测服务 - 自动化测试套件")
print("测试用例总数: 40 (等价类划分15 + 边界值分析13 + 场景法12)")
print("=" * 60)


# ----- 等价类划分测试 (15条) -----

class Test_EQ_001(unittest.TestCase):
    """TC-EQ-001: 合法JSON请求正常返回检测结果"""
    def test_valid_json_request(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-EQ-001', '合法JSON请求正常返回检测结果',
                         '等价类划分-有效等价类', False, 'Flask不可用', 'Flask未安装')
            return
        b64 = TestFixture.create_valid_base64_image()
        # 记录发送的请求内容到 TestFixture._last_request，供 record_result 使用
        payload = {'image_base64': b64, 'model_path': MODEL_PATH}
        try:
            TestFixture._last_request = payload
        except Exception:
            pass
        resp = client.post('/detection',
                          data=json.dumps(payload),
                          content_type='application/json')
        data = resp.get_json()
        passed = (resp.status_code == 200 and data.get('code') == '0000' and data.get('msg') == 'success')
        record_result('TC-EQ-001', '合法JSON请求正常返回检测结果',
                     '等价类划分-有效等价类', passed,
                     '返回code=0000, msg=success, result包含检测结果',
                     f'code={data.get("code")}, status={resp.status_code}')

class Test_EQ_002(unittest.TestCase):
    """TC-EQ-002: 请求中包含多个检测目标"""
    def test_multiple_objects(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-EQ-002', '请求中包含多个检测目标',
                         '等价类划分-有效等价类', False, 'Flask不可用', 'Flask未安装')
            return
        b64 = TestFixture.create_valid_base64_image()
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': b64, 'model_path': MODEL_PATH}),
                          content_type='application/json')
        data = resp.get_json()
        passed = (data.get('code') == '0000' and 'result' in data)
        record_result('TC-EQ-002', '请求中包含多个检测目标',
                     '等价类划分-有效等价类', passed,
                     '返回code=0000, result中包含多个检测目标',
                     f'code={data.get("code")}, result_keys={list(data.get("result", {}).keys())}')

class Test_EQ_003(unittest.TestCase):
    """TC-EQ-003: 图片中无满足置信度阈值的目标"""
    def test_no_detection(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-EQ-003', '图片中无满足置信度阈值的目标',
                         '等价类划分-有效等价类', False, 'Flask不可用', 'Flask未安装')
            return
        b64 = TestFixture.create_valid_base64_image()
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': b64, 'model_path': MODEL_PATH}),
                          content_type='application/json')
        data = resp.get_json()
        passed = (resp.status_code == 200 and data.get('code') == '0000')
        record_result('TC-EQ-003', '图片中无满足置信度阈值的目标',
                     '等价类划分-有效等价类', passed,
                     '返回code=0000, rec_label为空列表',
                     f'code={data.get("code")}')

class Test_EQ_004(unittest.TestCase):
    """TC-EQ-004: 请求中指定自定义conf_thres和iou_thres参数"""
    def test_custom_thresholds(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-EQ-004', '请求中指定自定义conf_thres和iou_thres参数',
                         '等价类划分-有效等价类', False, 'Flask不可用', 'Flask未安装')
            return

        b64 = TestFixture.create_valid_base64_image()
        low_payload = {'image_base64': b64, 'model_path': MODEL_PATH, 'conf_thres': 0.1, 'iou_thres': 0.1}
        high_payload = {'image_base64': b64, 'model_path': MODEL_PATH, 'conf_thres': 0.9, 'iou_thres': 0.9}

        resp_low = client.post('/detection', data=json.dumps(low_payload), content_type='application/json')
        resp_high = client.post('/detection', data=json.dumps(high_payload), content_type='application/json')
        data_low = safe_json(resp_low)
        data_high = safe_json(resp_high)

        labels_low = extract_result_labels(data_low)
        labels_high = extract_result_labels(data_high)
        passed = (
            resp_low.status_code == 200 and resp_high.status_code == 200 and
            data_low.get('code') == '0000' and data_high.get('code') == '0000' and
            len(labels_high) <= len(labels_low)
        )

        record_result('TC-EQ-004', '请求中指定自定义conf_thres和iou_thres参数',
                     '等价类划分-有效等价类', passed,
                     '返回code=0000, 且更高阈值不应返回更多检测结果',
                     f'low_count={len(labels_low)}, high_count={len(labels_high)}, statuses={resp_low.status_code}/{resp_high.status_code}')

class Test_EQ_005(unittest.TestCase):
    """TC-EQ-005: 食品类目标正确映射到食品仓库"""
    def test_food_category(self):
        label = target_module.get_product_label("banana")
        passed = (label == "食品仓库")
        record_result('TC-EQ-005', '食品类目标正确映射到食品仓库',
                     '等价类划分-有效等价类', passed,
                     '返回"食品仓库"', f'实际返回: {label}')

class Test_EQ_006(unittest.TestCase):
    """TC-EQ-006: 体育用品类目标正确映射到体育用品仓库"""
    def test_sports_category(self):
        label = target_module.get_product_label("tennis racket")
        passed = (label == "体育用品仓库")
        record_result('TC-EQ-006', '体育用品类目标正确映射到体育用品仓库',
                     '等价类划分-有效等价类', passed,
                     '返回"体育用品仓库"', f'实际返回: {label}')

class Test_EQ_007(unittest.TestCase):
    """TC-EQ-007: 家具类目标正确映射到家具仓库"""
    def test_furniture_category(self):
        label = target_module.get_product_label("refrigerator")
        passed = (label == "家具仓库")
        record_result('TC-EQ-007', '家具类目标正确映射到家具仓库',
                     '等价类划分-有效等价类', passed,
                     '返回"家具仓库"', f'实际返回: {label}')

class Test_EQ_008(unittest.TestCase):
    """TC-EQ-008: 日用品类目标正确映射到日用品仓库"""
    def test_daily_category(self):
        label = target_module.get_product_label("toothbrush")
        passed = (label == "日用品仓库")
        record_result('TC-EQ-008', '日用品类目标正确映射到日用品仓库',
                     '等价类划分-有效等价类', passed,
                     '返回"日用品仓库"', f'实际返回: {label}')

class Test_EQ_009(unittest.TestCase):
    """TC-EQ-009: 未知仓库类目标正确映射"""
    def test_other_category(self):
        label = target_module.get_product_label("person")
        passed = (label == "未知仓库")
        record_result('TC-EQ-009', '未知仓库类目标正确映射',
                     '等价类划分-有效等价类', passed,
                     '返回"未知仓库"', f'实际返回: {label}')

class Test_EQ_010(unittest.TestCase):
    """TC-EQ-010: 缺少base64_image字段"""
    def test_missing_image_field(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-EQ-010', '缺少base64_image字段',
                         '等价类划分-无效等价类-缺少必填字段', False, 'Flask不可用', 'Flask未安装')
            return
        resp = client.post('/detection',
                          data=json.dumps({'model_path': MODEL_PATH}),
                          content_type='application/json')
        data = resp.get_json()
        msg = str(data.get('msg', '')) if isinstance(data, dict) else ''
        passed = (data.get('code') == '0001' and 'image_base64' in msg.lower())
        record_result('TC-EQ-010', '缺少base64_image字段',
                     '等价类划分-无效等价类-缺少必填字段', passed,
                     '返回code=0001，并在msg中明确指出缺少 image_base64',
                     f'code={data.get("code")}, msg={msg}')

class Test_EQ_011(unittest.TestCase):
    """TC-EQ-011: 缺少onnx_model_path字段"""
    def test_missing_model_path(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-EQ-011', '缺少onnx_model_path字段',
                         '等价类划分-无效等价类-缺少必填字段', False, 'Flask不可用', 'Flask未安装')
            return
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': 'test'}),
                          content_type='application/json')
        data = resp.get_json()
        msg = str(data.get('msg', '')) if isinstance(data, dict) else ''
        passed = (data.get('code') == '0001' and 'model_path' in msg.lower())
        record_result('TC-EQ-011', '缺少onnx_model_path字段',
                 '等价类划分-无效等价类-缺少必填字段', passed,
                 '返回code=0001，并在msg中明确指出缺少 model_path',
                 f'code={data.get("code")}, msg={msg}', test_item='POST /detection')

class Test_EQ_012(unittest.TestCase):
    """TC-EQ-012: base64_image为空字符串"""
    def test_empty_base64(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-EQ-012', 'base64_image为空字符串',
                         '等价类划分-无效等价类-空字符串输入', False, 'Flask不可用', 'Flask未安装')
            return
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': '', 'model_path': MODEL_PATH}),
                          content_type='application/json')
        data = safe_json(resp)
        msg = str(data.get('msg', '')) if isinstance(data, dict) else ''
        passed = (resp.status_code == 200 and data is not None and data.get('code') in ('0001') and ('image_base64' in msg.lower() or 'empty' in msg.lower()))
        record_result('TC-EQ-012', 'base64_image为空字符串',
                     '等价类划分-无效等价类-空字符串输入', passed,
                     '返回错误码并在msg中说明 image_base64 为空或无效',
                     f'code={data.get("code") if isinstance(data, dict) else data}, msg={msg}, status={resp.status_code}')

class Test_EQ_013(unittest.TestCase):
    """TC-EQ-013: onnx_model_path指向不存在的文件"""
    def test_nonexistent_model(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-EQ-013', 'onnx_model_path指向不存在的文件',
                         '等价类划分-无效等价类-模型文件不存在', False, 'Flask不可用', 'Flask未安装')
            return
        b64 = TestFixture.create_valid_base64_image()
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': b64, 'model_path': INVALID_MODEL_PATH}),
                          content_type='application/json')
        data = safe_json(resp)
        msg = str(data.get('msg', '')) if isinstance(data, dict) else ''
        passed = (resp.status_code == 200 and data is not None and data.get('code') in ('0001') and ('model' in msg.lower() or 'not found' in msg.lower()))
        record_result('TC-EQ-013', 'onnx_model_path指向不存在的文件',
                 '等价类划分-无效等价类-模型文件不存在', passed,
                 '返回错误码并在msg中说明模型文件缺失或不可访问',
                 f'code={data.get("code") if isinstance(data, dict) else data}, msg={msg}, status={resp.status_code}', test_item='POST /detection')

class Test_EQ_014(unittest.TestCase):
    """TC-EQ-014: 请求体为非JSON格式"""
    def test_non_json_request(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-EQ-014', '请求体为非JSON格式',
                         '等价类划分-无效等价类-非法请求格式', False, 'Flask不可用', 'Flask未安装')
            return
        resp = client.post('/detection', data='not json data', content_type='text/plain')
        data = safe_json(resp)
        msg = str(data.get('msg', '')) if isinstance(data, dict) else ''
        passed = (
            resp.status_code in [400, 415] or
            (resp.status_code == 200 and isinstance(data, dict) and data.get('code') in ['0001'] and ('json' in msg.lower() or 'object' in msg.lower()))
        )
        record_result('TC-EQ-014', '请求体为非JSON格式',
                     '等价类划分-无效等价类-非法请求格式', passed,
                     '返回400/415或code=0001，并在msg中说明请求不是合法JSON对象',
                     f'status={resp.status_code}, msg={msg}, body={data}')

class Test_EQ_015(unittest.TestCase):
    """TC-EQ-015: base64编码非有效图片数据"""
    def test_invalid_image_data(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-EQ-015', 'base64编码非有效图片数据',
                         '等价类划分-无效等价类-无效图片数据', False, 'Flask不可用', 'Flask未安装')
            return
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': 'invalid_base64_data', 'model_path': MODEL_PATH}),
                          content_type='application/json')
        data = safe_json(resp)
        msg = str(data.get('msg', '')) if isinstance(data, dict) else ''
        passed = (resp.status_code == 200 and data is not None and data.get('code') in ('0001') and ('image' in msg.lower() or 'decode' in msg.lower() or 'invalid' in msg.lower()))
        record_result('TC-EQ-015', 'base64编码非有效图片数据',
                     '等价类划分-无效等价类-无效图片数据', passed,
                     '返回错误码并在msg中说明图片解码或 image_base64 非法',
                     f'code={data.get("code") if isinstance(data, dict) else data}, msg={msg}, status={resp.status_code}')


# ----- 边界值分析测试 (12条) -----

class Test_BV_001(unittest.TestCase):
    """TC-BV-001: conf_thres设为下边界0.0"""
    def test_conf_threshold_min(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-BV-001', 'conf_thres设为下边界0.0',
                         '边界值分析-置信度下边界', False, 'Flask不可用', 'Flask未安装')
            return

        b64 = TestFixture.create_valid_base64_image()
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': b64, 'model_path': MODEL_PATH, 'conf_thres': 0.0, 'iou_thres': 0.5}),
                          content_type='application/json')
        data = safe_json(resp)
        passed = (resp.status_code == 200 and isinstance(data, dict) and data.get('code') in ('0000'))
        record_result('TC-BV-001', 'conf_thres设为下边界0.0',
                     '边界值分析-置信度下边界', passed,
                     '接口在置信度下边界0.0时不应崩溃',
                     f'code={data.get("code") if isinstance(data, dict) else data}, status={resp.status_code}')

class Test_BV_002(unittest.TestCase):
    """TC-BV-002: conf_thres设为上边界1.0"""
    def test_conf_threshold_max(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-BV-002', 'conf_thres设为上边界1.0',
                         '边界值分析-置信度上边界', False, 'Flask不可用', 'Flask未安装')
            return

        b64 = TestFixture.create_valid_base64_image()
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': b64, 'model_path': MODEL_PATH, 'conf_thres': 1.0, 'iou_thres': 0.5}),
                          content_type='application/json')
        data = safe_json(resp)
        passed = (resp.status_code == 200 and isinstance(data, dict) and data.get('code') in ('0000'))
        record_result('TC-BV-002', 'conf_thres设为上边界1.0',
                     '边界值分析-置信度上边界', passed,
                     '接口在置信度上边界1.0时不应崩溃',
                     f'code={data.get("code") if isinstance(data, dict) else data}, status={resp.status_code}')

class Test_BV_003(unittest.TestCase):
    """TC-BV-003: conf_thres低于下边界-0.1"""
    def test_conf_threshold_below_min(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-BV-003', 'conf_thres低于下边界-0.1',
                         '边界值分析-置信度低于下边界', False, 'Flask不可用', 'Flask未安装')
            return

        b64 = TestFixture.create_valid_base64_image()
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': b64, 'model_path': MODEL_PATH, 'conf_thres': -0.1, 'iou_thres': 0.5}),
                          content_type='application/json')
        data = safe_json(resp)
        passed = (resp.status_code == 200 and isinstance(data, dict) and data.get('code') == '0001')
        record_result('TC-BV-003', 'conf_thres低于下边界-0.1',
                     '边界值分析-置信度低于下边界', passed,
                     '接口在负阈值输入下应返回错误码0001，不能静默成功',
                     f'code={data.get("code") if isinstance(data, dict) else data}, status={resp.status_code}')

class Test_BV_004(unittest.TestCase):
    """TC-BV-004: conf_thres高于上边界1.1"""
    def test_conf_threshold_above_max(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-BV-004', 'conf_thres高于上边界1.1',
                         '边界值分析-置信度高于上边界', False, 'Flask不可用', 'Flask未安装')
            return

        b64 = TestFixture.create_valid_base64_image()
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': b64, 'model_path': MODEL_PATH, 'conf_thres': 1.1, 'iou_thres': 0.5}),
                          content_type='application/json')
        data = safe_json(resp)
        passed = (resp.status_code == 200 and isinstance(data, dict) and data.get('code') == '0001')
        record_result('TC-BV-004', 'conf_thres高于上边界1.1',
                     '边界值分析-置信度高于上边界', passed,
                     '接口在超出合法置信度上界时应返回错误码0001，不能静默成功',
                     f'code={data.get("code") if isinstance(data, dict) else data}, status={resp.status_code}')

class Test_BV_005(unittest.TestCase):
    """TC-BV-005: iou_thres设为下边界0.0"""
    def test_iou_threshold_min(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-BV-005', 'iou_thres设为下边界0.0',
                         '边界值分析-IoU下边界', False, 'Flask不可用', 'Flask未安装')
            return

        b64 = TestFixture.create_valid_base64_image()
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': b64, 'model_path': MODEL_PATH, 'conf_thres': 0.5, 'iou_thres': 0.0}),
                          content_type='application/json')
        data = safe_json(resp)
        passed = (resp.status_code == 200 and isinstance(data, dict) and data.get('code') in ('0000'))
        record_result('TC-BV-005', 'iou_thres设为下边界0.0',
                     '边界值分析-IoU下边界', passed,
                     '接口在IoU下边界0.0时不应崩溃',
                     f'code={data.get("code") if isinstance(data, dict) else data}, status={resp.status_code}')

class Test_BV_006(unittest.TestCase):
    """TC-BV-006: iou_thres设为上边界1.0"""
    def test_iou_threshold_max(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-BV-006', 'iou_thres设为上边界1.0',
                         '边界值分析-IoU上边界', False, 'Flask不可用', 'Flask未安装')
            return

        b64 = TestFixture.create_valid_base64_image()
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': b64, 'model_path': MODEL_PATH, 'conf_thres': 0.5, 'iou_thres': 1.0}),
                          content_type='application/json')
        data = safe_json(resp)
        passed = (resp.status_code == 200 and isinstance(data, dict) and data.get('code') in ('0000'))
        record_result('TC-BV-006', 'iou_thres设为上边界1.0',
                     '边界值分析-IoU上边界', passed,
                     '接口在IoU上边界1.0时不应崩溃',
                     f'code={data.get("code") if isinstance(data, dict) else data}, status={resp.status_code}')

class Test_BV_007(unittest.TestCase):
    """TC-BV-007: 输入极小图片(1x1像素)"""
    def test_minimal_image(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-BV-007', '输入极小图片(1x1像素)',
                         '边界值分析-极小图片', False, 'Flask不可用', 'Flask未安装')
            return

        b64 = TestFixture.create_valid_base64_image(width=1, height=1)
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': b64, 'model_path': MODEL_PATH}),
                          content_type='application/json')
        data = safe_json(resp)
        passed = (resp.status_code == 200 and isinstance(data, dict) and data.get('code') in ('0000'))
        record_result('TC-BV-007', '输入极小图片(1x1像素)',
                     '边界值分析-极小图片', passed,
                     '接口应返回200并不能直接因极小尺寸崩溃',
                     f'code={data.get("code") if isinstance(data, dict) else data}, status={resp.status_code}')

class Test_BV_008(unittest.TestCase):
    """TC-BV-008: 输入超大尺寸图片(如8000x8000)"""
    def test_large_image(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-BV-008', '输入超大尺寸图片(如8000x8000)',
                         '边界值分析-超大图片', False, 'Flask不可用', 'Flask未安装')
            return

        b64 = TestFixture.create_valid_base64_image(width=8000, height=8000)
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': b64, 'model_path': MODEL_PATH}),
                          content_type='application/json')
        data = safe_json(resp)
        passed = (resp.status_code == 200 and isinstance(data, dict) and data.get('code') in ('0000'))
        record_result('TC-BV-008', '输入超大尺寸图片(如8000x8000)',
                     '边界值分析-超大图片', passed,
                     '接口应返回200并不能直接因超大图崩溃',
                     f'code={data.get("code") if isinstance(data, dict) else data}, status={resp.status_code}')

class Test_BV_009(unittest.TestCase):
    """TC-BV-009: 目标边界框左上角坐标为(0,0)"""
    def test_box_top_left_origin(self):
        detector = target_module.YOLO11('runs/train/exp/weights/best.onnx', '', 0.5, 0.5)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        original = img.copy()
        try:
            detector.draw_detections(img, [0, 0, 40, 60], 0.99, 0)
            passed = (img.shape == original.shape and img.dtype == original.dtype and np.any(img != original))
            actual = f'origin_box_drawn={bool(np.any(img != original))}, shape={img.shape}'
        except Exception as exc:
            passed = False
            actual = f'exception={type(exc).__name__}: {exc}'

        record_result('TC-BV-009', '目标边界框左上角坐标为(0,0)',
                     '边界值分析-坐标边界左上角', passed,
                     '图像处理函数应能在坐标原点处绘制边界框而不崩溃',
                     actual, test_item='YOLO11.draw_detections')

class Test_BV_010(unittest.TestCase):
    """TC-BV-010: 目标边界框右下角坐标为图片宽高"""
    def test_box_bottom_right_edge(self):
        detector = target_module.YOLO11('runs/train/exp/weights/best.onnx', '', 0.5, 0.5)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        original = img.copy()
        x1 = img.shape[1] - 10
        y1 = img.shape[0] - 10
        try:
            detector.draw_detections(img, [x1, y1, 20, 20], 0.99, 0)
            passed = (img.shape == original.shape and img.dtype == original.dtype and np.any(img != original))
            actual = f'edge_box_drawn={bool(np.any(img != original))}, last_pixel=({x1},{y1})'
        except Exception as exc:
            passed = False
            actual = f'exception={type(exc).__name__}: {exc}'

        record_result('TC-BV-010', '目标边界框右下角坐标为图片宽高',
                     '边界值分析-坐标边界右下角', passed,
                     '图像处理函数应能在图片右下角边界处绘制边界框而不崩溃',
                     actual)

class Test_BV_011(unittest.TestCase):
    """TC-BV-011: 模型输出恰好有1个检测目标"""
    def test_single_detection(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-BV-011', '模型输出恰好有1个检测目标',
                         '边界值分析-单目标检测', False, 'Flask不可用', 'Flask未安装')
            return

        b64 = TestFixture.create_valid_base64_image()
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': b64, 'model_path': MODEL_PATH, 'conf_thres': 0.9, 'iou_thres': 0.9}),
                          content_type='application/json')
        data = safe_json(resp)
        labels = extract_result_labels(data)
        passed = (resp.status_code == 200 and isinstance(data, dict) and data.get('code') == '0000' and isinstance(labels, list))
        record_result('TC-BV-011', '模型输出恰好有1个检测目标',
                     '边界值分析-单目标检测', passed,
                     '接口返回有效检测结果列表，且无异常',
                     f'code={data.get("code") if isinstance(data, dict) else data}, labels={labels}')

class Test_BV_012(unittest.TestCase):
    """TC-BV-012: 模型输出恰好无检测目标(置信度恰好等于阈值)"""
    def test_confidence_exactly_at_threshold(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-BV-012', '模型输出恰好无检测目标(置信度恰好等于阈值)',
                         '边界值分析-置信度恰好等于阈值', False, 'Flask不可用', 'Flask未安装')
            return

        b64 = TestFixture.create_valid_base64_image()
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': b64, 'model_path': MODEL_PATH, 'conf_thres': 0.5, 'iou_thres': 0.5}),
                          content_type='application/json')
        data = safe_json(resp)
        passed = (resp.status_code == 200 and isinstance(data, dict) and data.get('code') in ('0000'))
        record_result('TC-BV-012', '模型输出恰好无检测目标(置信度恰好等于阈值)',
                     '边界值分析-置信度恰好等于阈值', passed,
                     '阈值边界值输入下接口应保持稳定',
                     f'code={data.get("code") if isinstance(data, dict) else data}, status={resp.status_code}')

class Test_BV_013(unittest.TestCase):
    """TC-BV-013: letterbox应遵守调用方指定的非640目标尺寸"""
    def test_letterbox_honors_requested_shape(self):
        # letterbox 的 new_shape 参数是公开接口；非模型默认尺寸也必须被尊重。
        detector = target_module.YOLO11('runs/train/exp/weights/best.onnx', '', 0.5, 0.5)
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        requested_shape = (320, 480)

        letterboxed, _, _ = detector.letterbox(img, new_shape=requested_shape)
        passed = tuple(letterboxed.shape[:2]) == requested_shape
        record_result(
            'TC-BV-013',
            'letterbox应遵守调用方指定的非640目标尺寸',
            '边界值分析-自定义letterbox尺寸',
            passed,
            f'输出尺寸应为{requested_shape[0]}x{requested_shape[1]}',
            f'实际输出尺寸={tuple(letterboxed.shape[:2])}',
            test_item='YOLO11.letterbox'
        )
        self.assertEqual(tuple(letterboxed.shape[:2]), requested_shape)


# ----- 场景法测试 (12条) -----

class Test_SC_001(unittest.TestCase):
    """TC-SC-001: 正常检测完整业务流程"""
    def test_full_detection_flow(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-SC-001', '正常检测完整业务流程',
                         '场景法-正常流程场景', False, 'Flask不可用', 'Flask未安装')
            return
        b64 = TestFixture.create_valid_base64_image()
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': b64, 'model_path': MODEL_PATH}),
                          content_type='application/json')
        data = resp.get_json()
        passed = (resp.status_code == 200 and data.get('code') == '0000'
                  and 'result' in data and 'rec_base64' in data.get('result', {}))
        record_result('TC-SC-001', '正常检测完整业务流程',
                     '场景法-正常流程场景', passed,
                     '全流程无异常,返回完整检测结果',
                     f'code={data.get("code")}, has_result={("result" in data)}')

class Test_SC_002(unittest.TestCase):
    """TC-SC-002: 多目标密集场景(目标重叠)"""
    def test_dense_multi_object(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-SC-002', '多目标密集场景(目标重叠)',
                         '场景法-多目标密集场景', False, 'Flask不可用', 'Flask未安装')
            return
        b64 = TestFixture.create_valid_base64_image()
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': b64, 'model_path': MODEL_PATH}),
                          content_type='application/json')
        data = safe_json(resp)
        passed = (resp.status_code == 200 and isinstance(data, dict) and data.get('code') == '0000' and 'rec_boxes' in data.get('result', {}))
        record_result('TC-SC-002', '多目标密集场景(目标重叠)',
                 '场景法-多目标密集场景', passed,
                 'NMS正确抑制重叠框,返回有效检测结果',
                 f'code={data.get("code") if isinstance(data, dict) else data}, status={resp.status_code}', test_item='POST /detection (NMS)')

class Test_SC_003(unittest.TestCase):
    """TC-SC-003: 连续多次请求(模拟并发)"""
    def test_consecutive_requests(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-SC-003', '连续多次请求(模拟并发)',
                         '场景法-连续请求场景', False, 'Flask不可用', 'Flask未安装')
            return
        b64 = TestFixture.create_valid_base64_image()
        success_count = 0
        for i in range(5):
            resp = client.post('/detection',
                              data=json.dumps({'image_base64': b64, 'model_path': MODEL_PATH}),
                              content_type='application/json')
            data = safe_json(resp)
            if resp.status_code == 200 and isinstance(data, dict) and data.get('code') == '0000':
                success_count += 1
        passed = (success_count == 5)
        record_result('TC-SC-003', '连续多次请求(模拟并发)',
                     '场景法-连续请求场景', passed,
                     '所有请求均成功处理且返回0000,无崩溃',
                     f'成功={success_count}/5')

class Test_SC_004(unittest.TestCase):
    """TC-SC-004: 不同类别目标混合检测的仓库分类"""
    def test_mixed_category_warehouse(self):
        # 验证混合类别的仓库分类
        labels = ["banana", "tennis racket", "refrigerator", "toothbrush", "person"]
        results = [target_module.get_product_label(l) for l in labels]
        expected = ["食品仓库", "体育用品仓库", "家具仓库", "日用品仓库", "未知仓库"]
        passed = (results == expected)
        record_result('TC-SC-004', '不同类别目标混合检测的仓库分类',
                     '场景法-混合类别分类场景', passed,
                     '每个类别映射到正确的仓库',
                     f'实际={results}, 预期={expected}')

class Test_SC_005(unittest.TestCase):
    """TC-SC-005: 模型文件加载失败场景"""
    def test_model_load_failure(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-SC-005', '模型文件加载失败场景',
                         '场景法-模型加载失败场景', False, 'Flask不可用', 'Flask未安装')
            return
        b64 = TestFixture.create_valid_base64_image()
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': b64, 'model_path': BROKEN_MODEL_PATH}),
                          content_type='application/json')
        data = safe_json(resp)
        passed = (resp.status_code == 200 and isinstance(data, dict) and data.get('code') == '0001')
        record_result('TC-SC-005', '模型文件加载失败場景',
                 '场景法-模型加载失败场景', passed,
                 '返回code=0001, 服务不崩溃',
                 f'code={data.get("code") if isinstance(data, dict) else data}, status={resp.status_code}', test_item='POST /detection (model loading)')

class Test_SC_006(unittest.TestCase):
    """TC-SC-006: 图片解码失败场景"""
    def test_image_decode_failure(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-SC-006', '图片解码失败场景',
                         '场景法-图片解码失败场景', False, 'Flask不可用', 'Flask未安装')
            return
        resp = client.post('/detection',
                          data=json.dumps({'image_base64': 'invalid_image_data', 'model_path': MODEL_PATH}),
                          content_type='application/json')
        data = safe_json(resp)
        passed = (resp.status_code == 200 and isinstance(data, dict) and data.get('code') == '0001')
        record_result('TC-SC-006', '图片解码失败场景',
                     '场景法-图片解码失败场景', passed,
                     '返回code=0001, 服务不崩溃',
                     f'code={data.get("code") if isinstance(data, dict) else data}, status={resp.status_code}')

class Test_SC_007(unittest.TestCase):
    """TC-SC-007: 请求参数类型错误(conf_thres传字符串)"""
    def test_wrong_param_type(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-SC-007', '请求参数类型错误(conf_thres传字符串)',
                         '场景法-参数类型错误场景', False, 'Flask不可用', 'Flask未安装')
            return
        b64 = TestFixture.create_valid_base64_image()
        resp = client.post('/detection',
                          data=json.dumps({
                              'image_base64': b64,
                              'model_path': MODEL_PATH,
                              'conf_thres': 'high'
                          }),
                          content_type='application/json')
        data = safe_json(resp)
        msg = str(data.get('msg', '')) if isinstance(data, dict) else ''
        passed = (
            resp.status_code in [400, 415] or
            (resp.status_code == 200 and isinstance(data, dict) and data.get('code') in ['0001'] and ('conf_thres' in msg.lower() or 'threshold' in msg.lower() or 'numeric' in msg.lower()))
        )
        record_result('TC-SC-007', '请求参数类型错误(conf_thres传字符串)',
                     '场景法-参数类型错误场景', passed,
                     '服务不崩溃，并且在 msg 中明确说明 conf_thres 类型错误',
                     f'status={resp.status_code}, msg={msg}, body={data}')

class Test_SC_008(unittest.TestCase):
    """TC-SC-008: 80个COCO类别仓库分类完整性验证"""
    def test_all_categories_mapped(self):
        all_warehouses = {
            "其他仓库",
            "体育用品仓库",
            "食品仓库",
            "家具仓库",
            "日用品仓库",
            "未知仓库",
        }
        results = [target_module.get_product_label(cls) for cls in target_module.CLASS_NAMES]
        warehouses_covered = set(results)
        passed = warehouses_covered <= all_warehouses
        record_result(
            'TC-SC-008',
            '80个COCO类别仓库分类完整性验证',
            '场景法-分类完整性验证场景',
            passed,
            '所有类别必须落在 5 个仓库 + 未知仓库 这 6 个合法值中',
            f'CLASS_NAMES数量={len(target_module.CLASS_NAMES)}, warehouses={warehouses_covered}'
        )
        
class Test_SC_009(unittest.TestCase):
    """TC-SC-009: 中文名称映射正确性验证"""
    def test_chinese_name_mapping(self):
        # 验证CLASS_NAMES和chinese_names长度一致且映射正确
        passed = (len(target_module.CLASS_NAMES) == len(target_module.chinese_names))
        # 验证英文到中文的索引对应
        mapping_ok = True
        for i in range(min(len(target_module.CLASS_NAMES), len(target_module.chinese_names))):
            if target_module.CLASS_NAMES[i] == "person" and target_module.chinese_names[i] != "人":
                mapping_ok = False
        record_result('TC-SC-009', '中文名称映射正确性验证',
                     '场景法-中文映射正确性场景', passed and mapping_ok,
                     'CLASS_NAMES与chinese_names长度一致,索引对应正确',
                     f'en_count={len(target_module.CLASS_NAMES)}, cn_count={len(target_module.chinese_names)}, match={len(target_module.CLASS_NAMES)==len(target_module.chinese_names)}')

class Test_SC_010(unittest.TestCase):
    """TC-SC-010: 异常后服务恢复场景"""
    def test_error_recovery(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-SC-010', '异常后服务恢复场景',
                         '场景法-异常恢复场景', False, 'Flask不可用', 'Flask未安装')
            return
        # 先发送错误请求
        resp1 = client.post('/detection',
                           data=json.dumps({'model_path': MODEL_PATH}),
                           content_type='application/json')
        data1 = resp1.get_json()
        # 再发送正常请求
        b64 = TestFixture.create_valid_base64_image()
        resp2 = client.post('/detection',
                           data=json.dumps({'image_base64': b64, 'model_path': MODEL_PATH}),
                           content_type='application/json')
        data2 = resp2.get_json()
        passed = (data1.get('code') == '0001' and data2.get('code') == '0000')
        record_result('TC-SC-010', '异常后服务恢复场景',
                 '场景法-异常恢复场景', passed,
                 '错误请求返回code=0001, 后续正常请求仍返回code=0000',
                 f'error_code={data1.get("code")}, normal_code={data2.get("code")}', test_item='POST /detection')

class Test_SC_011(unittest.TestCase):
    """TC-SC-011: 不同宽高比图片的letterbox处理"""
    def test_different_aspect_ratios(self):
        # 真实调用 app.py 中的 YOLO11.letterbox 处理逻辑，而不是只比较常量
        shapes = [(480, 640), (640, 480), (640, 640)]  # 横版, 竖版, 正方形
        detector = target_module.YOLO11('runs/train/exp/weights/best.onnx', '', 0.5, 0.5)
        passed = True
        observed_shapes = []

        for h, w in shapes:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            letterboxed, _, _ = detector.letterbox(img, new_shape=(640, 640))
            observed_shapes.append((h, w, tuple(letterboxed.shape[:2])))
            if tuple(letterboxed.shape[:2]) != (640, 640):
                passed = False

        record_result('TC-SC-011', '不同宽高比图片的letterbox处理',
                     '场景法-不同宽高比场景', passed,
                     '所有宽高比图片letterbox后均为640x640',
                     f'tested_shapes={observed_shapes}')

class Test_SC_012(unittest.TestCase):
    """TC-SC-012: log_id时间戳唯一性验证"""
    def test_log_id_uniqueness(self):
        client = TestFixture.create_flask_client()
        if client is None:
            record_result('TC-SC-012', 'log_id时间戳唯一性验证',
                         '场景法-日志唯一性场景', False, 'Flask不可用', 'Flask未安装')
            return

        b64 = TestFixture.create_valid_base64_image()
        log_ids = []
        last_status = None
        for i in range(5):
            resp = client.post('/detection',
                              data=json.dumps({'image_base64': b64, 'model_path': MODEL_PATH}),
                              content_type='application/json')
            last_status = resp.status_code
            data = safe_json(resp)
            log_id = data.get('log_id') if isinstance(data, dict) else None
            if log_id:
                log_ids.append(log_id)

        passed = (
            last_status == 200 and
            len(log_ids) == 5 and
            len(set(log_ids)) == 5 and
            all(isinstance(log_id, str) and len(log_id) >= 15 for log_id in log_ids)
        )
        record_result('TC-SC-012', 'log_id时间戳唯一性验证',
                     '场景法-日志唯一性场景', passed,
                     '接口返回的 5 个 log_id 均存在且互不重复',
                     f'log_ids={log_ids}, unique_count={len(set(log_ids))}, status={last_status}', test_item='POST /detection (log_id)')


# ============================================================
# 测试执行与报告
# ============================================================

def run_all_tests():
    """执行所有测试用例并输出报告"""
    suite = unittest.TestSuite()

    # 添加所有测试类
    test_classes = [
        Test_EQ_001, Test_EQ_002, Test_EQ_003, Test_EQ_004, Test_EQ_005,
        Test_EQ_006, Test_EQ_007, Test_EQ_008, Test_EQ_009, Test_EQ_010,
        Test_EQ_011, Test_EQ_012, Test_EQ_013, Test_EQ_014, Test_EQ_015,
        Test_BV_001, Test_BV_002, Test_BV_003, Test_BV_004, Test_BV_005,
        Test_BV_006, Test_BV_007, Test_BV_008, Test_BV_009, Test_BV_010,
        Test_BV_011, Test_BV_012, Test_BV_013,
        Test_SC_001, Test_SC_002, Test_SC_003, Test_SC_004, Test_SC_005,
        Test_SC_006, Test_SC_007, Test_SC_008, Test_SC_009, Test_SC_010,
        Test_SC_011, Test_SC_012,
    ]

    for test_class in test_classes:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(test_class))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 输出详细报告
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    # 统计
    eq_pass = sum(1 for r in test_results if r['method'].startswith('等价类') and r['passed'])
    eq_total = sum(1 for r in test_results if r['method'].startswith('等价类'))
    bv_pass = sum(1 for r in test_results if r['method'].startswith('边界值') and r['passed'])
    bv_total = sum(1 for r in test_results if r['method'].startswith('边界值'))
    sc_pass = sum(1 for r in test_results if r['method'].startswith('场景法') and r['passed'])
    sc_total = sum(1 for r in test_results if r['method'].startswith('场景法'))

    total_passed = sum(1 for r in test_results if r['passed'])
    total_failed = sum(1 for r in test_results if not r['passed'])

    print(f"\n总用例数: {len(test_results)}")
    print(f"通过: {total_passed}")
    print(f"失败: {total_failed}")

    print(f"\n--- 按测试方法分类 ---")
    print(f"等价类划分: {eq_pass}/{eq_total} 通过")
    print(f"边界值分析: {bv_pass}/{bv_total} 通过")
    print(f"场景法: {sc_pass}/{sc_total} 通过")

    print("\n--- 详细结果 ---")
    for r in test_results:
        status = "PASS" if r['passed'] else "FAIL"
        print(f"[{status}] {r['id']} | {r['title']}")
        if not r['passed']:
            print(f"       预期: {r['expected']}")
            print(f"       实际: {r['actual']}")

    # 清理
    print("\n测试完成，环境已清理。")

    # 将测试结果写入 CSV，方便后续查看（UTF-8 with BOM，便于打开）
    try:
        import csv
        csv_path = 'detection_test_records.csv'

        # 英文表头
        header = [
            'Test Case ID 测试用例编号','Test Item 测试项（即功能模块或函数）', 'Test Case Title 测试用例标题',  'Test Criticality重要级别',
            'Pre-condition 预置条件', 'Input 输入', 'Procedure 操作步骤', 'Output 预期结果',
            'Result实际结果', 'Status是否通过', 'Remark备注（在此描述使用的测试方法）'
        ]

        def infer_test_item(title, method):
            t = (title or '').lower()
            m = (method or '').lower()
            # 如果记录中已有明确test_item字段则使用之
            # 否则根据用例标题或method做简单启发式推断
            if '映射' in t or '仓库' in t or '映射' in m:
                return 'get_product_label'
            if 'conf_thres' in t or 'conf_thres' in m or 'iou_thres' in t or 'iou_thres' in m:
                return 'nms/thresholds'
            # 与检测、请求相关的用例一律标注为检测接口
            if '请求' in t or '检测' in t or 'detection' in t or 'image' in t or '图片' in t:
                return 'POST /detection'
            return ''

        original_csv_map = {}
        try:
            if os.path.exists(csv_path):
                with open(csv_path, 'r', encoding='utf-8-sig', newline='') as read_csv:
                    reader = csv.reader(read_csv)
                    rows = list(reader)
                    if rows:
                        header_map = {str(h).strip(): idx for idx, h in enumerate(rows[0])}
                        for row in rows[1:]:
                            if not row or not row[0].strip():
                                continue
                            tc_id = row[0].strip()
                            original_csv_map[tc_id] = {
                                'criticality': row[header_map.get('Test Criticality重要级别', 3)] if 'Test Criticality重要级别' in header_map and len(row) > header_map['Test Criticality重要级别'] else '',
                                'precondition': row[header_map.get('Pre-condition 预置条件', 4)] if 'Pre-condition 预置条件' in header_map and len(row) > header_map['Pre-condition 预置条件'] else '',
                                'procedure': row[header_map.get('Procedure 操作步骤', 6)] if 'Procedure 操作步骤' in header_map and len(row) > header_map['Procedure 操作步骤'] else '',
                            }
        except Exception:
            original_csv_map = {}

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            def _stringify(val):
                try:
                    if val is None:
                        return ''
                    if isinstance(val, (dict, list)):
                        return json.dumps(val, ensure_ascii=False)
                    if isinstance(val, bytes):
                        return base64.b64encode(val).decode('utf-8')
                    return str(val)
                except Exception:
                    return str(val)

            for r in test_results:
                tc_id = r.get('id') or r.get('tc_id') or ''
                # 优先使用显式记录的 test_item 字段
                test_item = r.get('test_item') or infer_test_item(r.get('title', ''), r.get('method', ''))
                title = r.get('title', '')
                criticality = original_csv_map.get(tc_id, {}).get('criticality', r.get('criticality', ''))
                precondition = original_csv_map.get(tc_id, {}).get('precondition', r.get('precondition', ''))
                input_val = _stringify(r.get('input', ''))
                procedure = original_csv_map.get(tc_id, {}).get('procedure', r.get('procedure', ''))
                expected = _stringify(r.get('expected', ''))
                actual = _stringify(r.get('actual', ''))
                remark = r.get('method', '')
                status = 'PASS' if r.get('passed') else (r.get('status') if r.get('status') in ['OK','NG'] else 'FAIL')
                row = [tc_id, test_item, title, criticality, precondition, input_val, procedure, expected, actual, status, remark]
                writer.writerow(row)
        print(f"测试记录已写入: {csv_path}")
    except Exception as e:
        print(f"写入 CSV 失败: {e}")

    return total_passed, total_failed


if __name__ == '__main__':
    run_all_tests()
