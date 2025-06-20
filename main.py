import os
import pandas as pd
import requests
import json
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# It's recommended to set the API key in an environment variable for security
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

SCENE_TAGS = [
    {
        "层级1": "账户与身份管理",
        "层级2": [
            {"登录与注册": ["登录", "注册", ...]},
            {"账号绑定与解绑": ["绑定手机号", ...]},
            ...
        ]
    },
    ...
]

def call_deepseek_api(prompt):
    """
    A general function to call the DeepSeek API.
    """
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "YOUR_DEEPSEEK_API_KEY_HERE":
        print("Error: DEEPSEEK_API_KEY is not set. Please create a .env file and add your API key.")
        return None

    # Headers for the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }

    # Payload for the request
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are an expert in analyzing privacy policies and extracting personal information entities."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1, # Lower temperature for more deterministic results
    }

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        api_response = response.json()
        if "choices" in api_response and len(api_response["choices"]) > 0:
            content = api_response["choices"][0]["message"]["content"]
            return content.strip()
        else:
            print("API response does not contain expected data.")
            print("Full response:", api_response)
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the API call: {e}")
        return None

def extract_personal_data(text):
    """
    优化后的实体抽取，要求大模型输出严格的JSON数组，并做后处理。
    """
    print("Step 1: Extracting personal data...")
    prompt = f'''
    请从下列文本中全面、细致地抽取所有出现的"个人信息相关数据项"，包括但不限于：个人基本资料、身份信息、联系方式、设备信息、网络身份、健康信息、财产信息、教育信息、工作信息、上网记录、位置信息、标签信息、运动信息、第三方关联信息、其他敏感信息等。

    - 只输出数据项本身，格式为严格的 JSON 数组。例如：["姓名","手机号码","设备型号","头像","昵称"]。
    - 不要输出任何解释、注释或多余内容。
    - 数据项应尽量细化、全面，不要遗漏任何与个人信息相关的实体。
    - 如有同义项（如"手机号""手机号码"），请统一为标准表达（如"手机号码"）。
    - 只输出严格闭合的 JSON，不要输出任何解释、注释、错误提示、代码块标记或多余内容。
    - 输出必须是合法的 JSON 格式，不能缺少任何括号或逗号。

    文本："{text}"
    '''
    response_content = call_deepseek_api(prompt)
    if response_content:
        try:
            json_str = extract_first_json(response_content)
            if json_str:
                items = json.loads(json_str)
                if isinstance(items, list):
                    # 去重、去空
                    return list({i.strip() for i in items if isinstance(i, str) and i.strip()})
        except Exception as e:
            print("实体抽取后处理异常：", e)
    return []

def classify_data_items(data_items):
    """
    Classifies the extracted data items using DeepSeek API.
    """
    print("Step 2: Classifying data items...")
    if not data_items:
        return {}

    prompt = f"""
    Based on the detailed classification schema provided below, classify the following personal data items.
    For each item, provide its "一级类别" (Level 1 Category) and "二级类别" (Level 2 Category).
    Return the result as a single, valid JSON object where keys are the data items. Do not include any other text or explanation outside the JSON object.

    ### Data Items to Classify
    {json.dumps(data_items, ensure_ascii=False)}

    ### Classification Schema

    **一级类别: 个人基本资料**
    - 二级类别: 个人基本资料
    - 示例: 个人姓名、生日、年龄、性别、民族、国籍、籍贯、政治面貌、婚姻状况、家庭关系、住址、个人电话号码、电子邮件地址、兴趣爱好

    **一级类别: 个人身份信息**
    - 二级类别: 个人身份信息
    - 示例: 身份证、军官证、护照、驾驶证、工作证、社保卡、居住证、港澳台通行证等证件号码、证件照片或影印件

    **一级类别: 个人生物识别信息**
    - 二级类别: 生物识别信息
    - 示例: 个人面部识别特征、虹膜、指纹、基因、声纹、步态、耳廓、眼纹

    **一级类别: 网络身份标识信息**
    - 二级类别: 网络身份标识信息
    - 示例: 用户账号、用户ID、即时通信账号、网络社交用户账号、用户头像、昵称、个性签名、IP地址

    **一级类别: 个人健康生理信息**
    - 二级类别: 健康状况信息 (示例: 体重、身高、体温、肺活量、血压、血型)
    - 二级类别: 医疗健康信息 (示例: 医疗就诊记录、生育信息、既往病史)

    **一级类别: 个人教育信息**
    - 二级类别: 个人教育信息
    - 示例: 学历、学位、教育经历、学号、成绩单、资质证书、培训记录、奖惩信息

    **一级类别: 个人工作信息**
    - 二级类别: 个人工作信息
    - 示例: 个人职业、职位、职称、工作单位、工作地点、工作经历、工资、简历

    **一级类别: 个人财产信息**
    - 二级类别: 金融账户信息 (示例: 银行、证券等账户的账号、密码)
    - 二级类别: 个人交易信息 (示例: 交易订单、交易金额、支付记录、账单)
    - 二级类别: 个人资产信息 (示例: 个人收入状况、房产信息、存款信息、车辆信息、虚拟财产)
    - 二级类别: 个人借贷信息 (示例: 借款信息、还款信息、信贷记录、征信信息)

    **一级类别: 身份鉴别信息**
    - 二级类别: 身份鉴别信息
    - 示例: 账号口令、数字证书、短信验证码、密码提示问题

    **一级类别: 个人通信信息**
    - 二级类别: 个人通信信息
    - 示例: 通信记录，短信、彩信、话音、电子邮件、即时通信等通信内容

    **一级类别: 联系人信息**
    - 二级类别: 联系人信息
    - 示例: 通讯录、好友列表、群列表、电子邮件地址列表、家庭关系

    **一级类别: 个人上网记录**
    - 二级类别: 个人操作记录 (示例: 网页浏览记录、软件使用记录、点击记录、收藏列表、搜索记录)
    - 二级类别: UGC内容数据 (示例: 发布的图文/视频、弹幕内容、直播画面、上传的文件)
    - 二级类别: 业务行为数据 (示例: 游戏登录时间、视频观看记录、文章停留时长、音乐播放列表)

    **一级类别: 个人设备信息**
    - 二级类别: 可变更的唯一设备识别码 (示例: AndroidID, IDFA, OAID)
    - 二级类别: 不可变更的唯一设备识别码 (示例: IMEI, MEID, MAC地址, 硬件序列号)
    - 二级类别: 应用软件列表 (示例: 安装的应用程序列表)
    - 二级类别: 设备参数 (示例: 设备型号、品牌、操作系统版本、屏幕分辨率, CPU型号, 内存大小)
    - 二级类别: 技术运维数据 (示例: 崩溃日志、错误日志、性能数据)
    - 二级类别: 设备状态数据 (示例: 网络信号强度、电池温度、CPU使用率)
    - 二级类别: 网络状态信息 (示例: Wi-Fi状态、网络环境、IP地址)

    **一级类别: 个人位置信息**
    - 二级类别: 粗略位置信息 (示例: 地区代码、城市代码)
    - 二级类别: 行踪轨迹信息
    - 二级类别: 住宿出行信息 (示例: 个人住宿信息, 乘坐交通工具信息)

    **一级类别: 个人标签信息**
    - 二级类别: 个人标签信息
    - 示例: 基于个人上网记录等加工产生的个人用户标签、画像信息, 行为习惯, 兴趣偏好

    **一级类别: 个人运动信息**
    - 二级类别: 个人运动信息
    - 示例: 步数、步频、运动时长、运动距离、运动方式、运动心率

    **一级类别: 第三方关联信息**
    - 二级类别: 第三方关联信息
    - 示例: 微信好友关系、QQ群公告、企业域名、第三方账号绑定状态

    **一级类别: 其他个人信息**
    - 二级类别: 其他个人信息
    - 示例: 性取向、婚史、宗教信仰、未公开的违法犯罪记录

    ### Example Output Format
    {{
      "姓名": {{ "一级类别": "个人基本资料", "二级类别": "个人基本资料" }}
    }}
    """
    
    response_content = call_deepseek_api(prompt)
    if response_content:
        try:
            json_str = extract_first_json(response_content)
            if json_str:
                classified_data = robust_json_loads(json_str)
                if isinstance(classified_data, dict):
                    return classified_data
        except Exception as e:
            print("分类后处理异常：", e)
            print("原始模型输出：\n", response_content)
            # 可选：保存到文件
            with open("model_output_error.txt", "w", encoding="utf-8") as f:
                f.write(response_content)
            raise
    return {}

def extract_first_json(text):
    """
    提取第一个完整的 JSON 对象或数组（支持多行和嵌套），丢弃其余内容。
    """
    # 去除 markdown 代码块标记
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip("` \n")
    # 寻找第一个 { 或 [
    start = min([i for i in [text.find("{"), text.find("[")] if i != -1] or [-1])
    if start == -1:
        return None
    stack = []
    for i, c in enumerate(text[start:], start=start):
        if c in "{[":
            stack.append(c)
        elif c in "]}":
            if not stack:
                break
            last = stack.pop()
            if (last == "{" and c != "}") or (last == "[" and c != "]"):
                break
            if not stack:
                # 匹配到最外层闭合
                return text[start:i+1]
    return None

def robust_json_loads(json_str):
    json_str = extract_first_json(json_str)  # 先提取第一个合法JSON
    try:
        return json.loads(json_str)
    except Exception as e:
        print("JSON解析异常：", e)
        print("出错内容片段：\n", json_str[:500])  # 打印前500字符
        with open("json_parse_error.txt", "w", encoding="utf-8") as f:
            f.write(json_str)
        raise

def recognize_scenarios_and_build_relations(text, data_items_classified):
    print("Step 3: Recognizing scenarios and building relations...")
    data_items_str = json.dumps(data_items_classified, ensure_ascii=False, indent=2)
    prompt = f"""
你是一名隐私政策场景分析专家。请根据下方的三层级场景标签体系，结合隐私政策文本，识别每个数据项涉及的具体场景。

【任务要求】
- 结合三层及场景标签体系和本段文本内容，判断该文本涉及的最合适的场景（优先匹配到层级3或者2，无法匹配时可用层级1）。
- 输出格式为：[[层级1, 层级2, 层级3]]，如无层级3则为[[层级1, 层级2]]。
- 如该段落涉及多个场景，可全部列出。
- 只输出JSON数组，不要有多余解释。

【待分析文本】
{text}

【已分类数据项】
{data_items_str}

【三层级场景标签体系】
# 层级1: 账户与身份管理
- 层级2: 登录与注册
  - 层级3: 登录, 注册, 注册账号, 登录账号, 或使用电子邮箱登录, 通过手机号码登录, 使用第三方账号登录
- 层级2: 账号绑定与解绑
  - 层级3: 绑定手机号, 绑定微信账号, 绑定支付宝账号, 绑定第三方账户, 解绑第三方账户登录方式, 解绑已注册手机号码
- 层级2: 账号安全与维护
  - 层级3: 修改密码, 更换手机号, 修改账号信息, 冻结账号, 解冻账号, 注销账号
- 层级2: 身份认证
  - 层级3: 实名认证, 身份认证, 人脸识别验证, 申请认证, 学生身份认证

# 层级1: 内容交互与发布
- 层级2: 内容创建与发布
  - 层级3: 发布动态, 发布视频, 发布图文, 发布评论, 发布直播内容, 编辑内容, 上传内容
- 层级2: 内容消费与浏览
  - 层级3: 浏览内容, 观看视频, 观看直播, 阅读小说, 浏览商品, 查看信息
- 层级2: 内容分享与传播
  - 层级3: 分享内容至第三方平台, 分享视频, 分享图片, 分享位置, 转发内容
- 层级2: 互动与反馈
  - 层级3: 点赞内容, 评论, 回复评论, 举报, 点踩内容, 反馈意见

# 层级1: 交易与商务处理
- 层级2: 购买与支付
  - 层级3: 购买商品, 购买服务, 付费购买, 支付订单, 使用支付功能, 充值得物币, 购买虚拟商品
- 层级2: 订单管理
  - 层级3: 下单, 结算订单, 管理订单, 变更订单信息, 确认收货, 订单
- 层级2: 预订与预约
  - 层级3: 预订酒店, 预订机票, 预订门票, 预约直播, 办理入住手续, 申请出行用车
- 层级2: 售后与维权
  - 层级3: 申请售后, 申请退款, 处理投诉, 申请维权, 产生退款

# 层级1: 位置与地理服务
- 层级2: 位置共享与记录
  - 层级3: 开启位置权限, 记录位置信息, 设置通勤, 使用基于位置的服务, 启用地理位置信息
- 层级2: 附近搜索与交互
  - 层级3: 查看附近的人, 浏览附近直播, 查找司机, 搜索附近门店, 推荐附近职位
- 层级2: 导航与出行
  - 层级3: 使用导航功能, 查询出行路径, 叫车, 使用出行服务, 发起网络约车

# 层级1: 设备功能与权限控制
- 层级2: 权限开启与关闭
  - 层级3: 开启麦克风权限, 开启定位权限, 开启相机权限, 关闭蓝牙功能, 拒绝授权精确地理位置权限, 授权通讯录权限
- 层级2: 设备功能使用
  - 层级3: 使用相机, 使用麦克风, 使用扫描二维码功能, 录制视频, 拍摄照片, 连接蓝牙设备
- 层级2: 系统操作与维护
  - 层级3: 更新软件, 备份文件, 恢复出厂设定, 安装应用, 卸载应用, 储存信息

# 层级1: 数据收集与处理
- 层级2: 数据收集
  - 层级3: 收集个人信息, 收集设备信息, 收集语音内容, 获取验证码, 读取设备识别码
- 层级2: 数据处理与分析
  - 层级3: 分析用户行为, 分析数据统计, 生成医美报告, 识别需求, 基于位置优化天气功能
- 层级2: 个性化与推荐
  - 层级3: 处于个性化推荐场景, 提供个性化商品, 推荐信息, 使用智能推荐服务, 获得个性化内容展示
"""
    response_content = call_deepseek_api(prompt)
    if not response_content:
        return []
    triplets = robust_json_loads(response_content)
    return triplets if triplets else []

def detect_min_level(text, max_chars=1000):
    """
    自动检测前max_chars字的最小分级层级（如1.、1.1、1.1.1等）。
    返回最常见的分级深度。
    """
    sample = text[:max_chars]
    # 匹配如 1. 1.1 1.1.1 （1） (1) 一、二、等
    patterns = [
        r'\d+(?:\.\d+)*[、.．．]?',         # 数字点分级
        r'\([一二三四五六七八九十\d]+\)',    # (1)（1）
        r'（[一二三四五六七八九十\d]+）',
        r'[一二三四五六七八九十]+、'
    ]
    all_matches = []
    for pat in patterns:
        all_matches += re.findall(pat, sample)
    # 统计最大点数
    max_depth = 1
    for m in all_matches:
        depth = m.count('.') + 1 if '.' in m else 1
        if depth > max_depth:
            max_depth = depth
    return max_depth

def build_level_regex(level):
    """
    根据分级深度生成正则
    """
    if level == 1:
        return r'(?:^|\n)(\d+[、.．．]|[一二三四五六七八九十]+、|（[一二三四五六七八九十\d]+）|\([一二三四五六七八九十\d]+\))'
    elif level == 2:
        return r'(?:^|\n)(\d+\.\d+[、.．．]?)'
    elif level == 3:
        return r'(?:^|\n)(\d+\.\d+\.\d+[、.．．]?)'
    else:
        # 支持更深层级
        return r'(?:^|\n)(' + r'\\.'.join([r'\\d+']*level) + r'[、.．．]?)'

def split_by_detected_level(text):
    min_level = detect_min_level(text)
    pattern = re.compile(build_level_regex(min_level))
    matches = list(pattern.finditer(text))
    if not matches:
        return [text.strip()] if text.strip() else []
    paras = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        paras.append(text[start:end].strip())
    # 检查最后一段后是否还有内容
    last_end = matches[-1].end()
    if last_end < len(text):
        tail = text[last_end:].strip()
        if tail:
            paras.append(tail)
    return [p for p in paras if p]

def get_policy_column(df):
    if 'policy' in df.columns:
        return 'policy'
    possible_columns = [col for col in df.columns if 'policy' in str(col).lower() or 'text' in str(col).lower() or 'content' in str(col).lower()]
    if possible_columns:
        print(f"Warning: 'policy' column not found. Using the first likely candidate: '{possible_columns[0]}'")
        return possible_columns[0]
    raise ValueError(f"No policy text column found. Available columns: {list(df.columns)}")

def get_scene_tags(para):
    scene_triplets = recognize_scenarios_and_build_relations(para, {})
    scene_tags = []
    for t in scene_triplets:
        if isinstance(t, list) and len(t) >= 2:
            scene_tags.append(t)
    return scene_tags

def extract_entities(para):
    response = extract_personal_data(para)
    return response if response else []

def classify_entities(entities):
    if not entities:
        return {}
    for _ in range(2):  # 最多尝试2次
        response = classify_data_items(entities)
        if response and isinstance(response, dict) and all(ent in response for ent in entities):
            return response
    # 兜底：为未分类实体补充"未分类"
    return {ent: {"一级类别": "未分类", "二级类别": "未分类"} for ent in entities}

def build_entity_list(entities, classified):
    return [
        {
            "数据项": ent,
            "一级类别": classified.get(ent, {}).get("一级类别", ""),
            "二级类别": classified.get(ent, {}).get("二级类别", "")
        } for ent in entities
    ]

def build_relations(scene_tags, entities):
    rels = []
    for scene in scene_tags:
        for ent in entities:
            rels.append({
                "场景": scene,
                "关系": "收集",  # 可根据实际需求调整
                "数据项": ent
            })
    return rels

def analyze_actions(text, scene_tags, entities):
    """
    输入：原文、场景标签列表、实体列表
    输出：每个(场景, 实体)的动作，格式为 [层级1, 层级2, 层级3, 动作, 数据项]
    """
    if not scene_tags or not entities:
        return []
    prompt = f'''
你是一名隐私政策分析专家。请根据下方的场景标签和实体列表，结合原文内容，判断每个"场景-实体"对的真实动作（如"收集""使用""分析""存储""共享""披露""删除""传输""展示""公开"等），如无法判断请输出"未识别"。
输出格式为严格的 JSON 数组，每个元素为 [层级1, 层级2, 层级3, 动作, 数据项]。

【原文】
{text}

【场景标签】
{json.dumps(scene_tags, ensure_ascii=False)}

【实体列表】
{json.dumps(entities, ensure_ascii=False)}

【常见动作词表】
["收集", "使用", "分析", "存储", "共享", "披露", "删除", "传输", "展示", "公开"]
只输出严格闭合的 JSON，不要输出任何解释、注释、错误提示等。
'''
    response = call_deepseek_api(prompt)
    if response:
        try:
            json_str = extract_first_json(response)
            return robust_json_loads(json_str) if json_str else []
        except Exception as e:
            print("动作分析后处理异常：", e)
            print("原始模型输出：\n", response)
            return []
    return []

def flatten_results(policy_name, processed_content):
    flat = []
    for para_info in processed_content:
        for rel in para_info["关系标注"]:
            scene = rel["场景"]
            entity = rel["数据项"]
            entity_cat = next((e for e in para_info["实体"] if e["数据项"] == entity), {})
            flat.append({
                "隐私政策名称": policy_name,
                "段号": para_info["段号"],
                "数据项": entity,
                "一级类别": entity_cat.get("一级类别", ""),
                "二级类别": entity_cat.get("二级类别", ""),
                "使用场景层级一": scene[0] if len(scene) > 0 else "",
                "使用场景层级二": scene[1] if len(scene) > 1 else "",
                "使用场景层级三": scene[2] if len(scene) > 2 else "",
                "动作": rel["关系"]
            })
    return flat

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def process_policies(file_path):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return

    policy_column = get_policy_column(df)
    policy_name_col = 'name' if 'name' in df.columns else None

    results = []
    final_results = []
    for index, row in df.iterrows():
        policy_text = row[policy_column]
        if not isinstance(policy_text, str) or not policy_text.strip():
            continue
        policy_name = row[policy_name_col] if policy_name_col else f"政策_{index+1}"
        paragraphs = split_by_detected_level(policy_text)
        processed_content = []
        for i, para in enumerate(paragraphs):
            try:
                scene_tags = get_scene_tags(para)
                entities = extract_entities(para)
                classified = classify_entities(entities)
                entity_list = build_entity_list(entities, classified)
                relations_raw = analyze_actions(para, scene_tags, entities)
                # 转换为 dict 结构，兼容 flatten_results
                relations = []
                for rel in relations_raw:
                    if isinstance(rel, list) and len(rel) >= 5:
                        relations.append({
                            "场景": rel[:3],
                            "关系": rel[3],
                            "数据项": rel[4]
                        })
                processed_content.append({
                    "段落": para,
                    "段号": i+1,
                    "场景标签": scene_tags,
                    "实体": entity_list,
                    "关系标注": relations
                })
            except Exception as e:
                print(f"处理段落{i+1}异常：{e}")
        results.append({
            "隐私政策名称": policy_name,
            "处理后内容": processed_content
        })
        final_results.extend(flatten_results(policy_name, processed_content))
    save_json('entity extraction/processed_results.json', results)
    save_json('entity extraction/final_structured_results.json', final_results)

def build_scene_tags_prompt():
    # 自动拼接结构化体系为 prompt 文本
    ...

if __name__ == "__main__":
    # The input file is assumed to be in the entity extraction directory.
    input_file_path = 'entity extraction/privacy policy name and content demo.xlsx'
    process_policies(input_file_path) 