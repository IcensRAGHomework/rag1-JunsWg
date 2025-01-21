import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
#from langchain.chains import RunnableSequence
import subprocess
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage

#from PIL import Image
import base64
from mimetypes import guess_type
from openai import AzureOpenAI

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)
calendarific_api_key = "F3k6kLzjG1WjphHwkjsrsxdg0FMIs9ss"

def getLLM():
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    return llm

prompt_template = """
你是一个非常聪明的助手，能够回答用户指定地点指定时间内的纪念日日期和纪念日名称，并以 JSON 格式返回结果。

问题：{question}

请确保你的回答是针对指定月份和年份的纪念日，并且以以下 JSON 格式返回：

{{
    "Result": [
        {{
                "date": "yyyy-mm-dd",
                "name": "纪念日名称"
        }},
        {{
                "date": "yyyy-mm-dd",
                "name": "纪念日名称"
        }}
        # 继续列出该月的所有节假日
    ]
}}

请确保回答的纪念日的时间和用户的一致，例如，用户问 "2024年台灣1月紀念日有哪些?" 时，你应该返回2024年台湾1月的所有紀念日，而不是10月的，或者非台灣地区的，或者非2024年的等等。
"""

def generate_hw01(question):
    llm = getLLM()
    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)
    response = (prompt | llm).invoke({"question": question})
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    #response = chain.invoke({"question": question})
    # 6. 提取 response 中的内容并清洗
    response_content = response['text'] if isinstance(response, dict) else response.content

    # 7. 去除 JSON 格式的代码块标记部分
    json_content = response_content.strip('```json\n').strip('```')

    # 8. 将结果解析为 JSON
    try:
        parsed_result = json.loads(json_content)
        print("返回的 JSON 结果：")
        json_string = json.dumps(parsed_result, indent=4, ensure_ascii=False)
        #print(json_string)  # 美化输出，确保中文显示正常
    except json.JSONDecodeError as e:
        print("解析 JSON 时出错:", e)
    return json_string


prompt_hw02_extract_template = """
你是一个非常聪明的助手，能够根据用户关于纪念日，节假日等的提问提取全部指定的地点以及对应的时间，如果用户提问的时间类似于问今年，下个月这类的时间，请根据当前的时间正确换算，并以 JSON 格式返回结果。

问题：{question}

请确保你的回答正确提取全部地点的月份和年份，如果没有提取到年份，默认为2025年，如果没有指定月份请用null字符串替换，地点需要按ISO 3166-1 alpha-2转换 如果没有地点请用cn替换，并且以以下 JSON 格式返回：

{{
    "Result": [
        {{
                "location": "地点名称的ISO 3166-1 alpha-2编码",
                "year": "yyyy"
                "month": "mm"
        }},
        {{
                "location": "地点名称的ISO 3166-1 alpha-2编码",
                "year": "yyyy"
                "month": "mm"
        }}
        # 继续列出所有地点地点对应的年份和月份
    ]
}}

请确保回答所有的地点和时间，使用ISO 3166-1 alpha-2编码转换地点，按人类的使用习惯转换时间，你应该根据今年是2025年换算,明年是2026年，去年是2024年，前年是2023年，如果没有年份或者换算不出来就用2025年替换，并确保它们的一致性，例如，用户问 "今年年台灣地区1月和美国2024年5月以及韩国明年的紀念日有哪些?" 时，返回tw，2025，01和us，2024，05,和kr，2026，null。
"""

def hw02_extract_time_location(question):
    llm = getLLM()
    prompt = PromptTemplate(input_variables=["question"], template=prompt_hw02_extract_template)
    response = (prompt | llm).invoke({"question": question})
    # 6. 提取 response 中的内容并清洗
    response_content = response['text'] if isinstance(response, dict) else response.content

    # 7. 去除 JSON 格式的代码块标记部分
    json_content = response_content.strip('```json\n').strip('```')

    # 8. 将结果解析为 JSON
    try:
        parsed_result = json.loads(json_content)
        #print("返回的时间地点的 JSON 结果：")
        #json_string = json.dumps(parsed_result, indent=4, ensure_ascii=False)
        #print(json_string)  # 美化输出，确保中文显示正常
    except json.JSONDecodeError as e:
        print("解析 JSON 时出错:", e)
        return None
    return parsed_result

prompt_hw02_show_template = """
你是一个非常聪明的助手，能够根据JSON 格式的内容提取里面的各个name和iso并以JSON格式返回结果。

问题：{question}

请确保你的回答正确提取JSON格式内容里面的list中全部，将name的内容翻译成中文后以以下 JSON 格式返回：

{{
    "Result": [
        {{
                "date": "iso",
                "name": "name翻译成中文"
        }},
        {{
                "date": "iso",
                "name": "name翻译成中文"
        }}
        # 继续列出所有iso和翻译成中文的name
    ]
}}
只回答JSON内容即可
"""

def generate_hw02(question):
    location_time = hw02_extract_time_location(question)
    # 获取 "Result" 列表
    result = location_time['Result']
    llm = getLLM()
    if result is not None:
        # 遍历 "Result" 列表并打印每个条目
        for entry in result:
            location = entry['location']
            year = entry['year']
            month = entry['month']
            #print(f"Location: {location}, Year: {year}, Month: {month}")
            if month is None:
                url = (
                    "https://calendarific.com/api/v2/holidays?&api_key={calendarific_api_key}&country={location}&year={year}").format(
                    calendarific_api_key=calendarific_api_key, location=location, year=year)
            else:
                url = (
                    "https://calendarific.com/api/v2/holidays?&api_key={calendarific_api_key}&country={location}&year={year}&month={month}").format(
                    calendarific_api_key=calendarific_api_key, location=location, year=year,
                    month=month)
            #print(f"url: {url}")
            curl_command = ["curl", "-G", url]

            # 执行 curl 命令并获取返回的结果
            result = subprocess.run(curl_command, capture_output=True, text=True)

            # 获取输出（内容）
            if result.returncode == 0:
                #print("Response:", result.stdout)
                prompt = PromptTemplate(input_variables=["question"], template=prompt_hw02_show_template)
                response = (prompt | llm).invoke({"question": result.stdout})
                response_content = response['text'] if isinstance(response, dict) else response.content
                #print(f"返回的时间地点的 JSON 结果：{response_content}")
                json_content = response_content.strip('```json\n').strip('```')
                #print("response_content:", response_content)
                #print("strip:", json_content)
                #try:
                #    parsed_result = json.loads(json_content)
                #except json.JSONDecodeError as e:
                #    print("解析 JSON 时出错:", e)
                #    return None
                return json_content

            else:
                print("Error:", result.stderr)
    pass

prompt_hw03_answer_template = """
你是一个非常聪明的助手，根據先前JSON 格式的節日清單，判断用户提供的節日是否在该清單中，並回應是否需要新增該節日,并以JSON格式返回结果。

问题：{question}

{{
     "Result": 
        {{
            "add": 布林值,
            "reason": "描述為什麼需要或不需要新增節日的具體說明"
        }}
        # 继续列出所有iso和翻译成中文的name
}}
add : 這是一個布林值，表示是否需要將節日新增到節日清單中。根據問題判斷該節日是否存在於清單中，如果不存在，則為 true；否則為 false。
reason : 描述為什麼需要或不需要新增節日，具體說明是否該節日已經存在於清單中，以及當前清單的內容。
"""
# 使用 ConversationBufferMemory 来管理消息历史
#memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#def get_session_history():
#    return memory.load_memory_variables({})["chat_history"]

# 定义一个类，将其用于存储消息历史
class MessageHistory:
    def __init__(self, messages=None):
        self.messages = messages or []

    def add_message(self, message):
        self.messages.append(message)

    def add_messages(self, messages):
        """批量添加消息"""
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, (AIMessage, HumanMessage)):
                    self.messages.append(message)
                else:
                    raise ValueError("All items in the list must be instances of AIMessage or HumanMessage.")
        else:
            raise ValueError("Input must be a list of messages.")

# 创建一个 MessageHistory 实例
history = MessageHistory()
def generate_hw03(question2, question3):
    llm = getLLM()


    # 创建对话链，结合了模型和内存
    processed_result_1 = generate_hw02(question2)
    #response_1 = conversation_chain.run(user_input_1)
    json_message = AIMessage(content=processed_result_1)  # 或者使用 HumanMessage 根据需要
    history.add_message(json_message)
    #memory.save_context({"input": question2}, {"output": processed_result_1})
    #print("processed_result_1:", processed_result_1)
    #print(memory.load_memory_variables({}))
    prompt = PromptTemplate(input_variables=["question"], template=prompt_hw03_answer_template)
    # 将 Prompt 模板转化为 Runnable 对象
    prompt_runnable = prompt | llm

    runnable = RunnableWithMessageHistory(
        prompt_runnable,
        get_session_history=lambda: history
    )
    response = runnable.invoke({"input": question3})
    #print("response:", response)
    response_content = response['text'] if isinstance(response, dict) else response.content
    #print("response_content:", response_content)
    # print(f"返回的时间地点的 JSON 结果：{response_content}")
    json_content = response_content.strip('```json\n').strip('```')
    return json_content

json_data = {
    "Result": {
        "score": "从图片中解析到的分数"
    }
}
json_str = json.dumps(json_data)
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def generate_hw04(question):
    return
    client =  AzureOpenAI(
        api_key=gpt_config['api_key'],
        api_version=gpt_config['api_version'],
        base_url=f"{gpt_config['api_base']}/openai/deployments/{gpt_config['deployment_name']}")

    # 加载图像
    image_path = 'baseball.png'
    data_url = local_image_to_data_url(image_path)
    #print("Data URL:", data_url)
    response = client.chat.completions.create(
        model=gpt_config['deployment_name'],
        messages=[
            {"role": "system", "content": f"你是一个非常聪明的助手，可以理解和分析图片内容。以以下 JSON 格式返回：{json_str}，仅需回答JSON部分即可"},
            {"role": "user", "content": [
                {
                    "type": "text",
                    "text": f"请根据以下图片回答问题：{question}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]}
        ],
        max_tokens=2000
    )
    content_str = response.choices[0].message.content
    return content_str


    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response
