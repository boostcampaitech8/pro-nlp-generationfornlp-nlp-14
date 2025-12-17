"""프롬프트 및 템플릿 정의 모듈"""

# Gemma 모델용 Chat Template
# base code 기반 작성인데 가독성이 구려서 수정 필요
CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set system_message = messages[0]['content'] %}"
    "{% endif %}"
    "{% if system_message is defined %}"
    "{{ system_message }}"
    "{% endif %}"
    "{% for message in messages %}"
    "{% set content = message['content'] %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<start_of_turn>user\\n' + content + '<end_of_turn>\\n<start_of_turn>model\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ content + '<end_of_turn>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
)
