---
title: 'vLLM+OpenWebUI部署大模型网页问答简易指南'
date: 2025-03-15
permalink: /posts/2025/03/vLLMOpenWebUI/
tags:
  - 本地部署
  - LLM
---

## 下载权重文件

> 来源：[如何快速下载huggingface模型——全方法总结](https://zhuanlan.zhihu.com/p/663712983)

由于众所周知的原因，我们无法直接连接Hugging face下载。因此，除了使用梯子以外，更推荐使用镜像来进行下载，又快又好：
```bash
export HF_ENDPOINT='https://hf-mirror.com'
```
然后我们下载Hugging face官方的下载工具链`huggingface-cli`：
```bash
pip install -U huggingface_hub
```
接下来开始将模型权重下载到指定路径：
```bash
cd /data/share
huggingface-cli download Qwen/QwQ-32B --local-dir QwQ-32B
```

---
## 部署vLLM

> 来源：[vLLM官方文档](https://docs.vllm.ai/en/stable/index.html)

最好新建一个全新的`conda`环境以避免可能的错误：
```bash
conda create -n vllm python=3.12 -y
conda activate vllm
```

官方推荐使用`uv`进行环境管理，可以参考[这里](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/index.html)。但在此处，我们还是使用传统的`conda`+`pip`来管理环境：
```bash
pip install vllm
```

接下来直接启动vllm：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /data/share/QwQ-32B \
  --enable-reasoning --reasoning-parser deepseek_r1 \
  --served-model-name QwQ-32B \
  --tool-call-parser hermes \
  --max-model-len=68864 \
  --tensor-parallel-size 4 \
  --port 8000
```
参数解读：
- `enable-reasoning`和`--reasoning-parser`表示将QwQ的输出格式化为`deepseek_r1`的形式，这在后续使用Open WebUI处理思考过程时将变得有用。
- `served-model-name`是调用该API时使用的模型名称。
- `tensor-parallel-size`指定模型参数并行在单节点`4`卡GPU上。
- `tool-call-parser`用于指定如何解析和处理工具调用。它定义了：
	- 如何解析工具调用的输入：将模型的输出解析为工具调用的结构化数据。
	- 如何处理工具调用的结果：将工具调用的结果返回给模型，以便模型继续生成后续内容。
	- `hermes` 表示使用Hermes解析器来处理工具调用。
- `max-model-len`用于指定上下文长度，受`KVCache`影响，上下文长度有最大值，超出则无法正常启动服务。
- `port`用于指定服务启动在哪个端口上。
---

## 部署OpenWeb UI

> 来源：[OpenWeb UI官方GitHub](https://github.com/open-webui/open-webui)

### 部署并运行Open WebUI本体
这一步非常简单，只需要配置好环境并一键运行即可：
```bash
conda activate vllm
pip install open-webui
open-webui serve
```

`open-webui`默认开启在`8080`端口上。

### 配置Open WebUI
在开始之前确保`vLLM`服务已开启在`8000`端口上：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /data/share/QwQ-32B \
  --enable-reasoning --reasoning-parser deepseek_r1 \
  --served-model-name QwQ-32B \
  --tool-call-parser hermes \
  --max-model-len=68864 \
  --tensor-parallel-size 4 \
  --port 8000
```

然后访问`http://YourLocalHostIP:8080/`，在初次开启服务时需要新建管理员账户，登录名为邮箱，名称为显示在控制台上的名字，邮箱只用做登录名无须验证。

接下来，点击左下角头像，进入`管理员面板`，找到`设置`。

- 点击`外部链接`
- 点击`管理OpenAI API连接`旁边的`+`加号

按照图示操作完成API端口的配置。
![[Pasted image 20250308142936.png]]

保存后你将可以看到在`模型`选项卡出现`QwQ-32B`的选项。
![[Pasted image 20250308143353.png]]

启用该模型后，已经可以在对话中调用该模型了。但是，你会发现模型的思考过程没有很优雅的展示出来，而且无法显示输入输出的token数量和输出速度，所以需要对`函数`选项卡进行配置。

### 配置函数
转到`函数`选项卡，并新建一个函数，填入函数名称、ID和描述（注意不要包含任何特殊字符，否则会报错）：
```python
"""
title: qwq-32b-preview Output Beautifier
author: nitefood
author_url: https://github.com/nitefood
version: 0.1
"""

from pydantic import BaseModel
from typing import Optional
import re


class Filter:

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        messages = body.get("messages", [])
        thought_pattern = re.compile(r"^(.*?)\n\n\*\*Final Answer\*\*\n\n", re.DOTALL)
        output_pattern = re.compile(r"\n\n\*\*Final Answer\*\*\n\n(.*?)$", re.DOTALL)

        for message in messages:
            content = message.get("content", "")
            if "\n\n**Final Answer**\n\n" in content:
                thought_match = thought_pattern.search(content)
                thought = thought_match.group(1).strip()
                # Format the Thought block with a blockquote in Markdown
                thought = thought.replace("\n", "\n> ")
                output_match = output_pattern.search(content)
                output = output_match.group(1).strip()
                # Format the model's response with the revised Thought block, an "Output" header, and the Final Answer
                content = f"\n\n#### Thought:\n> {thought}\n\n#### Output:\n{output}\n"
                message["content"] = content

        body["messages"] = messages
        return body
```

上述函数能让模型的思考过程优雅的显示出来，下面的函数则可以计算输入输出的token数量和输出速度（可选），新建一个新的函数：
```python
"""
title: Time Token Tracker
author: owndev
author_url: https://github.com/owndev
project_url: https://github.com/owndev/Open-WebUI-Functions
funding_url: https://github.com/owndev/Open-WebUI-Functions
version: 2.3.0
license: MIT
description: A filter for tracking the response time and token usage of a request.
features:
  - Tracks the response time of a request.
  - Tracks Token Usage.
  - Calculates the average tokens per message.
  - Calculates the tokens per second.
"""

import time
from typing import Optional
import tiktoken
from pydantic import BaseModel, Field

# Global variables to track start time and token counts
global start_time, request_token_count, response_token_count


class Filter:
    class Valves(BaseModel):
        priority: int = Field(
            default=0, description="Priority level for the filter operations."
        )
        CALCULATE_ALL_MESSAGES: bool = Field(
            default=True,
            description="If true, calculate tokens for all messages. If false, only use the last user and assistant messages.",
        )
        SHOW_AVERAGE_TOKENS: bool = Field(
            default=True,
            description="Show average tokens per message (only used if CALCULATE_ALL_MESSAGES is true).",
        )
        SHOW_RESPONSE_TIME: bool = Field(
            default=True, description="Show the response time."
        )
        SHOW_TOKEN_COUNT: bool = Field(
            default=True, description="Show the token count."
        )
        SHOW_TOKENS_PER_SECOND: bool = Field(
            default=True, description="Show tokens per second for the response."
        )

    def __init__(self):
        self.name = "Time Token Tracker"
        self.valves = self.Valves()

    async def inlet(
        self, body: dict, __user__: Optional[dict] = None, __event_emitter__=None
    ) -> dict:
        global start_time, request_token_count
        start_time = time.time()

        model = body.get("model", "default-model")
        all_messages = body.get("messages", [])

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        # If CALCULATE_ALL_MESSAGES is true, use all "user" and "system" messages
        if self.valves.CALCULATE_ALL_MESSAGES:
            request_messages = [
                m for m in all_messages if m.get("role") in ("user", "system")
            ]
        else:
            # If CALCULATE_ALL_MESSAGES is false and there are exactly two messages
            # (one user and one system), sum them both.
            request_user_system = [
                m for m in all_messages if m.get("role") in ("user", "system")
            ]
            if len(request_user_system) == 2:
                request_messages = request_user_system
            else:
                # Otherwise, take only the last "user" or "system" message if any
                reversed_messages = list(reversed(all_messages))
                last_user_system = next(
                    (
                        m
                        for m in reversed_messages
                        if m.get("role") in ("user", "system")
                    ),
                    None,
                )
                request_messages = [last_user_system] if last_user_system else []

        request_token_count = sum(
            len(encoding.encode(m["content"])) for m in request_messages
        )

        return body

    async def outlet(
        self, body: dict, __user__: Optional[dict] = None, __event_emitter__=None
    ) -> dict:
        global start_time, request_token_count, response_token_count
        end_time = time.time()
        response_time = end_time - start_time

        model = body.get("model", "default-model")
        all_messages = body.get("messages", [])

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        reversed_messages = list(reversed(all_messages))

        # If CALCULATE_ALL_MESSAGES is true, use all "assistant" messages
        if self.valves.CALCULATE_ALL_MESSAGES:
            assistant_messages = [
                m for m in all_messages if m.get("role") == "assistant"
            ]
        else:
            # Take only the last "assistant" message if any
            last_assistant = next(
                (m for m in reversed_messages if m.get("role") == "assistant"), None
            )
            assistant_messages = [last_assistant] if last_assistant else []

        response_token_count = sum(
            len(encoding.encode(m["content"])) for m in assistant_messages
        )

        # Calculate tokens per second (only for the last assistant response)
        if self.valves.SHOW_TOKENS_PER_SECOND:
            last_assistant_msg = next(
                (m for m in reversed_messages if m.get("role") == "assistant"), None
            )
            last_assistant_tokens = (
                len(encoding.encode(last_assistant_msg["content"]))
                if last_assistant_msg
                else 0
            )
            resp_tokens_per_sec = (
                0 if response_time == 0 else last_assistant_tokens / response_time
            )

        # Calculate averages only if CALCULATE_ALL_MESSAGES is true
        avg_request_tokens = avg_response_tokens = 0
        if self.valves.SHOW_AVERAGE_TOKENS and self.valves.CALCULATE_ALL_MESSAGES:
            req_count = len(
                [m for m in all_messages if m.get("role") in ("user", "system")]
            )
            resp_count = len([m for m in all_messages if m.get("role") == "assistant"])
            avg_request_tokens = request_token_count / req_count if req_count else 0
            avg_response_tokens = response_token_count / resp_count if resp_count else 0

        # Shorter style, e.g.: "10.90s | Req: 175 (Ø 87.50) | Resp: 439 (Ø 219.50) | 40.18 T/s"
        description_parts = []
        if self.valves.SHOW_RESPONSE_TIME:
            description_parts.append(f"{response_time:.2f}s")
        if self.valves.SHOW_TOKEN_COUNT:
            if self.valves.SHOW_AVERAGE_TOKENS and self.valves.CALCULATE_ALL_MESSAGES:
                # Add averages (Ø) into short output
                short_str = (
                    f"输入: {request_token_count} (Ø {avg_request_tokens:.2f}) | "
                    f"输出: {response_token_count} (Ø {avg_response_tokens:.2f})"
                )
            else:
                short_str = (
                    f"输入: {request_token_count} | 输出: {response_token_count}"
                )
            description_parts.append(short_str)
        if self.valves.SHOW_TOKENS_PER_SECOND:
            description_parts.append(f"{resp_tokens_per_sec:.2f} T/s")
        description = " | ".join(description_parts)

        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": description, "done": True},
            }
        )
        return body

```

保存这两个函数并启用他们，并且在函数设置中启用`全局`。
现在转到`设置`选项卡，并进入`模型`设置，点击`QwQ-32B`模型并启用刚刚设置好的函数。
![[Pasted image 20250308144143.png]]

（模型的`icon`需要自己设置，直接从HF的网页上扒下来就可以了🐸）

为了让除管理员以外的用户访问到，还需要设置用户组权限。
![[Pasted image 20250308144313.png]]

到这里，所需要的设置已经全部搞定！