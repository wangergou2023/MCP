import asyncio
import json
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Configuration:
    """管理配置和环境变量"""

    def __init__(self) -> None:
        """初始化，加载环境变量，拿API key"""
        self.load_env()  # 加载.env里的变量
        self.api_key = os.getenv("LLM_API_KEY")  # 读取API key

    @staticmethod
    def load_env() -> None:
        """加载.env文件到环境变量"""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> dict[str, Any]:
        """
        读取JSON格式的server配置
        file_path: 配置文件路径
        返回：配置内容dict
        """
        with open(file_path, "r") as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """
        获取API key，如果没配，报错
        """
        if not self.api_key:
            raise ValueError("LLM_API_KEY not found in environment variables")
        return self.api_key


class Server:
    """管理MCP服务端连接和工具调用"""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name                     # server名字
        self.config: dict[str, Any] = config      # server配置
        self.session: ClientSession | None = None # 会话实例
        self._cleanup_lock: asyncio.Lock = asyncio.Lock() # 异步锁，防止多线程清理冲突
        self.exit_stack: AsyncExitStack = AsyncExitStack()# 资源自动释放

    async def initialize(self) -> None:
        """初始化server连接（拉起子进程，连接mcp）"""
        # 选用npx还是指定命令
        command = (
            shutil.which("npx")
            if self.config["command"] == "npx"
            else self.config["command"]
        )
        if command is None:
            raise ValueError("命令非法，不能为空")
        # 组装启动参数
        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]}
            if self.config.get("env")
            else None,
        )
        try:
            # 启动stdio连接（自动管理）
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            # 新建会话
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()    # 完成初始化
            self.session = session
        except Exception as e:
            logging.error(f"初始化server {self.name} 失败: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """
        获取server支持的所有工具（调用协议接口）
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} 未初始化")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                tools.extend(
                    Tool(tool.name, tool.description, tool.inputSchema)
                    for tool in item[1]
                )

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """
        执行指定工具，有重试机制
        tool_name: 工具名
        arguments: 调用参数
        retries: 最大重试次数
        delay: 失败重试间隔
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} 未初始化")

        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"执行工具 {tool_name} ...")
                result = await self.session.call_tool(tool_name, arguments)
                return result
            except Exception as e:
                attempt += 1
                logging.warning(
                    f"调用失败: {e}. 第{attempt}/{retries}次."
                )
                if attempt < retries:
                    logging.info(f"{delay}s后重试...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("重试次数已用完，放弃")
                    raise

    async def cleanup(self) -> None:
        """
        释放资源（关闭子进程/会话等）
        """
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
            except Exception as e:
                logging.error(f"清理server {self.name} 资源出错: {e}")

class Tool:
    """表示一个工具对象（包含说明、参数等）"""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name                    # 工具名
        self.description: str = description      # 工具功能描述
        self.input_schema: dict[str, Any] = input_schema  # 工具参数格式说明

    def format_for_llm(self) -> str:
        """
        格式化工具信息，给大模型看的字符串
        返回：工具名+说明+参数列表
        """
        args_desc = []
        if "properties" in self.input_schema:  # 有参数说明才处理
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"  # 参数名+描述
                )
                # 如果是必填参数，加标记
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """管理和大模型（LLM）服务端的通信"""

    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key  # 保存API key

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """
        发送消息给大模型，拿回复
        messages: 聊天历史，格式为列表（每条是字典）
        返回：大模型回复内容（字符串）
        """
        url = "https://llxspace.shop/v1/chat/completions"  # LLM服务接口地址

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",   # 带上API key
        }
        payload = {
            "messages": messages,  # 聊天上下文
            "model": "gpt-4o-mini",  # 指定模型
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 1,
            "stream": False,
            "stop": None,
        }

        try:
            with httpx.Client() as client:
                response = client.post(url, headers=headers, json=payload)  # 发送请求
                response.raise_for_status()  # HTTP错误直接抛异常
                data = response.json()
                return data["choices"][0]["message"]["content"]  # 取出大模型回复内容

        except httpx.RequestError as e:
            # 请求失败时记录日志，返回错误提示
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)

            if isinstance(e, httpx.HTTPStatusError):
                status_code = e.response.status_code
                logging.error(f"Status code: {status_code}")
                logging.error(f"Response details: {e.response.text}")

            return (
                f"I encountered an error: {error_message}. "
                "Please try again or rephrase your request."
            )


class ChatSession:
    """主控整个对话流程（用户、LLM、大模型工具）"""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers      # 所有Server实例
        self.llm_client: LLMClient = llm_client   # LLM客户端实例

    async def cleanup_servers(self) -> None:
        """清理所有server资源（逆序清理，防止依赖）"""
        for server in reversed(self.servers):
            try:
                await server.cleanup()
            except Exception as e:
                logging.warning(f"清理异常: {e}")

    async def process_llm_response(self, llm_response: str) -> str:
        """
        处理大模型回复。如果模型让用工具，这里负责调工具并返回结果
        llm_response: LLM的输出（可能是json，也可能是普通话）
        返回：最终结果（工具输出或原始回复）
        """
        import json

        try:
            tool_call = json.loads(llm_response)  # 解析模型输出（期望json）
            if "tool" in tool_call and "arguments" in tool_call:
                logging.info(f"要执行工具: {tool_call['tool']}")
                logging.info(f"参数: {tool_call['arguments']}")

                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(
                                tool_call["tool"], tool_call["arguments"]
                            )

                            # 进度信息特殊处理
                            if isinstance(result, dict) and "progress" in result:
                                progress = result["progress"]
                                total = result["total"]
                                percentage = (progress / total) * 100
                                logging.info(
                                    f"进度: {progress}/{total} ({percentage:.1f}%)"
                                )

                            return f"工具执行结果: {result}"
                        except Exception as e:
                            error_msg = f"工具执行出错: {str(e)}"
                            logging.error(error_msg)
                            return error_msg

                return f"没有找到该工具: {tool_call['tool']}"
            return llm_response   # 不是工具调用，直接返回
        except json.JSONDecodeError:
            return llm_response   # 不是json格式，直接返回

    async def start(self) -> None:
        """整个对话循环主流程"""
        try:
            # 先初始化所有server
            for server in self.servers:
                try:
                    await server.initialize()
                except Exception as e:
                    logging.error(f"server初始化失败: {e}")
                    await self.cleanup_servers()
                    return

            all_tools = []
            for server in self.servers:
                tools = await server.list_tools()
                all_tools.extend(tools)

            # 拼接所有工具描述给LLM
            tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])

            # 系统提示，约束LLM如何调用工具
            system_message = (
                "You are a helpful assistant with access to these tools:\n\n"
                f"{tools_description}\n"
                "Choose the appropriate tool based on the user's question. "
                "If no tool is needed, reply directly.\n\n"
                "IMPORTANT: When you need to use a tool, you must ONLY respond with "
                "the exact JSON object format below, nothing else:\n"
                "{\n"
                '    "tool": "tool-name",\n'
                '    "arguments": {\n'
                '        "argument-name": "value"\n'
                "    }\n"
                "}\n\n"
                "After receiving a tool's response:\n"
                "1. Transform the raw data into a natural, conversational response\n"
                "2. Keep responses concise but informative\n"
                "3. Focus on the most relevant information\n"
                "4. Use appropriate context from the user's question\n"
                "5. Avoid simply repeating the raw data\n\n"
                "Please use only the tools that are explicitly defined above."
            )

            messages = [{"role": "system", "content": system_message}]

            # 主循环：不停等待用户输入
            while True:
                try:
                    user_input = input("You: ").strip().lower()
                    if user_input in ["quit", "exit"]:
                        logging.info("\n退出对话...")
                        break

                    messages.append({"role": "user", "content": user_input})

                    llm_response = self.llm_client.get_response(messages)   # 让LLM回复
                    logging.info("\nAssistant: %s", llm_response)

                    result = await self.process_llm_response(llm_response)  # 判断是否要调工具

                    if result != llm_response:
                        # 工具调用场景，再加一轮
                        messages.append({"role": "assistant", "content": llm_response})
                        messages.append({"role": "system", "content": result})

                        final_response = self.llm_client.get_response(messages)
                        logging.info("\n最终回复: %s", final_response)
                        messages.append(
                            {"role": "assistant", "content": final_response}
                        )
                    else:
                        # 普通聊天，直接回复
                        messages.append({"role": "assistant", "content": llm_response})

                except KeyboardInterrupt:
                    logging.info("\n用户主动退出...")
                    break

        finally:
            await self.cleanup_servers()   # 无论如何都清理


async def main() -> None:
    """初始化并启动整个聊天会话"""
    config = Configuration()  # 加载环境变量、API key等
    server_config = config.load_config("servers_config.json")  # 读取server配置文件
    servers = [
        Server(name, srv_config)
        for name, srv_config in server_config["mcpServers"].items()  # 构建所有Server实例
    ]
    llm_client = LLMClient(config.llm_api_key)  # 创建大模型API客户端
    chat_session = ChatSession(servers, llm_client)  # 创建主对话会话对象
    await chat_session.start()  # 启动主流程，进入对话循环

if __name__ == "__main__":
    # 脚本入口，启动主协程
    asyncio.run(main())
