import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")

@mcp.tool()
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate BMI given weight in kg and height in meters"""
    return weight_kg / (height_m**2)

@mcp.tool()
async def fetch_weather(city: str) -> str:
    """查询指定城市的天气（支持中文城市名）"""
    async with httpx.AsyncClient() as client:
        try:
            # 使用 wttr.in 接口，格式化输出简洁
            url = f"https://wttr.in/{city}?format=3"
            response = await client.get(url, timeout=5)
            response.raise_for_status()
            return response.text
        except Exception as e:
            return f"无法获取天气: {e}"

if __name__ == "__main__":
    mcp.run()