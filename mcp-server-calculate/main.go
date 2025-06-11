package main

import (
    "context"
    "fmt"

    "github.com/mark3labs/mcp-go/mcp"
    "github.com/mark3labs/mcp-go/server"
)

func main() {
    // 创建一个新的 MCP 服务
    s := server.NewMCPServer(
        "Calculator Demo",      // 服务名
        "1.0.0",                // 版本号
        server.WithToolCapabilities(false), // 关闭 Tool Capabilities
        server.WithRecovery(),              // 开启异常恢复
    )

    // 新建 calculator 工具，支持四则运算
    calculatorTool := mcp.NewTool("calculate",
        mcp.WithDescription("Perform basic arithmetic operations"), // 描述
        mcp.WithString("operation",                                // 操作类型
            mcp.Required(),
            mcp.Description("The operation to perform (add, subtract, multiply, divide)"),
            mcp.Enum("add", "subtract", "multiply", "divide"),
        ),
        mcp.WithNumber("x", mcp.Required(), mcp.Description("First number")), // 第一个数
        mcp.WithNumber("y", mcp.Required(), mcp.Description("Second number")),// 第二个数
    )

    // 注册 calculator handler
    s.AddTool(calculatorTool, func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
        // 解析参数
        op, err := request.RequireString("operation")
        if err != nil {
            return mcp.NewToolResultError(err.Error()), nil
        }
        x, err := request.RequireFloat("x")
        if err != nil {
            return mcp.NewToolResultError(err.Error()), nil
        }
        y, err := request.RequireFloat("y")
        if err != nil {
            return mcp.NewToolResultError(err.Error()), nil
        }

        var result float64
        // 判断操作类型
        switch op {
        case "add":
            result = x + y
        case "subtract":
            result = x - y
        case "multiply":
            result = x * y
        case "divide":
            if y == 0 {
                return mcp.NewToolResultError("cannot divide by zero"), nil // 除数为0报错
            }
            result = x / y
        default:
            return mcp.NewToolResultError("unknown operation"), nil // 未知操作报错
        }

        // 返回结果，保留两位小数
        return mcp.NewToolResultText(fmt.Sprintf("%.2f", result)), nil
    })

    // 启动服务（标准输入输出模式）
    if err := server.ServeStdio(s); err != nil {
        fmt.Printf("Server error: %v\n", err)
    }
}
